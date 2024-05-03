import torch
import torch.nn.functional as F
from .codebook import Codebook
from .decoder import Decoder
from .discriminator import NLayerDiscriminator2D, NLayerDiscriminator3D
from .encoder import Encoder
from .lpips import LPIPS
from .same_pad_conv_3d import SamePadConv3D
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from .utils import adopt_weight, shift_dim


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


discriminator_losses = {"vanilla": vanilla_d_loss, "hinge": hinge_d_loss}


class VQGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.global_step = 0

        self.number_of_hiddens = cfg.model.number_of_hiddens
        self.downsample = cfg.model.downsample
        self.image_channels = cfg.dataset.image_channels
        self.norm_type = cfg.model.norm_type
        self.padding_type = cfg.model.padding_type
        self.num_groups = cfg.model.num_groups

        self.encoder = Encoder(
            self.number_of_hiddens,
            self.downsample,
            self.image_channels,
            norm_type=self.norm_type,
            padding_type=self.padding_type,
            num_groups=self.num_groups,
        )
        self.encoder_output_channels = self.encoder.out_channels

        self.decoder = Decoder(
            self.number_of_hiddens,
            self.downsample,
            self.image_channels,
            norm_type=self.norm_type,
            num_groups=self.num_groups,
        )

        self.embedding_dim = cfg.model.embedding_dim
        self.number_of_codes = cfg.model.number_of_codes
        self.no_random_restart = cfg.model.no_random_restart
        self.restart_threshold = cfg.model.restart_threshold

        self.codebook = Codebook(
            self.number_of_codes,
            self.embedding_dim,
            no_random_restart=self.no_random_restart,
            restart_thres=self.restart_threshold,
        )

        self.pre_vq_conv = SamePadConv3D(
            self.encoder_output_channels,
            self.embedding_dim,
            kernel_size=1,
            padding_type=self.padding_type,
        )
        self.post_vq_conv = SamePadConv3D(
            self.embedding_dim,
            self.encoder_output_channels,
            kernel_size=1,
            padding_type=self.padding_type,
        )

        self.gan_feat_weight = cfg.model.weights.gan_feat
        self.discriminator_channels = cfg.model.discriminator.channels
        self.discriminator_layers = cfg.model.discriminator.layers

        self.image_discriminator = NLayerDiscriminator2D(
            self.image_channels,
            self.discriminator_channels,
            self.discriminator_layers,
            norm_layer=nn.BatchNorm2d,
        )
        self.video_discriminator = NLayerDiscriminator3D(
            self.image_channels,
            self.discriminator_channels,
            self.discriminator_layers,
            norm_layer=nn.BatchNorm3d,
        )

        discriminator_loss_type = cfg.model.discriminator.loss_type
        self.discriminator_loss = discriminator_losses[discriminator_loss_type]

        self.perceptual_model = LPIPS().eval()
        self.perceptual_weight = cfg.model.weights.perceptual

        self.image_gan_weight = cfg.model.weights.image_gan
        self.video_gan_weight = cfg.model.weights.video_gan

        self.l1_weight = cfg.model.weights.l1
        self.discriminator_iter_start = cfg.model.discriminator.iter_start
        # self.save_hyperparameters()

    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))

        if quantize:
            quantized = self.codebook(h)

            if include_embeddings:
                return quantized["embeddings"], quantized["encodings"]
            
            return quantized["encodings"]
        
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            quantized = self.codebook(latent)
            latent = quantized["encodings"]

        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(
        self, x, optimizer_idx=None, epoch=0, writer: SummaryWriter | None = None
    ):
        self.global_step += 1
        B, C, T, H, W = x.shape

        z = self.pre_vq_conv(self.encoder(x))
        quantized = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(quantized["embeddings"]))

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # Selects one random 2D image from each 3D image
        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if optimizer_idx == 0:
            ae_loss, perceptual_loss, gan_feat_loss = self.train_generator(
                x, x_recon, frames, frames_recon, writer, epoch
            )

            if writer:
                writer.add_scalar("train/recon_loss", recon_loss, epoch)
                writer.add_scalar(
                    "train/commitment_loss", quantized["commitment_loss"], epoch
                )
                writer.add_scalar("train/perplexity", quantized["perplexity"], epoch)

            return (
                recon_loss,
                x_recon,
                quantized,
                ae_loss,
                perceptual_loss,
                gan_feat_loss,
            )

        if optimizer_idx == 1:
            discriminator_loss = self.train_discriminator(
                x, x_recon, frames, frames_recon, writer, epoch
            )
            return discriminator_loss

        perceptual_loss = (
            self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
        )
        return recon_loss, x_recon, quantized, perceptual_loss

    def train_generator(
        self, x, x_recon, frames, frames_recon, writer: SummaryWriter | None, epoch
    ):
        # Perceptual loss
        perceptual_loss = 0
        if self.perceptual_weight > 0:
            perceptual_loss = (
                self.perceptual_model(frames, frames_recon).mean()
                * self.perceptual_weight
            )

        # Discriminator loss
        logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
        logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
        generator_image_loss = -torch.mean(logits_image_fake)
        generator_video_loss = -torch.mean(logits_video_fake)
        generator_loss = (
            self.image_gan_weight * generator_image_loss
            + self.video_gan_weight * generator_video_loss
        )
        discriminator_factor = adopt_weight(
            self.global_step, threshold=self.discriminator_iter_start
        )
        ae_loss = discriminator_factor * generator_loss

        # GAN feature matching loss
        image_gan_feat_loss = 0
        video_gan_feat_loss = 0

        if self.image_gan_weight > 0:
            _, pred_image_real = self.image_discriminator(frames)

            for i in range(len(pred_image_fake) - 1):
                image_gan_feat_loss += (
                    F.l1_loss(pred_image_fake[i], pred_image_real[i].detach())
                    * self.image_gan_weight
                )

        if self.video_gan_weight > 0:
            _, pred_video_real = self.video_discriminator(x)

            for i in range(len(pred_video_fake) - 1):
                video_gan_feat_loss += (
                    F.l1_loss(pred_video_fake[i], pred_video_real[i].detach())
                    * self.video_gan_weight
                )

        gan_feat_loss = (
            discriminator_factor
            * self.gan_feat_weight
            * (image_gan_feat_loss + video_gan_feat_loss)
        )

        if writer:
            writer.add_scalar("train/ae_loss", ae_loss, epoch)
            writer.add_scalar("train/perceptual_loss", perceptual_loss, epoch)
            writer.add_scalar("train/generator_image_loss", generator_image_loss, epoch)
            writer.add_scalar("train/generator_video_loss", generator_video_loss, epoch)
            writer.add_scalar("train/image_gan_feat_loss", image_gan_feat_loss, epoch)
            writer.add_scalar("train/video_gan_feat_loss", video_gan_feat_loss, epoch)

        return ae_loss, perceptual_loss, gan_feat_loss

    def train_discriminator(
        self, x, x_recon, frames, frames_recon, writer: SummaryWriter | None, epoch
    ):
        logits_image_real, _ = self.image_discriminator(frames.detach())
        logits_video_real, _ = self.video_discriminator(x.detach())

        logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
        logits_video_fake, _ = self.video_discriminator(x_recon.detach())

        discriminator_image_loss = self.discriminator_loss(
            logits_image_real, logits_image_fake
        )
        discriminator_video_loss = self.discriminator_loss(
            logits_video_real, logits_video_fake
        )
        discriminator_factor = adopt_weight(
            self.global_step, threshold=self.discriminator_iter_start
        )
        discriminator_loss = discriminator_factor * (
            self.image_gan_weight * discriminator_image_loss
            + self.video_gan_weight * discriminator_video_loss
        )

        if writer:
            writer.add_scalar(
                "train/logits_image_real", logits_image_real.mean().detach(), epoch
            )
            writer.add_scalar(
                "train/logits_image_fake", logits_image_fake.mean().detach(), epoch
            )
            writer.add_scalar(
                "train/logits_video_real", logits_video_real.mean().detach(), epoch
            )
            writer.add_scalar(
                "train/logits_video_fake", logits_video_fake.mean().detach(), epoch
            )
            writer.add_scalar(
                "train/discriminator_image_loss", discriminator_image_loss, epoch
            )
            writer.add_scalar(
                "train/discriminator_video_loss", discriminator_video_loss, epoch
            )
            writer.add_scalar("train/discriminator_loss", discriminator_loss, epoch)

        return discriminator_loss
