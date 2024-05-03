import os
import pathlib
from datetime import datetime

import hydra
import torch
import torchio as tio
from dataset import MRIDataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from vqgan.vqgan import VQGAN

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


@hydra.main(version_base=None, config_path="../config", config_name="vqgan_config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    base_path = os.path.join(ABS_PATH, "results", now)
    models_path = os.path.join(base_path, "models")
    training_path = os.path.join(base_path, "images", "training")
    test_path = os.path.join(base_path, "images", "test")

    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(models_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(training_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(base_path, "config.yaml"), "x") as f:
        f.write(OmegaConf.to_yaml(cfg))

    writer = SummaryWriter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    full_dataset = MRIDataset(cfg.dataset.root_dir)
    train_dataset, validation_dataset, test_dataset = random_split(
        full_dataset, [0.8, 0.1, 0.1]
    )

    batch_size = cfg.model.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = VQGAN(cfg)
    model.to(device)

    optimizer_ae = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.pre_vq_conv.parameters())
        + list(model.post_vq_conv.parameters())
        + list(model.codebook.parameters()),
        lr=cfg.model.learning_rate,
        betas=(0.5, 0.9),
    )
    optimizer_discriminator = torch.optim.Adam(
        list(model.image_discriminator.parameters())
        + list(model.video_discriminator.parameters()),
        lr=cfg.model.learning_rate,
        betas=(0.5, 0.9),
    )

    print("Train model")
    best_loss = 999999999999
    for epoch in tqdm(range(cfg.epochs)):
        model.train()

        train_epoch_loss = 0
        train_generator_loss = 0
        train_discriminator_loss = 0
        train_recon_loss = 0
        train_commitment_loss = 0
        train_perceptual_loss = 0

        validation_epoch_loss = 0
        validation_recon_loss = 0
        validation_commitment_loss = 0
        validation_perceptual_loss = 0

        for data in tqdm(train_dataloader):
            data = data.to(device)

            # Generator
            optimizer_ae.zero_grad()
            recon_loss, x_recon, quantized, ae_loss, perceptual_loss, gan_feat_loss = (
                model(data, 0, epoch + 1, writer)
            )
            commitment_loss = quantized["commitment_loss"]
            generator_loss = (
                recon_loss + commitment_loss + ae_loss + perceptual_loss + gan_feat_loss
            )
            epoch_loss = recon_loss + commitment_loss + perceptual_loss

            train_epoch_loss += epoch_loss.item()
            train_generator_loss += generator_loss.item()
            train_recon_loss += recon_loss.item()
            train_commitment_loss += commitment_loss.item()
            train_perceptual_loss += perceptual_loss.item()

            generator_loss.backward()
            optimizer_ae.step()

            # Discriminator
            optimizer_discriminator.zero_grad()
            discriminator_loss = model(data, 1, epoch + 1, writer)

            train_discriminator_loss += discriminator_loss.item()

            discriminator_loss.backward()
            optimizer_discriminator.step()

        model.eval()
        with torch.no_grad():
            for data in tqdm(validation_dataloader):
                data = data.to(device)

                recon_loss, x_recon, quantized, perceptual_loss = model(data)
                commitment_loss = quantized["commitment_loss"]
                loss = recon_loss + commitment_loss + perceptual_loss

                validation_epoch_loss += loss.item()
                validation_recon_loss += recon_loss.item()
                validation_commitment_loss += commitment_loss.item()
                validation_perceptual_loss += perceptual_loss.item()

        train_epoch_loss /= len(train_dataset)
        train_generator_loss /= len(train_dataset)
        train_discriminator_loss /= len(train_dataset)
        train_recon_loss /= len(train_dataset)
        train_commitment_loss /= len(train_dataset)
        train_perceptual_loss /= len(train_dataset)

        validation_epoch_loss /= len(validation_dataset)
        validation_recon_loss /= len(validation_dataset)
        validation_commitment_loss /= len(validation_dataset)
        validation_perceptual_loss /= len(validation_dataset)

        print(
            f"Epoch {epoch + 1}, Training Loss {train_epoch_loss:.4f}, Validation Loss {validation_epoch_loss:.4f}, Generator Loss {train_generator_loss:.4f}, Discriminator Loss {train_discriminator_loss}"
        )

        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": train_epoch_loss, "Validation": validation_epoch_loss},
            epoch + 1,
        )

        writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch + 1)
        writer.add_scalar("train/recon_loss", train_recon_loss, epoch + 1)
        writer.add_scalar("train/commitment_loss", train_commitment_loss, epoch + 1)
        writer.add_scalar("train/perceptual_loss", train_perceptual_loss, epoch + 1)

        writer.add_scalar("validation/epoch_loss", validation_epoch_loss, epoch + 1)
        writer.add_scalar("validation/recon_loss", validation_recon_loss, epoch + 1)
        writer.add_scalar(
            "validation/commitment_loss", validation_commitment_loss, epoch + 1
        )
        writer.add_scalar(
            "validation/perceptual_loss", validation_perceptual_loss, epoch + 1
        )

        torch.save(model.state_dict(), os.path.join(models_path, "latest_model.pth"))
        if validation_epoch_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(models_path, "best_model.pth"))
            best_loss = validation_epoch_loss

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), os.path.join(models_path, f"model_{epoch + 1}.pth")
            )
            tio.ScalarImage(tensor=data[0].cpu().detach()).save(os.path.join(training_path, f"{epoch + 1}_original.nii"))  # type: ignore
            tio.ScalarImage(tensor=x_recon[0].cpu().detach()).save(os.path.join(training_path, f"{epoch + 1}_reconstructed.nii"))  # type: ignore

    print("Test model")
    model.load_state_dict(torch.load(os.path.join(models_path, "best_model.pth")))
    model.eval()
    with torch.no_grad():
        test_loss = 0
        total_recon_loss = 0
        total_commitment_loss = 0
        total_perceptual_loss = 0

        for index, data in tqdm(enumerate(test_dataloader)):
            data = data.to(device)

            recon_loss, x_recon, quantized, perceptual_loss = model(data)
            commitment_loss = quantized["commitment_loss"]
            loss = recon_loss + commitment_loss + perceptual_loss

            test_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_commitment_loss += commitment_loss.item()
            total_perceptual_loss += perceptual_loss.item()

            if index % 50 == 0:
                tio.ScalarImage(tensor=data[0].cpu().detach()).save(
                    os.path.join(test_path, f"{index}_original.nii")
                )
                tio.ScalarImage(tensor=x_recon[0].cpu().detach()).save(
                    os.path.join(test_path, f"{index}_reconstructed.nii")
                )

        test_loss /= len(test_dataset)
        total_recon_loss /= len(test_dataset)
        total_commitment_loss /= len(test_dataset)
        total_perceptual_loss /= len(test_dataset)

        print(f"Test loss: {test_loss:.4f}")
        writer.add_scalar("test/test_loss", test_loss)
        writer.add_scalar("test/recon_loss", total_recon_loss)
        writer.add_scalar("test/commitment_loss", total_commitment_loss)
        writer.add_scalar("test/perceptual_loss", total_perceptual_loss)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    train()
