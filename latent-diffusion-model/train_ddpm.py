import os
import pathlib
from datetime import datetime

import hydra
import torch
import torchio as tio
from dataset.dataset import MRIDataset
from ddpm.gaussian_diffusion import GaussianDiffusion
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from ddpm.unet_3d import Unet3D
from vqgan.vqgan import VQGAN

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


def save_model(model, scaler, models_path, name):
    torch.save(model.state_dict(), os.path.join(models_path, f"{name}_model.pth"))
    torch.save(
        scaler.state_dict(), os.path.join(models_path, f"{name}_scaler_model.pth")
    )


@hydra.main(version_base=None, config_path="../config", config_name="ddpm_config")
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
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    vqgan_path = cfg.vqgan_path
    vqgan_conf = OmegaConf.load(os.path.join(vqgan_path, "config.yaml"))
    vqgan = VQGAN(vqgan_conf)
    vqgan.to(device)
    vqgan.load_state_dict(torch.load(os.path.join(vqgan_path, "models", "best_model.pth")))
    vqgan.eval()
    
    unet = Unet3D(
        dim=cfg.model.diffusion_img_size,
        dimension_mults=cfg.model.dimension_mults,
        channels=cfg.model.diffusion_num_channels,
    )
    unet.to(device)

    model = GaussianDiffusion(
        unet,
        vqgan=vqgan,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    )
    model = nn.DataParallel(model)
    model.to(device)


    optimizer = Adam(model.parameters(), lr=cfg.model.learning_rate)
    scaler = GradScaler(enabled=cfg.model.amp)

    print("Train model")
    step_counter = 0
    best_loss = 99999999999999
    for epoch in tqdm(range(cfg.epochs)):
        model.train()

        train_epoch_loss = 0
        validation_epoch_loss = 0

        for data in tqdm(train_dataloader):
            data = data.to(device)

            optimizer.zero_grad()

            with autocast(enabled=cfg.model.amp):
                loss = model(data, prob_focus_present=0, focus_present_mask=None).sum()
                train_epoch_loss += loss.item()

                scaler.scale(loss / batch_size).backward()

            scaler.step(optimizer)
            scaler.update()

            step_counter += 1

        model.eval()
        with torch.no_grad():
            for data in tqdm(validation_dataloader):
                data = data.to(device)

                with autocast(enabled=cfg.model.amp):
                    loss = model(data, prob_focus_present=0, focus_present_mask=None).sum()
                    validation_epoch_loss += loss.item()

        train_epoch_loss /= len(train_dataset)
        validation_epoch_loss /= len(validation_dataset)

        print(
            f"Epoch {epoch + 1}, Training Loss {train_epoch_loss:.4f}, Validation Loss {validation_epoch_loss:.4f}"
        )

        writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch + 1)
        writer.add_scalar("validation/epoch_loss", validation_epoch_loss, epoch + 1)
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": train_epoch_loss, "Validation": validation_epoch_loss},
            epoch + 1,
        )

        save_model(model, scaler, models_path, "latest")
        if validation_epoch_loss < best_loss:
            save_model(model, scaler, models_path, "best")
            best_loss = validation_epoch_loss

        if epoch % 10 == 0:
            save_model(model, scaler, models_path, f"{epoch + 1}_epoch")

            # Sample and save an example image
            sample = model.module.sample(batch_size=1)
            tio.ScalarImage(tensor=sample[0].cpu().detach()).save(
                os.path.join(training_path, f"{epoch + 1}_sample.nii")
            )

    print("Test model")
    model.load_state_dict(torch.load(os.path.join(models_path, "best_model.pth")))
    model.eval()
    with torch.no_grad():
        test_loss = 0

        for index, data in tqdm(enumerate(test_dataloader)):
            data = data.to(device)
            with autocast(enabled=cfg.model.amp):
                loss = model(data, prob_focus_present=0, focus_present_mask=None).sum()
                test_loss += loss.item()

            if index % 10 == 0:
                # Sample and save an example image
                sample = model.module.sample(batch_size=1)
                tio.ScalarImage(tensor=sample[0].cpu().detach()).save(
                    os.path.join(test_path, f"{index}_sample.nii")
                )

        test_loss /= len(test_dataset)

        print(f"Test loss: {test_loss:.4f}")
        writer.add_scalar("test/test_loss", test_loss)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    train()
