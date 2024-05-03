import os
import pathlib
from datetime import datetime

import hydra
import numpy as np
import torch
import torchio as tio
from ddpm.sampler import Sampler
from ddpm.gaussian_diffusion import GaussianDiffusion
from ddpm.unet_3d import Unet3D
from omegaconf import DictConfig, OmegaConf
from resize_right import interp_methods, resize
from torch import nn
from tqdm import tqdm
from vqgan.vqgan import VQGAN

RESCALE_INTENSITY = tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(1, 99))

PREPROCESSING_TRANSORMS = tio.Compose(
    [tio.ToCanonical(), tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(1, 99))]
)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

TOOL = "fsl"


def read_mask(
    org_mask,
    device,
    dest_size=(32, 32, 32),
    img_size=(256, 256, 256),
):
    mask = resize(org_mask, out_shape=dest_size, interp_method=interp_methods.linear)
    mask = mask.to(device)  # type: ignore

    org_mask = resize(
        org_mask, out_shape=img_size, interp_method=interp_methods.lanczos3
    )
    org_mask = np.array(org_mask).astype(np.float32)
    org_mask = org_mask[None]
    org_mask[org_mask < 0.5] = 0
    org_mask[org_mask >= 0.5] = 1
    org_mask = torch.from_numpy(org_mask).to(device)

    return mask, org_mask


@hydra.main(version_base=None, config_path="../config", config_name="sampler_config")
def run(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    base_path = os.path.join(ABS_PATH, "results", "sampler", f"{TOOL}_{now}")

    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(base_path, "config.yaml"), "x") as f:
        f.write(OmegaConf.to_yaml(cfg))

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
    unet.eval()
    unet.to(device)

    gaussian_model = GaussianDiffusion(
        unet,
        vqgan=vqgan,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    )
    gaussian_model.eval()
    gaussian_model = nn.DataParallel(gaussian_model)

    ddpm_path = cfg.ddpm_path
    gaussian_model.load_state_dict(torch.load(os.path.join(ddpm_path, "models", "best_model.pth")))
    gaussian_model.to(device)

    sampler = Sampler(gaussian_model.module)

    channels = cfg.model.diffusion_num_channels
    num_frames = cfg.model.diffusion_depth_size
    image_size = cfg.model.diffusion_img_size
    shape = (channels, num_frames, image_size, image_size)

    nifti_file_names = os.listdir(os.path.join(cfg.dataset.root_dir, "original"))
    file_paths = [
        {
            "name": nifti_file_name.split(".")[0],
            "original": os.path.join(cfg.dataset.root_dir, "original", nifti_file_name),
            "mask": os.path.join(cfg.dataset.root_dir, f"{TOOL}/masks", nifti_file_name),
            "defaced": os.path.join(cfg.dataset.root_dir, f"{TOOL}/defaced", nifti_file_name)
        }
        for nifti_file_name in nifti_file_names
        if nifti_file_name.endswith(".nii") or nifti_file_name.endswith(".nii.gz")
    ]

    sample_batch_size = cfg.sample_batch_size
    for i, file_path in tqdm(enumerate(file_paths), desc="Files", total=len(file_paths)):
        result_path = os.path.join(base_path, f"sample_{i}")
        pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

        init_image_orig = tio.ScalarImage(file_path["original"])
        init_image_orig = PREPROCESSING_TRANSORMS(init_image_orig)

        init_image = tio.ScalarImage(file_path["original"])
        init_image = PREPROCESSING_TRANSORMS(init_image)
        init_image.save(os.path.join(result_path, f"{file_path["name"]}.nii"))

        init_image = init_image.data  # type: ignore
        init_image = init_image.to(device)  # type: ignore
        init_image = init_image.unsqueeze(0)

        defaced = tio.ScalarImage(file_path["defaced"])
        defaced = PREPROCESSING_TRANSORMS(defaced)
        defaced.save(os.path.join(result_path, f"{file_path["name"]}_defaced.nii"))
        
        mask = tio.ScalarImage(file_path["mask"])
        mask.save(os.path.join(result_path, f"{file_path["name"]}_mask.nii"))
        mask = mask.data

        mask, org_mask = read_mask(
            org_mask=mask,
            device=device,
            dest_size=(32, 32, 32),
            img_size=(64, 64, 64),
        )

        # Load init image and mask
        samples_ddim = sampler.sample(
            batch_size=sample_batch_size,
            shape=shape,
            init_image=init_image,
            mask=mask,
        )
        samples_ddim = (
            ((samples_ddim + 1) / 2)
            * (
                gaussian_model.module.vqgan.codebook.embeddings.max()
                - gaussian_model.module.vqgan.codebook.embeddings.min()
            )
        ) + gaussian_model.module.vqgan.codebook.embeddings.min()

        x_samples = gaussian_model.module.vqgan.decode(samples_ddim, quantize=True)
        x_samples = x_samples * (1 - org_mask) + init_image * (org_mask)

        for i in range(sample_batch_size):
            init_image_orig.set_data(x_samples[i].cpu().detach())
            init_image_orig.save(os.path.join(result_path, f"{file_path["name"]}_sample_{i}.nii"))

            # tio.ScalarImage(tensor=x_samples[i].cpu().detach()).save(os.path.join(result_path, f"{file_path["name"]}_sample_{i}.nii"))


if __name__ == "__main__":
    run()
