import torch
from tqdm import tqdm

from .gaussian_diffusion import GaussianDiffusion


class Sampler:
    def __init__(self, model: GaussianDiffusion):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        shape,
        mask=None,
        x0=None,
        init_image=None,
    ):
        # Sampling
        C, H, W, D = shape
        size = (batch_size, C, H, W, D)
        print(f"Data shape for sampling is {size}")

        sample = self.ddim_sampling(
            size,
            mask=mask,
            x0=x0,
            init_image=init_image,
        )

        return sample

    @torch.no_grad()
    def ddim_sampling(
        self,
        shape,
        x_T=None,
        timesteps=None,
        mask=None,
        x0=None,
        init_image=None,
    ):
        device = self.model.betas.device
        b = shape[0]

        timesteps = self.ddpm_num_timesteps
        time_range = list(reversed(range(0, timesteps)))

        print(f"Running Sampling with {timesteps} timesteps")

        if init_image is not None:
            assert (
                x0 is None and x_T is None
            ), "Try to infer x0 and x_T from init_image, but they are already provided"

            x0 = self.model.vqgan.encode(init_image, quantize=False)

            # Normalize to -1 and 1
            x0 = (
                (x0 - self.model.vqgan.codebook.embeddings.min())
                / (
                    self.model.vqgan.codebook.embeddings.max()
                    - self.model.vqgan.codebook.embeddings.min()
                )
            ) * 2 - 1

            img = torch.randn(shape, device=device)
        elif x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        iterator = tqdm(time_range, desc="Sampler", total=timesteps)  # type: ignore
        for step in iterator:
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Latent level blending
            if mask is not None:
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1 - mask) * img

            img = self.model.p_sample(img, ts)

        return img
