# De-identification using 3D postconditional inpainting

This is the repository for the paper "Automatic De-Identification of Magnetic Resonance Images".

## Requirements

- Anaconda

## Installation

- conda env create -f environment.yml
- conda activate tqdt33

## Training

The system consists of two models: a VQ-VAE and an LDM. Firstly the VQ-VAE is trained and then the LDM is trained.

- Fill in the correct information in `config`. For the VQ-VAE it is the file called `vqgan_config.yaml` and for the LDM
the file is called `ddpm_config.yaml`. Check `config/README.md` for more information regarding the config files.
- Start training the VQ-VAE by running the python script `latent-diffusion-model/train_vqgan.py`.
- Start training the LDM by running the python script `latent-diffusion-model/train_ddpm.py`.

The result from the trainings get stored in `results/[TRAINING-DATE]`. The best and latest model is saved together with
one for each 10:th epoch. 

## Generate inpainted samples

- Fill in correct information in `config/sampler_config.yaml`
- Run the python script `latent-diffusion-model/generate_samples.py`

For this, the data structure has to follow:
.
└── root_dir/
    ├── original/
    │   └── ... IMAGE FILES
    ├── MASK_TOOL_1/
    │   ├── masks/
    │   │   └── ... MASK FILES
    │   └── defaced/
    │       └── ... DEFACED FILES
    └── MASK_TOOL_2/
        ├── masks/
        │   └── ... MASK FILES
        └── defaced/
            └── ... DEFACED FILES
