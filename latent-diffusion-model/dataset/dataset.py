import os

import torchio as tio
from torch.utils.data import Dataset

PREPROCESSING_TRANSORMS = tio.Compose(
    [tio.ToCanonical(), tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(1, 99))]
)

TRAIN_TRANSFORMS = tio.Compose(
    [
        tio.RandomFlip(axes=["inferior-superior"], flip_probability=0.5),  # type: ignore
    ]
)


class MRIDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        nifti_file_names = os.listdir(self.root_dir)
        folder_names = [
            os.path.join(self.root_dir, nifti_file_name)
            for nifti_file_name in nifti_file_names
            if nifti_file_name.endswith(".nii")
        ]
        return folder_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.file_paths[idx])
        img = self.preprocessing(img)
        img = self.transforms(img)
        return img.data  # type: ignore
