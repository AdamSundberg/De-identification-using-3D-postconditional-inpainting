import logging
import os

import numpy as np
import pywavefront
import torch
import torchio as tio
from pywavefront import Wavefront
from torch.utils.data import Dataset
from tqdm import tqdm

pywavefront.configure_logging(
    logging.ERROR,
)

PREPROCESSING_TRANSORMS = tio.Compose(
    [tio.ToCanonical(), tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(1, 99))]
)

TRAIN_TRANSFORMS = tio.Compose(
    [
        tio.RandomFlip(axes=["inferior-superior"], flip_probability=0.5),  # type: ignore
    ]
)


class VolumeDataset(Dataset):
    def __init__(self, faces_root_dir: str | None = None, no_faces_root_dir: str | None = None):
        super().__init__()
        self.faces_root_dir = faces_root_dir
        self.no_faces_root_dir = no_faces_root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.volumes_file_path, self.labels = self.get_data_files()

    def get_data_files(self):
        if self.faces_root_dir:
            faces_files = os.listdir(self.faces_root_dir)
            faces = [
                os.path.join(self.faces_root_dir, file_name)
                for file_name in faces_files
                if file_name.endswith(".nii") or file_name.endswith(".nii.gz")
            ]
        else:
            faces = []
        
        if self.no_faces_root_dir:
            no_faces_files = os.listdir(self.no_faces_root_dir)
            no_faces = [
                os.path.join(self.no_faces_root_dir, file_name)
                for file_name in no_faces_files
                if file_name.endswith(".nii") or file_name.endswith(".nii.gz")
            ]
        else:
            no_faces = []

        volumes = np.concatenate((faces, no_faces))
        labels = np.concatenate((np.ones(len(faces)), np.zeros(len(no_faces))))

        return volumes, labels

    def __len__(self):
        return len(self.volumes_file_path)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.volumes_file_path[idx])
        img = self.preprocessing(img)
        img = self.transforms(img)

        label = self.labels[idx]
        return img.data, label  # type: ignore


class MeshDataset(Dataset):
    def __init__(self, faces_root_dir: str, no_faces_root_dir: str, max_length=None):
        super().__init__()
        self.faces_root_dir = faces_root_dir
        self.no_faces_root_dir = no_faces_root_dir
        self.volumes_file_path, self.labels = self.get_data_files()
        if not max_length:
            self.max_length = self.calculate_max_length()
        else:
            self.max_length = max_length
        print("Max length", self.max_length)

    def get_data_files(self):
        faces_files = os.listdir(self.faces_root_dir)
        faces = [
            os.path.join(self.faces_root_dir, file_name)
            for file_name in faces_files
            if file_name.endswith(".obj")
        ]

        no_faces_files = os.listdir(self.no_faces_root_dir)
        no_faces = [
            os.path.join(self.no_faces_root_dir, file_name)
            for file_name in no_faces_files
            if file_name.endswith(".obj")
        ]

        volumes = np.concatenate((faces, no_faces))
        labels = np.concatenate((np.ones(len(faces)), np.zeros(len(no_faces))))

        return volumes, labels

    def calculate_max_length(self):
        return max(
            [
                len(Wavefront(file).vertices)
                for file in tqdm(self.volumes_file_path, desc="Calculate max length")
            ]
        )

    def __len__(self):
        return len(self.volumes_file_path)

    def __getitem__(self, idx: int):
        mesh = Wavefront(self.volumes_file_path[idx])
        vertices = torch.tensor(mesh.vertices)

        if len(vertices) > self.max_length:
            vertices = vertices[
                : self.max_length
            ]  # Truncate sequence if longer than max_length
        elif len(vertices) < self.max_length:
            # Pad sequence with zeros if shorter than max_length
            pad_length = self.max_length - len(vertices)
            vertices = torch.cat(
                [vertices, torch.zeros(pad_length, vertices.size(1))], dim=0
            )

        label = self.labels[idx]
        return vertices, label
