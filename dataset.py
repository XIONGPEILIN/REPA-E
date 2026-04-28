"""This files contains training loss implementation.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
import io
import json
import os
import glob


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
from torchvision import transforms
import datasets


def _load_label_map(label_map_path):
    if label_map_path is None:
        return None
    with open(label_map_path, 'r') as f:
        raw_map = json.load(f)
    if isinstance(raw_map, list):
        return {i: int(v) for i, v in enumerate(raw_map)}
    if isinstance(raw_map, dict):
        return {int(k): int(v) for k, v in raw_map.items()}
    raise ValueError(f"Unsupported label map format: {type(raw_map)}")


def _decode_hf_image_field(image_field):
    if isinstance(image_field, PIL.Image.Image):
        return image_field
    if isinstance(image_field, dict):
        image_bytes = image_field.get('bytes', None)
        if image_bytes is not None:
            return PIL.Image.open(io.BytesIO(image_bytes))
        image_path = image_field.get('path', None)
        if image_path is not None:
            return PIL.Image.open(image_path)
    if isinstance(image_field, np.ndarray):
        return PIL.Image.fromarray(image_field)
    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def _ensure_uint8_chw(tensor):
    if tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    return tensor


def load_h5_file(hf, path):
    # Helper function to load files from h5 file
    if path.endswith('.png'):
        rtn = np.array(PIL.Image.open(io.BytesIO(np.array(hf[path]))))
        rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)
    elif path.endswith('.json'):
        rtn = json.loads(np.array(hf[path]).tobytes().decode('utf-8'))
    elif path.endswith('.npy'):
        rtn= np.array(hf[path])
    else:
        raise ValueError('Unknown file type: {}'.format(path))
    return rtn


class ParquetDataset(Dataset):
    def __init__(self, data_dir, resolution=256, label_map_path=None):
        self.resolution = resolution
        self.label_map = _load_label_map(label_map_path)
        
        # Use glob to find all training parquet files
        data_files = glob.glob(os.path.join(data_dir, 'train-*.parquet'))
        
        # Load the dataset using Hugging Face datasets
        self.dataset = datasets.load_dataset('parquet', data_files=data_files, split='train')

        # Define the transformation
        # The training loop expects a uint8 tensor [C, H, W] in range [0, 255]
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(self.resolution),
            transforms.Lambda(lambda x: np.array(x).transpose(2, 0, 1)), # HWC -> CHW
            transforms.Lambda(lambda x: torch.from_numpy(x)),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item['image']
        label = item['label']
        if self.label_map is not None:
            label = self.label_map[int(label)]

        # Parquet image payload can be bytes dict, PIL image, or ndarray.
        pil_image = _decode_hf_image_field(image)
        
        # Apply transformations
        transformed_image = _ensure_uint8_chw(self.transform(pil_image))
        
        return transformed_image, torch.tensor(label, dtype=torch.long)


class HFImageNetDataset(Dataset):
    def __init__(self, cache_dir=None, resolution=256, label_map_path=None):
        self.resolution = resolution
        self.label_map = _load_label_map(label_map_path)
        self.dataset = datasets.load_dataset(
            "ILSVRC/imagenet-1k",
            split="train",
            cache_dir=cache_dir,
        )
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(self.resolution),
            transforms.Lambda(lambda x: np.array(x).transpose(2, 0, 1)),
            transforms.Lambda(lambda x: torch.from_numpy(x)),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item["image"]
        label = item["label"]
        if self.label_map is not None:
            label = self.label_map[int(label)]
        pil_image = _decode_hf_image_field(image)
        transformed_image = _ensure_uint8_chw(self.transform(pil_image))
        return transformed_image, torch.tensor(label, dtype=torch.long)


class CustomINH5Dataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "images.h5")
        self.h5_json_path = os.path.join(self.data_dir, "images_h5.json")
        
        if not os.path.exists(self.h5_path) or not os.path.exists(self.h5_json_path):
            raise FileNotFoundError(f"HDF5 dataset not found at {self.data_dir}. Please run preprocessing.py first.")

        self.h5f = h5py.File(self.h5_path, 'r')

        with open(self.h5_json_path, 'r') as f:
            self.h5_json = json.load(f)
        self.filelist = {fname for fname in self.h5_json}
        self.filelist = sorted(fname for fname in self.filelist if self._file_ext(fname) in supported_ext)

        labels = load_h5_file(self.h5f, "dataset.json")["labels"]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.filelist]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def __len__(self):
        return len(self.filelist)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __del__(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()

    def __getitem__(self, index):
        """
        Images should be '.png'
        """
        image_fname = self.filelist[index]
        image = load_h5_file(self.h5f, image_fname)
        return torch.from_numpy(image), torch.tensor(self.labels[index])


class CustomH5Dataset(Dataset):
    def __init__(self, data_dir, vae_latents_name="repae-invae-400k"):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_h5 = h5py.File(os.path.join(data_dir, 'images.h5'), "r")
        self.features_h5 = h5py.File(os.path.join(data_dir, f'{vae_latents_name}.h5'), "r")
        images_json = os.path.join(data_dir, 'images_h5.json')
        features_json = os.path.join(data_dir, f'{vae_latents_name}_h5.json')

        with open(images_json, 'r') as f:
            images_json = json.load(f)
        with open(features_json, 'r') as f:
            features_json = json.load(f)

        # images
        self._image_fnames = {fname for fname in images_json}
        self.image_fnames = sorted(fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext)

        # features
        self._feature_fnames = {fname for fname in features_json}
        self.feature_fnames = sorted(fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext)
        
        # labels
        fname = 'dataset.json'
        labels = load_h5_file(self.features_h5, fname)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __del__(self):
        self.images_h5.close()
        self.features_h5.close()

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)

        image = load_h5_file(self.images_h5, image_fname)
        if image_ext == '.npy':
            # npy needs some extra care
            image = image.reshape(-1, *image.shape[-2:])

        features = load_h5_file(self.features_h5, feature_fname)
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])
