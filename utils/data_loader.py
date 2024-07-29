import json
import traceback
import typing

import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset
from monai.data import DataLoader as mDataLoader
from pathlib import Path


class DatasetMerged(Dataset):
    def __init__(
        self,
        paths,
        mode="train",
        data_type="Tr",
        image_size=128,
        transform=None,
        threshold=10,
        split_num=1,
        split_idx=0,
        pcc=False,
        dataset_max_size=None,
    ):
        self.paths = [paths] if isinstance(paths, Path) else paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx
        self.dataset_max_size = dataset_max_size
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc

        self._set_file_paths(self.paths, self.dataset_max_size)

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        # if '/ct_' in self.image_paths[index]: # TODO: what is this for?
        #     subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except Exception as e:
                print(traceback.format_exc())
                print(self.image_paths[index])

        if self.pcc:
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.label.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][
                    random_index[3]
                ] = 1
                subject.add_image(
                    tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                    image_name="crop_mask",
                )
                subject = tio.CropOrPad(
                    mask_name="crop_mask",
                    target_shape=(self.image_size, self.image_size, self.image_size),
                )(subject)

        if self.mode == "train" and self.data_type == "Tr":
            return (
                subject.image.data.clone().detach(),
                subject.label.data.clone().detach(),
            )
        else:
            return (
                subject.image.data.clone().detach(),
                subject.label.data.clone().detach(),
                self.image_paths[index],
            )

    def _set_file_paths(self, paths, max_size):
        self.image_paths = []
        self.label_paths = []
        self.label_volumes = []
        self.image_spacing = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            with open(path, "r") as f:
                json_data = json.load(f)

            if max_size is not None:
                import random
                print(f"Limiting data to {max_size} samples")
                random.shuffle(json_data)
                json_data = json_data[:max_size]

            self.image_paths.extend(
                [x["image"] for x in json_data if x["volume"] > self.threshold]
            )
            self.label_paths.extend(
                [x["label"] for x in json_data if x["volume"] > self.threshold]
            )
            self.label_volumes.extend(
                [x["volume"] for x in json_data if x["volume"] > self.threshold]
            )
            self.image_spacing.extend(
                [x["spacing"] for x in json_data if x["volume"] > self.threshold]
            )

    def get_filtered_json(self) -> typing.Dict:
        return [
            {
                "image": self.image_paths[i],
                "label": self.label_paths[i],
                "volume": self.label_volumes[i],
                "spacing": self.image_spacing[i],
            }
            for i in range(len(self.image_paths))
        ]


class DatasetValidation(DatasetMerged):
    def _set_file_paths(self, paths):
        super()._set_file_paths(paths)

        self.image_paths = self.image_paths[self.split_idx :: self.split_num]
        self.label_paths = self.label_paths[self.split_idx :: self.split_num]


class BackgroundDataLoader(mDataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Test_Single(Dataset):
    def __init__(self, paths, image_size=128, transform=None, threshold=500):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        return (
            subject.image.data.clone().detach(),
            subject.label.data.clone().detach(),
            self.image_paths[index],
        )

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace("images", "labels"))
