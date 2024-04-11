import json
import os
from pathlib import Path
import traceback

import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset


class Dataset_Union_ALL(Dataset): 
    def __init__(self, paths, mode='train', data_type='Tr', image_size=128, 
                 transform=None, threshold=500,
                 split_num=1, split_idx=0, pcc=False):
        self.paths = paths
        self.data_type = data_type
        self.split_num=split_num
        self.split_idx=split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc
    
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
            image = tio.ScalarImage.from_sitk(sitk_image),
            label = tio.LabelMap.from_sitk(sitk_label),
        )

        # if '/ct_' in self.image_paths[index]: # TODO: what is this for?
        #     subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except Exception as e:
                print(traceback.format_exc())
                print(self.image_paths[index])

        if(self.pcc):
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if(len(random_index)>=1):
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.label.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
                subject.add_image(tio.LabelMap(tensor=crop_mask,
                                                affine=subject.label.affine),
                                    image_name="crop_mask")
                subject = tio.CropOrPad(mask_name='crop_mask', 
                                        target_shape=(self.image_size,self.image_size,self.image_size))(subject)

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        
        if self.mode == "train" and self.data_type == 'Tr':
            return subject.image.data.clone().detach(), subject.label.data.clone().detach()
        else:
            return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]
 
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            with open(path, 'r') as f:
                json_data = json.load(f)

            self.image_paths.extend([x['image'] for x in json_data])
            self.label_paths.extend([x['label'] for x in json_data])
            # label_dirs = Path(path).glob(f"labels{self.data_type}/*")
            # for label_dir in label_dirs:
            #     if label_dir.is_dir():
            #         for file in label_dir.glob("*.nii*"):
            #             # label_path = path / label_dir / file.name
            #             # find all coresponding image files
            #             train_dirs = Path(path).glob(f"images{self.data_type}*")
            #             train_dirs = list(filter(lambda x: x.is_dir(), train_dirs))
            #             image_paths = [dir / file.name for dir in train_dirs]
            #             label_paths = [file] * len(image_paths)
            #             self.image_paths.extend(image_paths)
            #             self.label_paths.extend(label_paths)

class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, paths):
        super()._set_file_paths(paths)
        
        self.image_paths = self.image_paths[self.split_idx::self.split_num]
        self.label_paths = self.label_paths[self.split_idx::self.split_num]




class Union_Dataloader(DataLoader):
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
            image = tio.ScalarImage.from_sitk(sitk_image),
            label = tio.LabelMap.from_sitk(sitk_label),
        )

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])


        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        

        return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]
    
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace('images', 'labels'))
