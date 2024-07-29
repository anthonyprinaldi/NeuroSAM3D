from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import monai.data
import monai.transforms
from torch.utils.data import DataLoader

from .data_loader import BackgroundDataLoader, DatasetMerged


class NeuroSAMDataModule(L.LightningDataModule):

    def __init__(self, args) -> None:
        super().__init__()

        self.train_transforms = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                monai.transforms.ScaleIntensityRangePercentilesd(
                    keys=["image"],
                    lower=0.05,
                    upper=99.95,
                    b_min=-1,
                    b_max=1,
                    clip=True
                ),
                # monai.transforms.CropForegroundd( # TODO: do we need?
                #     keys=["image", "label"],
                #     source_key="image",
                #     select_fn=lambda x: x > -1,
                # ),
                monai.transforms.Compose([
                    monai.transforms.RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
                    monai.transforms.RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
                    monai.transforms.RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
                    monai.transforms.RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
                    monai.transforms.RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(1, 2)),
                    monai.transforms.RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 2)),
                ]),
                monai.transforms.ToTensord(keys=["image", "label"]),
                monai.transforms.OneOf([
                    monai.transforms.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
                    monai.transforms.RandGaussianNoised(keys=["image"], prob=0.5),
                    monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
                    monai.transforms.RandHistogramShiftd(keys=["image"], prob=0.5, num_control_points=10),
                    monai.transforms.RandGaussianSharpend(keys=["image"], prob=0.5),
                ]),
                monai.transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=[args.img_size, args.img_size, args.img_size],
                    pos=1,
                    neg=0,
                    num_samples=1,
                    allow_smaller=True,
                ),
                # ensure that we have the right shape in the end
                monai.transforms.ResizeWithPadOrCropd(
                    keys=["image", "label"],
                    spatial_size=[args.img_size, args.img_size, args.img_size],
                ),
                monai.transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )

        self.val_transforms = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                monai.transforms.ScaleIntensityRangePercentilesd(
                    keys=["image"],
                    lower=0.05,
                    upper=99.95,
                    b_min=-1,
                    b_max=1,
                    clip=True
                ),
                # monai.transforms.CropForegroundd( # TODO: do we need?
                #     keys=["image", "label"],
                #     source_key="image",
                #     select_fn=lambda x: x > -1,
                # ),
                monai.transforms.ToTensord(keys=["image", "label"]),
                # TODO: do we need this?
                # monai.transforms.ResizeWithPadOrCropd(
                #     keys=["image", "label"],
                #     spatial_size=[args.img_size, args.img_size, args.img_size],
                # ),
                monai.transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )

        self.args = args

    
    def prepare_data(self) -> None:
        print("No data preparation needed")
    
    def setup(self, stage: Optional[str]=None) -> None:
        
        if stage == "fit":
            dataset = DatasetMerged(
                # paths=TRAINING, # TODO: change
                paths=Path("data_fixed/medical_preprocessed/overall_Tr.json"),
                image_size=self.args.img_size,
                threshold=self.args.volume_threshold,
            )

            all_data=dataset.get_filtered_json()

            self.train_dataset = monai.data.Dataset(
                data=all_data,
                transform=self.train_transforms
            )

        if stage == "test":
            raise NotImplementedError("Data Module Test stage not implemented")

        if stage == "predict":
            raise NotImplementedError("Data Module Predict stage not implemented")
        

    def train_dataloader(self):
        return BackgroundDataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return None
    
    def test_dataloader(self):
        return None
    
    def predict_dataloader(self):
        return None
    
    def state_dict(self) -> Dict[str, Any]:
        return 