from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import monai.data
import monai.transforms
from torch.utils.data import DataLoader

from .transforms import get_train_trainsforms, get_val_transforms


class NeuroSAMDataModule(L.LightningDataModule):

    def __init__(self, args) -> None:
        super().__init__()

        self.train_transforms = get_train_trainsforms(img_size)
        self.val_transforms = get_val_transforms(img_size)

    
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