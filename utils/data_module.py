from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import monai.data
import monai.transforms
from torch.utils.data import DataLoader

from .data_list import POSSIBLE_DATASETS
from .data_loader import BackgroundDataLoader, DatasetJson
from .transforms import get_train_trainsforms, get_val_transforms


class NeuroSamDataModule(L.LightningDataModule):

    def __init__(self,
                 img_size: int,
                 batch_size: int,
                 num_workers: int,
                 volume_threshold: int,
                 training_sets: List[str],
                 validation_sets: List[str],
                 cache_dir: Union[Path, str, None]=None,
                 ) -> None:
        """Initialize the Data Module for the NeuroSAM model.

        :param img_size: Image size for the model
        :type img_size: int
        :param batch_size: Batch size to use in DataLoader (per node)
        :type batch_size: int
        :param num_workers: Number of workers to use in DataLoader
        :type num_workers: int
        :param volume_threshold: Minimum volume threshold for samples to be
            included in the dataset
        :type volume_threshold: int
        :param training_sets: List of training datasets to use
        :type training_sets: List[str]
        :param validation_sets: List of validation datasets to use
        :type validation_sets: List[str]
        :param cache_dir: Location to store temp files in slurm job
        :type cacge_dir: Union[Path, str, None]
        """
        super().__init__()

        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.volume_threshold = volume_threshold
        self.cache_dir = cache_dir

        self._validate_datasets(
            training_sets=training_sets,
            validation_sets=validation_sets
        )

        self.train_transforms = get_train_trainsforms(img_size)
        self.val_transforms = get_val_transforms(img_size)

    def _validate_datasets(self, training_sets: List[str], validation_sets: List[str]) -> None:
        for dataset in training_sets:
            if dataset not in POSSIBLE_DATASETS:
                raise ValueError(f"Invalid training dataset: {dataset}")
            
        self.training_sets = training_sets

        for dataset in validation_sets:
            if dataset not in POSSIBLE_DATASETS:
                raise ValueError(f"Invalid validation dataset: {dataset}")
            
        self.validation_sets = validation_sets

    
    def prepare_data(self) -> None:
        print("No data preparation needed")
        # TODO: maybe the resizing of images here
    
    def setup(self, stage: Optional[str]=None) -> None:
        
        if stage == "fit":
            train_json_fetcher = DatasetJson(
                dataset_list=self.training_sets,
                volume_threshold=self.volume_threshold,
            )

            train_data_paths = train_json_fetcher.get_filtered_json()

            self.train_dataset = (
                dataset_class(
                    data=train_data_paths,
                    transform=self.train_transforms,
                    cache_dir=self.cache_dir,
                )
                if issubclass(dataset_class, PersistentDataset)
                else dataset_class(
                    data=train_data_paths,
                    transform=self.train_transforms,
                )
            )

            val_json_fetcher = DatasetJson(
                dataset_list=self.validation_sets,
                volume_threshold=self.volume_threshold,
            )

            val_data_paths = val_json_fetcher.get_filtered_json()

            self.val_dataset = (
                dataset_class(
                    data=val_data_paths,
                    transform=self.val_transforms,
                    cache_dir=self.cache_dir,
                )
                if issubclass(dataset_class, PersistentDataset)
                else dataset_class(
                    data=val_data_paths,
                    transform=self.val_transforms,
                )
            )

        if stage == "test":
            raise NotImplementedError("Data Module Test stage not implemented")

        if stage == "predict":
            raise NotImplementedError("Data Module Predict stage not implemented")
        

    def train_dataloader(self):
        return BackgroundDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return BackgroundDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return None
    
    def predict_dataloader(self):
        return None
    
    def state_dict(self) -> Dict[str, Any]:
        return 