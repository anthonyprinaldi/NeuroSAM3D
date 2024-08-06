import json
import random
from typing import Dict, List, Optional

from monai.data import DataLoader
from prefetch_generator import BackgroundGenerator

from .prepare_data import TARGET_DATASET_DIR


class DatasetJson:
    def __init__(
        self,
        dataset_list: List[str] | str,
        volume_threshold: int=10,
        dataset_max_size: Optional[int]=None,
    ):
        dataset_list = [dataset_list] if isinstance(dataset_list, str) else dataset_list
        self.paths = [
            TARGET_DATASET_DIR / (dataset_name + ".json") for
            dataset_name in dataset_list
        ]
        self.dataset_max_size = dataset_max_size
        self.threshold = volume_threshold

        self._set_file_paths()

    def __len__(self):
        return len(self.label_paths)

    def _set_file_paths(self):
        self.image_paths = []
        self.label_paths = []
        self.label_volumes = []
        self.image_spacing = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in self.paths:
            with open(path, "r") as f:
                json_data = json.load(f)

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

    def get_filtered_json(self) -> List[Dict[str, str | float]]:
        if self.dataset_max_size is not None:
            print(f"Limiting data to {self.dataset_max_size} samples")
            num_samples = min(self.dataset_max_size, len(self.image_paths))
        else:
            num_samples = len(self.image_paths)
        
        all_paths = list(
            zip(
                self.image_paths,
                self.label_paths,
                self.label_volumes,
                self.image_spacing,
            )
        )

        random.shuffle(all_paths)

        self.image_paths, self.label_paths, self.label_volumes, self.image_spacing = zip(
            *all_paths
        )

        return [
            {
                "image": self.image_paths[i],
                "label": self.label_paths[i],
                "volume": self.label_volumes[i],
                "spacing": self.image_spacing[i],
            }
            for i in range(num_samples)
        ]


class DatasetValidation(DatasetJson): # TODO: fix for eval
    def _set_file_paths(self, paths):
        super()._set_file_paths(paths)

        self.image_paths = self.image_paths[self.split_idx :: self.split_num]
        self.label_paths = self.label_paths[self.split_idx :: self.split_num]


class BackgroundDataLoader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())