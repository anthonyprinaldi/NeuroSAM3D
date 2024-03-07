import glob
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List

ABDOMEN_CT_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/AbdomenCT_1K')
AMOS_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/AMOS')
BRATS2020_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/BraTs2020')
COVID_CT_LUNG_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/COVID_CT_Lung')
CT_STROKE_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/CTStroke')
HEALTHY_TOTAL_BODY_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/Healthy-Total-Body CTs NIfTI Segmentations and Segmentation Organ Values spreadsheet')
ISLES_2022_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/ISLES-2022')
KITS23_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/kits23')
KNEEMRI_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/KneeMRI')
LITS_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/LITS')
LUNA_16_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/LUNA16')
MM_WHS_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/MM-WHS 2017 Dataset')
MSD_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/MSD')
CT_ORG_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/PKG - CT-ORG')
UPENN_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/PKG - UPENN-GBM-NIfTI')
PROSTATE_MR_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/Prostate MR Image Segmentation')
SEGTHOR_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/SegTHOR')
TCIA_PANCREASE_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/TCIA_pancreas_labels-02-05-2017')
TOTALSEGMENTATOR_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/Totalsegmentator_dataset_v201')
ONDRI_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/wmh_hab')
WORD_DIR = Path('/home/arinaldi/project/aiconsgrp/med_sam/WORD-V0.1.0')


class BaseDatasetJSONGenerator(ABC):
    @classmethod
    @abstractmethod
    def generate(alternate_dir: Path = None):
        pass

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([f for f in dir.glob(f'*{ext}') if contains is None or contains in f.name])        


class AbdomenCTJSONGenerator(BaseDatasetJSONGenerator):
    dir = ABDOMEN_CT_DIR
    num_seg_classes = 4

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name.replace("_0000.nii.gz", ".nii.gz"))),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name.replace("_0000.nii.gz", ".nii.gz"))).exists()
                ]
            )

        return dataset_json
    
class AMOSJSONGenerator(BaseDatasetJSONGenerator):
    dir = AMOS_DIR
    num_seg_classes = 15

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        # imagesVal = cls.load_all_images(work_dir / 'imagesVal')

        # for image in imagesVal:
        #     dataset_json.extend(
        #         [
        #             {
        #                 "image": str(image),
        #                 "seg": str(work_dir / 'labelsVal' / (image.name)),
        #                 "seg_index": x + 1
        #             } for x in range(cls.num_seg_classes) if (work_dir / 'labelsVal' / (image.name)).exists()
        #         ]
        #     )

        return dataset_json
    
class BratsJSONGenerator(BaseDatasetJSONGenerator):
    dir = BRATS2020_DIR
    seg_class_values = [
        1, # necrotic and non-enhancing tumor core
        2, # peritumoral edema
        4, # GD-enhancing tumor
    ]

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x
                    } for x in cls.seg_class_values if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])
    

class CovidCTJSONGenerator(BaseDatasetJSONGenerator):
    dir = COVID_CT_LUNG_DIR
    num_seg_classes = 2

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_lung' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr_lung' / (image.name)).exists()
                ]
            )
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_infection' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(1) if (work_dir / 'labelsTr_infection' / (image.name)).exists()
                ]
            )
            

        return dataset_json
    
class CTStrokeJSONGenerator(BaseDatasetJSONGenerator):
    dir = CT_STROKE_DIR
    num_seg_classes = 1

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_CBF' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr_CBF' / (image.name)).exists()
                ]
            )
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_Tmax' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr_Tmax' / (image.name)).exists()
                ]
            )

        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])


class HealthyTotalBodyJSONGenerator(BaseDatasetJSONGenerator):
    dir = HEALTHY_TOTAL_BODY_DIR
    num_seg_classes = 36
    # TODO: double check the images and seg masks line up

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json
    
    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])

    
class ISLESJSONGenerator(BaseDatasetJSONGenerator):
    dir = ISLES_2022_DIR
    num_seg_classes = 1

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])
    
class KitsJSONGenerator(BaseDatasetJSONGenerator):
    dir = KITS23_DIR
    num_seg_classes = 2

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json
    
class KneeJSONGenerator(BaseDatasetJSONGenerator):
    dir = KNEEMRI_DIR
    num_seg_classes = 6

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)), # TODO: which seg labels?
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json
    
class LITSJSONGenerator(BaseDatasetJSONGenerator):
    dir = LITS_DIR
    num_seg_classes = 2

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr', ext=".nii")

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json
    
class LUNAJSONGenerator(BaseDatasetJSONGenerator):
    dir = LUNA_16_DIR
    num_seg_classes = 2

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json
    
class MMWHSJSONGenerator(BaseDatasetJSONGenerator):
    dir = MM_WHS_DIR
    seg_class_values = [
        500, # left ventricle
        600, # right ventricle
        420, # left atrium
        550, # right atrium
        205, # myocardium
        820, # ascending aorta
        850, # descending artery
    ]

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr_CT')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_CT' / (image.name)),
                        "seg_index": x
                    } for x in cls.seg_class_values if (work_dir / 'labelsTr_CT' / (image.name)).exists()
                ]
            )
        
        imagesTr = cls.load_all_images(work_dir / 'imagesTr_MR')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_MR' / (image.name)),
                        "seg_index": x
                    } for x in cls.seg_class_values if (work_dir / 'labelsTr_MR' / (image.name)).exists()
                ]
            )

        return dataset_json
    
class MSDJSONGenerator(BaseDatasetJSONGenerator):
    dir = MSD_DIR

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        base_work_dir = cls.dir if alternate_dir is None else alternate_dir

        sub_dirs = {
            'Task02_Heart': 1,
            'Task03_Liver': 2,
            'Task04_Hippocampus': 2,
            'Task05_Prostate': 2,
            'Task06_Lung': 1,
            'Task07_Pancreas': 2,
            'Task08_HepaticVessel': 2,
            'Task09_Spleen': 1,
            'Task10_Colon': 1,
        }

        for sub_dir, num_seg_classes in sub_dirs.items():
            work_dir = base_work_dir / sub_dir
            imagesTr = cls.load_all_images(work_dir / 'imagesTr')
            for image in imagesTr:
                dataset_json.extend(
                    [
                        {
                            "image": str(image),
                            "seg": str(work_dir / 'labelsTr' / (image.name)),
                            "seg_index": x + 1
                        } for x in range(num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                    ]
                )

        return dataset_json


class CTORGJSONGenerator(BaseDatasetJSONGenerator):
    dir = CT_ORG_DIR
    num_seg_classes = 5

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name.replace("volume", "labels"))),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name.replace("volume", "labels"))).exists()
                ]
            )

        return dataset_json

class UpennJSONGenerator(BaseDatasetJSONGenerator):
    dir = UPENN_DIR
    num_seg_classes = 4

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_auto' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr_auto' / (image.name)).exists()
                ]
            )
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_seg' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr_seg' / (image.name)).exists()
                ]
            )

        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])


class ProstateJSONGenerator(BaseDatasetJSONGenerator):
    dir = PROSTATE_MR_DIR
    num_seg_classes = 1

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json


class SegTHORJSONGenerator(BaseDatasetJSONGenerator):
    dir = SEGTHOR_DIR
    num_seg_classes = 4

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        return dataset_json
    
class TCIAPancreasJSONGenerator(BaseDatasetJSONGenerator):
    dir = TCIA_PANCREASE_DIR
    num_seg_classes = 1

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name.replace("image", "label"))),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name.replace("image", "label"))).exists()
                ]
            )

        return dataset_json


class TotalSegmentatorJSONGenerator(BaseDatasetJSONGenerator):
    dir = TOTALSEGMENTATOR_DIR

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')

        for image in imagesTr:
            # search for the label images
            label_files = work_dir.glob(f'labelsTr/{image.name.replace(".nii.gz", "")}*.nii.gz')
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(seg),
                        "seg_index": 1
                    } for seg in label_files
                ]
            )

        return dataset_json


class ONDRIJSONGenerator(BaseDatasetJSONGenerator):
    dir = ONDRI_DIR
    num_seg_classes = 1

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')
        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_brain' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr_brain' / (image.name)).exists()
                ]
            )
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_wmh' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr_wmh' / (image.name)).exists()
                ]
            )

        # imagesVal = cls.load_all_images(work_dir / 'imagesVal')
        # for image in imagesVal:
        #     if "masked" not in str(image):
        #         dataset_json.extend(
        #             [
        #                 {
        #                     "image": str(image),
        #                     "seg": str(work_dir / 'labelsVal_brain' / (image.name)),
        #                     "seg_index": x + 1
        #                 } for x in range(cls.num_seg_classes) if (work_dir / 'labelsVal_brain' / (image.name)).exists()
        #             ]
        #         )
        #     dataset_json.extend(
        #         [
        #             {
        #                 "image": str(image),
        #                 "seg": str(work_dir / 'labelsVal_wmh' / (image.name)),
        #                 "seg_index": x + 1
        #             } for x in range(cls.num_seg_classes) if (work_dir / 'labelsVal_wmh' / (image.name)).exists()
        #         ]
        #     )

        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])


class WORDJSONGenerator(BaseDatasetJSONGenerator):
    dir = WORD_DIR
    num_seg_classes = 16

    @classmethod
    def generate(cls, alternate_dir: Optional[Path] = None):
        dataset_json = []
        work_dir = cls.dir if alternate_dir is None else alternate_dir

        imagesTr = cls.load_all_images(work_dir / 'imagesTr')
        for image in imagesTr:
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / (image.name)).exists()
                ]
            )

        # imagesVal = cls.load_all_images(work_dir / 'imagesVal')
        # for image in imagesVal:
        #     dataset_json.extend(
        #         [
        #             {
        #                 "image": str(image),
        #                 "seg": str(work_dir / 'labelsVal' / (image.name)),
        #                 "seg_index": x + 1
        #             } for x in range(cls.num_seg_classes) if (work_dir / 'labelsVal' / (image.name)).exists()
        #         ]
        #     )

        return dataset_json



class JSONJoiner:
    def __init__(self, *args):
        self.dataset_classes: List[BaseDatasetJSONGenerator] = args
    
    def _generate(self) -> List[dict]:
        dataset_json = []
        for dataset_class in self.dataset_classes:
            res = dataset_class.generate()
            print(f"Generated {len(res)} items for {dataset_class.__name__}")
            dataset_json.extend(res)
        return dataset_json
    
    def save_json(self, filename: str = 'dataset.json'):
        with open(filename, 'w') as f:
            json.dump(self._generate(), f, indent=4)

    
if __name__ == "__main__":
    all_data_classes = (
        AbdomenCTJSONGenerator,
        AMOSJSONGenerator,
        BratsJSONGenerator,
        CovidCTJSONGenerator,
        CTStrokeJSONGenerator,
        HealthyTotalBodyJSONGenerator, # need image locations
        ISLESJSONGenerator,
        KitsJSONGenerator,
        KneeJSONGenerator, # need to choose seg
        LITSJSONGenerator,
        LUNAJSONGenerator,
        MMWHSJSONGenerator,
        MSDJSONGenerator,
        CTORGJSONGenerator,
        UpennJSONGenerator,
        ProstateJSONGenerator,
        SegTHORJSONGenerator,
        TCIAPancreasJSONGenerator, # need image locations
        TotalSegmentatorJSONGenerator,
        ONDRIJSONGenerator,
        WORDJSONGenerator
    )

    joiner = JSONJoiner(*all_data_classes)

    joiner.save_json('dataset.json')