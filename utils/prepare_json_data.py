import glob
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List

ABDOMEN_CT_DIR = Path('/home/arinaldi/project/aiconsgrp/data/AbdomenCT_1K')
AMOS_DIR = Path('/home/arinaldi/project/aiconsgrp/data/AMOS')
BRATS2020_DIR = Path('/home/arinaldi/project/aiconsgrp/data/BraTs2020')
COVID_CT_LUNG_DIR = Path('/home/arinaldi/project/aiconsgrp/data/COVID_CT_Lung')
CT_STROKE_DIR = Path('/home/arinaldi/project/aiconsgrp/data/CTStroke')
HEALTHY_TOTAL_BODY_DIR = Path('/home/arinaldi/project/aiconsgrp/data/Healthy-Total-Body CTs NIfTI Segmentations and Segmentation Organ Values spreadsheet')
ISLES_2022_DIR = Path('/home/arinaldi/project/aiconsgrp/data/ISLES-2022')
KITS23_DIR = Path('/home/arinaldi/project/aiconsgrp/data/kits23')
KNEEMRI_DIR = Path('/home/arinaldi/project/aiconsgrp/data/KneeMRI')
LITS_DIR = Path('/home/arinaldi/project/aiconsgrp/data/LITS')
LUNA_16_DIR = Path('/home/arinaldi/project/aiconsgrp/data/LUNA16')
MM_WHS_DIR = Path('/home/arinaldi/project/aiconsgrp/data/MM-WHS 2017 Dataset')
MSD_DIR = Path('/home/arinaldi/project/aiconsgrp/data/MSD')
CT_ORG_DIR = Path('/home/arinaldi/project/aiconsgrp/data/PKG - CT-ORG')
UPENN_DIR = Path('/home/arinaldi/project/aiconsgrp/data/PKG - UPENN-GBM-NIfTI')
PROSTATE_MR_DIR = Path('/home/arinaldi/project/aiconsgrp/data/Prostate MR Image Segmentation')
SEGTHOR_DIR = Path('/home/arinaldi/project/aiconsgrp/data/SegTHOR')
TCIA_PANCREASE_DIR = Path('/home/arinaldi/project/aiconsgrp/data/TCIA_pancreas_labels-02-05-2017')
TOTALSEGMENTATOR_DIR = Path('/home/arinaldi/project/aiconsgrp/data/Totalsegmentator_dataset_v201')
ONDRI_DIR = Path('/home/arinaldi/project/aiconsgrp/data/wmh_hab')
WORD_DIR = Path('/home/arinaldi/project/aiconsgrp/data/WORD-V0.1.0')


class BaseDatasetJSONGenerator(ABC):
    @classmethod
    @abstractmethod
    def generate(alternate_dir: Path = None):
        pass

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([f for f in dir.glob(f'*{ext}') if contains is None or contains in f.name])

    @classmethod
    def save_dataset_json(cls,
                          train_json: dict,
                          val_json: Optional[dict] = None,
                          test_json: Optional[dict] = None,
                          filename: str = 'dataset.json'
                          ):
        res = {}
        res['name'] = cls.name
        res['modality'] = cls.modality
        res['labels'] = cls.labels
        res['training'] = train_json
        res['validation'] = val_json
        res['testing'] = test_json

        with open(cls.dir / filename, 'w') as f:
            json.dump(res, f, indent=4)


class AbdomenCTJSONGenerator(BaseDatasetJSONGenerator):
    dir = ABDOMEN_CT_DIR
    num_seg_classes = 4
    name = "AbdomenCT"
    modality = ["CT"]
    labels = {
        0: "background",
        1: "liver",
        2: "kidney",
        3: "spleen",
        4: "pancreas"
    }

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
                        "seg": str(work_dir / 'labelsTr' / image.name),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / image.name).exists()
                ]
            )

        cls.save_dataset_json(dataset_json)
        return dataset_json
    
class AMOSJSONGenerator(BaseDatasetJSONGenerator):
    dir = AMOS_DIR
    num_seg_classes = 15
    name = "AMOS"
    modality = ["CT", "MRI"]
    labels = {
        0: "background",
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "inferior vena cava",
        10: "pancreas",
        11: "right adrenal gland",
        12: "left adrenal gland",
        13: "duodenum",
        14: "bladder",
        15: "prostate"
    }

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

        val_json = []
        imagesVal = cls.load_all_images(work_dir / 'imagesVal')

        for image in imagesVal:
            val_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsVal' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsVal' / (image.name)).exists()
                ]
            )

        test_json = []
        imagesTs = cls.load_all_images(work_dir / 'imagesTs')

        for image in imagesTs:
            test_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTs' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTs' / (image.name)).exists()
                ]
            )

        cls.save_dataset_json(dataset_json, val_json=val_json, test_json=test_json)
        return dataset_json
    
class BratsJSONGenerator(BaseDatasetJSONGenerator):
    dir = BRATS2020_DIR
    seg_class_values = [
        1, # necrotic and non-enhancing tumor core
        2, # peritumoral edema
        4, # enhancing tumor
    ]
    name = "BRATS"
    modality = ["MRI"]
    labels = {
        0: "background",
        1: "necrotic and nonenhancing tumor core",
        2: "peritumoral edema",
        4: "enhancing tumor"
    }

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

        cls.save_dataset_json(dataset_json)
        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])
    

class CovidCTJSONGenerator(BaseDatasetJSONGenerator):
    dir = COVID_CT_LUNG_DIR
    num_seg_classes_infection = 1
    num_seg_classes_lung = 2
    name = "CovidCT"
    modality = ["CT"]
    labels = {
        "infection" : {
            0: "background",
            1: "infection"
        },
        "lung" : {
            0: "background",
            1: "right lung",
            2: "left lung"
        }
    }

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
                    } for x in range(cls.num_seg_classes_lung) if (work_dir / 'labelsTr_lung' / (image.name)).exists()
                ]
            )
            dataset_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsTr_infection' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes_infection) if (work_dir / 'labelsTr_infection' / (image.name)).exists()
                ]
            )
            

        cls.save_dataset_json(dataset_json)
        return dataset_json
    
class CTStrokeJSONGenerator(BaseDatasetJSONGenerator):
    dir = CT_STROKE_DIR
    num_seg_classes = 1
    name = "CTStroke"
    modality = ["CT"]
    labels = {
        "CBF" : {
            0: "background",
            1: "CBF",
        },
        "Tmax" : {
            0: "background",
            1: "Tmax",
        }
    }

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

        cls.save_dataset_json(dataset_json)
        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])


class HealthyTotalBodyJSONGenerator(BaseDatasetJSONGenerator):
    dir = HEALTHY_TOTAL_BODY_DIR
    num_seg_classes = 36
    name = "HealthyTotalBody"
    modality = ["CT"]
    labels = {
        1:	"Adrenal glands",
        2:	"Aorta",
        3:	"Bladder",
        4:	"Brain",
        5:	"Heart",
        6:	"Kidneys",
        7:	"Liver",
        8:	"Pancreas",
        9:	"Spleen",
        10:	"Thyroid",
        11:	"VCI",
        12:	"Lung",
        13:	"Carpal",
        14:	"Clavicle",
        15:	"Femur",
        16:	"Fibula",
        17:	"Humerus",
        18:	"Metacarpal",
        19:	"Metatarsal",
        20:	"Patella",
        21:	"Pelvis",
        22:	"Fingers",
        23:	"Radius",
        24:	"Ribcage",
        25:	"Scapula",
        26:	"Skull",
        27:	"Spine",
        28:	"Sternum",
        29:	"Tarsal",
        30:	"Tibia",
        31:	"Toes",
        32:	"Ulna",
        33:	"Skeletal muscle",
        34:	"Subcutaneous fat",
        35:	"Torso fat",
        36:	"Psoas",
    }
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

        cls.save_dataset_json(dataset_json)
        return dataset_json
    
    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])

    
class ISLESJSONGenerator(BaseDatasetJSONGenerator):
    dir = ISLES_2022_DIR
    num_seg_classes = 1
    name = "ISLES"
    modality = ["MRI"]
    labels = {
        0: "background",
        1: "stroke lesion"
    }

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

        cls.save_dataset_json(dataset_json)
        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])
    
class KitsJSONGenerator(BaseDatasetJSONGenerator):
    dir = KITS23_DIR
    num_seg_classes = 2
    name = "Kits"
    modality = ["CT"]
    labels = {
        0: "background",
        1: "kidney",
        2: "tumor",
        3: "cyst",
    }


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

        cls.save_dataset_json(dataset_json)
        return dataset_json
    
class KneeJSONGenerator(BaseDatasetJSONGenerator):
    dir = KNEEMRI_DIR
    num_seg_classes = 6
    name = "StanfordKnee"
    modality = ["MRI"]
    labels = {
        0: "background",
        1: "patellar cartilage",
        2: "femoral cartilage",
        3: "tibial cartilage medial",
        4: "tibial cartilage lateral",
        5: "meniscus medial",
        6: "meniscus lateral",
    }

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
                        "seg": str(work_dir / 'labelsTr_dicom' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr_dicom' / (image.name)).exists()
                ]
            )

        cls.save_dataset_json(dataset_json)
        return dataset_json
    
class LITSJSONGenerator(BaseDatasetJSONGenerator):
    dir = LITS_DIR
    num_seg_classes = 2
    name = "LiTS"
    modality = ["CT"]
    labels = {
        0: "background",
        1: "liver",
        2: "tumor",
    }

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

        cls.save_dataset_json(dataset_json)
        return dataset_json
    
class LUNAJSONGenerator(BaseDatasetJSONGenerator):
    dir = LUNA_16_DIR
    seg_class_values = [
        3,
        4,
        5,
    ]
    name = "LUNA"
    modality = ["CT"]
    labels = {
        0: "background",
        3: "right lung",
        4: "left lung",
        5: "trachea",
    }

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

        cls.save_dataset_json(dataset_json)
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
    name = "MultiModalWholeHeart"
    modality = ["MRI", "CT"]
    labels = {
        0: "background",
        500: "left ventricle",
        600: "right ventricle",
        420: "left atrium",
        550: "right atrium",
        205: "myocardium",
        820: "ascending aorta",
        850: "descending artery",
    }

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

        cls.save_dataset_json(dataset_json)
        return dataset_json
    
class MSDJSONGenerator(BaseDatasetJSONGenerator):
    dir = MSD_DIR
    name = "MedSamDecathlon"
    modality = ["CT"]
    labels = {
        "Task02_Heart": {
            0: "background",
            1: "left atrium",
        },
        "Task03_Liver": {
            0: "background",
            1: "liver",
            2: "tumor",
        },
        "Task04_Hippocampus": {
            0: "background",
            1: "anterior hippocampus",
            2: "posterior hippocampus",
        },
        "Task05_Prostate": {
            0: "background",
            1: "peripheral zone",
            2: "transition zone",
        },
        "Task06_Lung": {
            0: "background",
            1: "tumor",
        },
        "Task07_Pancreas": {
            0: "background",
            1: "pancreatic parenchyma",
            2: "pancreatic mass",
        },
        "Task08_HepaticVessel": {
            0: "background",
            1: "vessel",
            2: "tumor",
        },
        "Task09_Spleen": {
            0: "background",
            1: "spleen",
        },
        "Task10_Colon": {
            0: "background",
            1: "colon cancer primaries",
        },
    }

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
            imagesTr = [f for f in imagesTr if not f.name.startswith("._")]
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

        cls.save_dataset_json(dataset_json)
        return dataset_json


class CTORGJSONGenerator(BaseDatasetJSONGenerator):
    dir = CT_ORG_DIR
    num_seg_classes = 5
    name = "CTOrgan"
    modality = ["CT"]
    labels = {
        0: "background",
        1: "liver",
        2: "bladder",
        3: "lungs",
        4: "kidneys",
        5: "bone",
        6: "brain",
    }

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
                        "seg": str(work_dir / 'labelsTr' / image.name),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / image.name).exists()
                ]
            )

        cls.save_dataset_json(dataset_json)
        return dataset_json

class UpennJSONGenerator(BaseDatasetJSONGenerator):
    dir = UPENN_DIR
    num_seg_classes = 4
    name = "MRIGlioblastoma"
    modality = ["MRI"]
    labels = {
        0: "background",
        1: "necrotic tumor core",
        2: "edema",
        # no class 3
        4: "enhancing tumor",
    }

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

        cls.save_dataset_json(dataset_json)
        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])


class ProstateJSONGenerator(BaseDatasetJSONGenerator):
    dir = PROSTATE_MR_DIR
    num_seg_classes = 1
    name = "ProstateMRI"
    modality = ["MRI"]
    labels = {
        0: "background",
        1: "prostate",
    }

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

        cls.save_dataset_json(dataset_json)
        return dataset_json


class SegTHORJSONGenerator(BaseDatasetJSONGenerator):
    dir = SEGTHOR_DIR
    num_seg_classes = 4
    name = "SegThoracicOrgans"
    modality = ["CT"]
    labels = {
        0: "background",
        1: "esophagus",
        2: "heart",
        3: "trachea",
        4: "aorta",
    }

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

        cls.save_dataset_json(dataset_json)
        return dataset_json
    
class TCIAPancreasJSONGenerator(BaseDatasetJSONGenerator):
    dir = TCIA_PANCREASE_DIR
    num_seg_classes = 1
    name = "PancreasCt"
    modality = ["CT"]
    labels = {
        0: "background",
        1: "pancreas",
    }

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
                        "seg": str(work_dir / 'labelsTr' / image.name),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsTr' / image.name).exists()
                ]
            )

        cls.save_dataset_json(dataset_json)
        return dataset_json


class TotalSegmentatorJSONGenerator(BaseDatasetJSONGenerator):
    dir = TOTALSEGMENTATOR_DIR
    name = "TotalSegmentator"
    modality = ["CT"]
    labels = {} # labels already separated

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

        cls.save_dataset_json(dataset_json)
        return dataset_json


class ONDRIJSONGenerator(BaseDatasetJSONGenerator):
    dir = ONDRI_DIR
    num_seg_classes = 1
    name = "ONDRI"
    modality = ["MRI"]
    labels = {
        0: "background",
        1: "brain",
        2: "wmh",
    }

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

        val_json = []
        imagesVal = cls.load_all_images(work_dir / 'imagesVal')
        for image in imagesVal:
            if "masked" not in str(image):
                val_json.extend(
                    [
                        {
                            "image": str(image),
                            "seg": str(work_dir / 'labelsVal_brain' / (image.name)),
                            "seg_index": x + 1
                        } for x in range(cls.num_seg_classes) if (work_dir / 'labelsVal_brain' / (image.name)).exists()
                    ]
                )
            val_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsVal_wmh' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsVal_wmh' / (image.name)).exists()
                ]
            )

        cls.save_dataset_json(dataset_json, val_json=val_json)
        return dataset_json

    @staticmethod
    def load_all_images(dir: Path, ext: str = '.nii.gz', contains: str = None):
        return sorted([Path(f) for f in glob.glob(f'{str(dir)}*/*{ext}') if contains is None or contains in Path(f).name])


class WORDJSONGenerator(BaseDatasetJSONGenerator):
    dir = WORD_DIR
    num_seg_classes = 16
    name = "WORD"
    modality = ["CT"]
    labels = {
        0: "background",
        1: "liver",
        2: "spleen",
        3: "kidney left",
        4: "kidney right",
        5: "stomach",
        6: "gallbladder",
        7: "esophagus",
        8: "pancreas",
        9: "duodenum",
        10: "colon",
        11: "intestine",
        12: "adrenal",
        13: "recturm",
        14: "bladder",
        15: "head of femur left",
        16: "head of femur right",
    }

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
        val_json = []
        imagesVal = cls.load_all_images(work_dir / 'imagesVal')
        for image in imagesVal:
            val_json.extend(
                [
                    {
                        "image": str(image),
                        "seg": str(work_dir / 'labelsVal' / (image.name)),
                        "seg_index": x + 1
                    } for x in range(cls.num_seg_classes) if (work_dir / 'labelsVal' / (image.name)).exists()
                ]
            )

        cls.save_dataset_json(dataset_json, val_json=val_json)
        return dataset_json



class JSONJoiner:
    def __init__(self, *args):
        self.dataset_classes: List[BaseDatasetJSONGenerator] = args
    
    def _generate(self) -> List[dict]:
        dataset_json = []
        total_data = 0
        for dataset_class in self.dataset_classes:
            res = dataset_class.generate()
            print(f"Generated {len(res)} items for {dataset_class.__name__}")
            total_data += len(res)
            dataset_json.extend(res)
        print(f"Total data items: {total_data}")
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
        HealthyTotalBodyJSONGenerator,
        ISLESJSONGenerator,
        KitsJSONGenerator,
        KneeJSONGenerator,
        LITSJSONGenerator,
        LUNAJSONGenerator,
        MMWHSJSONGenerator,
        MSDJSONGenerator,
        CTORGJSONGenerator,
        UpennJSONGenerator,
        ProstateJSONGenerator,
        SegTHORJSONGenerator,
        TCIAPancreasJSONGenerator,
        TotalSegmentatorJSONGenerator,
        ONDRIJSONGenerator,
        WORDJSONGenerator
    )

    joiner = JSONJoiner(*all_data_classes)

    joiner.save_json('dataset.json')
    # joiner._generate()