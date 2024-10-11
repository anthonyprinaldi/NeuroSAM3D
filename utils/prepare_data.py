import json
import os
import os.path as osp
from pathlib import Path

import nibabel as nib
import numpy as np
import torchio as tio
from .prepare_json_data import *
from tqdm import tqdm

DATASET_ROOT = "../data"
DATASET_LIST = [
    AbdomenCTJSONGenerator,
    ADNIJSONGenerator,
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
    WORDJSONGenerator,
]
TARGET_DATASET_DIR = Path("./data_fixed/medical_preprocessed")


def resample_nii(
    input_path: str,
    output_path: str,
    target_spacing: tuple=None,
    n=None,
    reference_image=None,
    mode="linear",
):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """

    # Load the nii.gz file using torchio
    def nib_loader(path):
        nib_img = nib.load(path)
        return (nib_img.get_fdata().astype(np.float32), nib_img.affine)
    
    subject = tio.Subject(
        img=tio.ScalarImage(
            input_path,
            reader=nib_loader,
        )
    )
    # check if temporal dimension is lenght 1, otherwise select the first index
    if(subject.img.shape[0]>1):
        print(f"Subject has two channels: {subject.img.shape}, {input_path}")
        subject.img.data = subject.img.data[0].unsqueeze(0)

    if target_spacing is None:
        save_image = subject.img
        save_image.save(output_path)
        return
    
    resampler = tio.Compose(
        [
            # tio.ToCanonical(),
            tio.Resample(target=target_spacing, image_interpolation=mode),
        ]
    )
    resampled_subject = resampler(subject)

    if n is not None:
        image = resampled_subject.img
        tensor_data = image.data
        if isinstance(n, int):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)  # TODO: do we need to crop
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img

    save_image.save(output_path)


def main(args):
    dt = args.dataset_type

    total_failed = 0
    total_success = 0

    overall_data_dict = []

    for dataset in DATASET_LIST:
        dataset_dir = dataset.dir
        dataset_json = []
        meta_info = json.load(open(osp.join(dataset_dir, "dataset.json")))

        print(meta_info["name"], meta_info["modality"])
        num_classes = len(meta_info["labels"]) - 1
        print("num_classes:", num_classes, meta_info["labels"])

        dataset_name = dataset.name

        target_save_dir = osp.join(TARGET_DATASET_DIR, dataset_name)

        data_list = meta_info[
            {"Tr": "training", "Val": "validation", "Ts": "testing"}[dt]
        ]

        if data_list is None:
            continue

        for item in tqdm(data_list, desc=f"{dataset_name}"):

            img, seg, seg_idx = item["image"], item["seg"], int(item["seg_index"])
            if dataset_name == "TotalSegmentator":
                cls_name = (
                    Path(seg).parts[-1].split("_", maxsplit=1)[1].replace(".nii.gz", "")
                )
            elif dataset_name == "MedSamDecathlon":
                task = Path(seg).parts[-3]
                cls_name = meta_info["labels"][task][str(seg_idx)].replace(" ", "_")
            elif dataset_name == "CovidCT":
                task = Path(seg).parts[-2].split("_")[1]
                cls_name = meta_info["labels"][task][str(seg_idx)].replace(" ", "_")
            elif dataset_name == "CTStroke":
                task = Path(seg).parts[-2].split("_")[1]
                cls_name = meta_info["labels"][task][str(seg_idx)].replace(" ", "_")
            elif dataset_name == "ADNI":
                cls_name = str(meta_info["labels"][str(seg_idx)])
            else:
                cls_name = meta_info["labels"][str(seg_idx)].replace(" ", "_")

            img_parent_folder = Path(img).parent.parts[-1]
            img_ext = (
                img_parent_folder.split("_")[-1] if "_" in img_parent_folder else ""
            )

            seg_parent_folder = Path(seg).parent.parts[-1]
            seg_ext = (
                seg_parent_folder.split("_")[-1] if "_" in seg_parent_folder else ""
            )

            target_img_dir = osp.join(
                target_save_dir, f"images{dt}" + (f"_{img_ext}" if img_ext else "")
            )
            target_seg_dir = osp.join(target_save_dir, f"labels{dt}")
            os.makedirs(target_img_dir, exist_ok=True)
            os.makedirs(target_seg_dir, exist_ok=True)

            resample_img = osp.join(target_img_dir, osp.basename(img))
            if not osp.exists(resample_img):
                tqdm.write("copying raw image...")
                resample_nii(img, resample_img)
            else:
                tqdm.write(f"skiping {resample_img} already exists")

            img = resample_img

            target_seg_class_dir = osp.join(
                target_seg_dir, cls_name + (f"_{seg_ext}" if seg_ext else "")
            )
            os.makedirs(target_seg_class_dir, exist_ok=True)

            if dataset_name == "TotalSegmentator":
                target_seg_path = osp.join(
                    target_seg_class_dir, osp.basename(seg).replace("_" + cls_name, "")
                )
            else:
                target_seg_path = osp.join(target_seg_class_dir, osp.basename(seg))

            seg_img = nib.load(seg)
            spacing = tuple(seg_img.header["pixdim"][1:4])
            spacing_voxel = spacing[0] * spacing[1] * spacing[2]
            seg_arr = seg_img.get_fdata()
            seg_arr[seg_arr != seg_idx] = 0
            seg_arr[seg_arr != 0] = 1
            # if(volume<10): # TODO: select this value later, don't skip for now
            #     tqdm.write(f"skiping too small:\n{img=}, {seg=}, {cls_name=}")
            #     total_failed += 1
            #     continue

            reference_image = tio.ScalarImage(img)
            reference_spacing = reference_image.spacing
            if osp.exists(target_seg_path):
                tqdm.write(f"skiping {target_seg_path} already exists")
            else:
                tqdm.write("resampling seg...")
                resample_nii(
                    seg,
                    target_seg_path,
                    target_spacing=reference_spacing,
                    n=seg_idx,
                    reference_image=reference_image,
                    mode="nearest",
                )
            
            new_seg_img = nib.load(target_seg_path)
            volume = new_seg_img.get_fdata().sum() * np.prod(new_seg_img.header["pixdim"][1:4])

            total_success += 1

            dataset_json.append(
                {
                    "image": img,
                    "label": target_seg_path,
                    "class": cls_name,
                    "volume": volume,
                    "spacing": str(spacing),
                }
            )

        with open(Path(TARGET_DATASET_DIR) / f"{dataset_name}_{dt}.json", "w") as f:
            json.dump(dataset_json, f, indent=4)

        overall_data_dict.extend(dataset_json)

    print(f"Total success: {total_success}\nTotal failed: {total_failed}")
    with open(Path(TARGET_DATASET_DIR) / f"overall_{dt}.json", "w") as f:
        json.dump(overall_data_dict, f, indent=4)