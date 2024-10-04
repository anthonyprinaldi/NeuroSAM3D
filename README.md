# SAM-Med3D \[[Paper](https://arxiv.org/abs/2310.15161)]

<a src="https://img.shields.io/badge/cs.CV-2310.15161-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2310.15161"> 
<img src="https://img.shields.io/badge/cs.CV-2310.15161-b31b1b?logo=arxiv&logoColor=red">
<a src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat" href="https://github.com/uni-medical/SAM-Med3D/tree/main?tab=readme-ov-file#-discussion-group"> <img src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat">
</a>


<div align="center">
  <img src="assets/motivation.png">
</div>

## 🔥🌻📰 News 📰🌻🔥
- **[New Checkpoints Release]** A newer version of finetuned SAM-Med3D named `SAM-Med3D-turbo` is released now. We fine-tuned it on 44 datasets ([list](https://github.com/uni-medical/SAM-Med3D/issues/2#issuecomment-1849002225)) to improve the performance. Hope this update can help you 🙂.
- **[New Checkpoints Release]** Finetuned SAM-Med3D for organ/brain segmentation is released now! Hope you enjoy the enhanced performance for specific tasks 😉. Details are in [results](https://github.com/uni-medical/SAM-Med3D/blob/main/readme.md#-dice-on-different-anatomical-architecture-and-lesions) and [ckpt](https://github.com/uni-medical/SAM-Med3D#-checkpoint).
- **[Recommendation]** If you are interested in computer vision, 
we recommend checking out [OpenGVLab](https://github.com/OpenGVLab) for more exciting projects like [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D/tree/main)!

## 🌟 Highlights
- 📚 Curated the most extensive volumetric medical dataset to date for training, boasting 131K 3D masks and 247 categories.
- 🚤 Achieved efficient promptable segmentation, requiring 10 to 100 times fewer prompt points for satisfactory 3D outcomes.
- 🏆 Conducted a thorough assessment of SAM-Med3D across 15 frequently used volumetric medical image segmentation datasets.

## 🔨 Usage
### <a name="training-finetuning"></a>Training / Fine-tuning
We recommend using `DinoV2` pretrained encoder weights to fine-tune the model. Reach out to [Tony Xu](https://tonyxu.me/) for the weights. 

To use these pretrained weights and encoder, one must set the following fiels in the [`configs/dino.yaml`](./configs/dino.yaml):

```
model:
  model_type: dinov2
  model_cfg: ../configs/train/vit3d_highres
  pretrained_weights: <FILEPATH HERE>
```

To train the SAM-Med3D model on your own data, follow these steps:

#### 0. **(Recommend) Prepare the Pre-trained Weights**

Download the checkpoint from [ckpt section](https://github.com/uni-medical/SAM-Med3D#-checkpoint) (or get checkpoint from [Tony above](#training-finetuning))and move the pth file into `NeuroSAM3D/ckpt/` (We recommand to use DinoV2 weights).


#### 1. Prepare Your Training Data: 

Ensure that your training data is organized according to the structure shown in the `data_fixed/medical_preprocessed` directories. The general path to an image is `data_fixed/medical_preprocessed/<DATASET_NAME>/images<SET>/<IMAGE_FILENAME>.nii.gz`. The general path to a label is `data_fixed/medical_preprocessed/<DATASET_NAME>/labels<SET>/<CLASS>/<IMAGE_FILENAME>.nii.gz`. 

The target file structures should be like the following:
```
data_fixed/medical_preprocessed
      ├── AMOS
      │ ├── imagesTr
      │ │ ├── amos_0001.nii.gz
      │ │ ├── ...
      │ ├── labelsTr
      │ │ ├── aorta
      │ │ │ ├── amos_0001.nii.gz
      │ │ │ ├── ...
      │ │ ├── bladder
      │ │ │ ├── amos_0001.nii.gz
      │ │ │ ├── ...
      │ │ ├── ...
      │ ├── imagesVal
      │ │ ├── ...
      │ ├── labelsVal
      │ │ ├── ...
      ├── BRATS
      ├── ...
```

Where `imagesVal` and `labelsVal` have the same structure as their respective train (`*Tr`) directories.

> There are two steps for generating this data structure.
> 
> (1) A user must add their dataset class to [`utils/prepare_json_data.py`](utils/prepare_json_data.py). They need to add a constant for the full path to the original data at the top of the file. They must then specify a class that inherits from `BaseDatasetJSONGenerator`. In this class, define the class variables `dir` (constant defined above), `num_seg_classes` (number of classes in labeled image), `name` (easily readable name for dataset), `modality` (list of modalities, usually `"CT"` or `"MRI"`), `labels` (dictionary mapping values in labeled image to a class name e.g., `4: "pancreas"`). Lastly, the user must define a `classmethod` called `generate` which will define how to create a dataset dictionary for their new dataset. Refer to other classes in [`utils/prepare_json_data.py`](utils/prepare_json_data.py) for reference. 
>
> After making the above changes, run [`utils/prepare_json_data.py`](utils/prepare_json_data.py) to generate `dataset.json` in your new dataset directory.
>
> (2) A user must add their class defined in step (1) to the `DATASET_LIST` in [`utils/prepare_data.py`](utils/prepare_data.py). *Sometimes if a dataset has a strange formatting, the user will need to modify the logic of [`utils/prepare_data.py`](utils/prepare_data.py).*
>
> After making the above changes, run [`data.py`](data.py) to generate the proper file structure defined above.
> 
> Then, modify the [`utils/data_list.py`](utils/data_list.py) according to your new dataset. The name of this new entry should be the `name` attribute you gave your dataset in step (1) and the sets available for the dataset (`Tr`, `Val`, `Ts`)
>
>```
>POSSIBLE_DATASETS = [
>    "WORD_Val",
>    "AMOS_Val",
>    "ONDRI_Tr",
>    ...
>]
>```


#### 2. **Run the Training Script**: 
Run [`mytrain.sh`](mytrain.sh) if running in a SLURM environment (some file modifications will be needed for your exact use case) or run [`train.py`](train.py) with the following command

```
python train.py fit ...
```

This will start the training process of the NeuroSAM3D model on your prepared data.

By default the model will use all arguments specified in [`configs/all.yaml`](configs/all.yaml). If you would like to modify any entries, either do so directly, or create a copy of the file and modify to your desire. When running with a new config file use the following command

```
python train.py fit --config <YOUR_CONFIG>.yaml
```

You can view all the possible arguments by running

```
python train.py fit --help
```

Individual arguments can be overridden by passing their flag directly to the CLI

```
python train.py fit --data.volume_threshold 1000
```


**Hint**: Use the `--model.checkpoint` to set the pre-trained weight path for the entire encoder-decoder. The model will load these weights and start training. Alternatively, use `--model.pretrained_weights` to load in encoder-only weights for the `DinoV2` model. Lastly, use `--ckpt_path` to load in a `lightning` checkpoint and resume training from this state (including step, optimizer state, hyperparams, etc.)


### Evaluation - UNTESTED

Below is all the old version of the README. We have not tested any validation with NeuroSAM3D, so it will likely need modification.


>Prepare your own dataset and refer to the samples in `data/validation` to replace them according to your specific scenario. 
>Then you can simply run `bash val.sh` to test SAM-Med3D on your data. 
>Make sure the masks are processed into the one-hot format (have only two values: the main image (foreground) and the background).
>
>```
>python validation.py --seed 2023\
> -vp ./results/vis_sam_med3d \
> -cp ./ckpt/sam_med3d_turbo.pth \
> -tdp ./data/medical_preprocessed -nc 1 \
> --save_name ./results/sam_med3d.py
>```
>
>- vp: visualization path, dir to save the final visualization files
>- cp: checkpoint path
>- tdp: test data path, where your data is placed
>- nc: number of clicks of prompt points
>- save_name: filename to save evaluation results 
>
>For validation of SAM and SAM-Med2D on 3D volumetric data, you can refer to `scripts/val_sam.sh` and `scripts/val_med2d.sh` for details.
>
>Hint: We also provide a simple script `sum_result.py` to help summarize the results from file like `./results/sam_med3d.py`. 


## 🔗 Checkpoint
**Our most recommended version is SAM-Med3D-turbo**

| Model | Google Drive | Baidu NetDisk |
|----------|----------|----------|
| SAM-Med3D | [Download](https://drive.google.com/file/d/1PFeUjlFMAppllS9x1kAWyCYUJM9re2Ub/view?usp=drive_link) | [Download (pwd:r5o3)](https://pan.baidu.com/s/18uhMXy_XO0yy3ODj66N8GQ?pwd=r5o3) |
| SAM-Med3D-organ    | [Download](https://drive.google.com/file/d/1kKpjIwCsUWQI-mYZ2Lww9WZXuJxc3FvU/view?usp=sharing) | [Download (pwd:5t7v)](https://pan.baidu.com/s/1Dermdr-ZN8NMWELejF1p1w?pwd=5t7v) |
| SAM-Med3D-brain    | [Download](https://drive.google.com/file/d/1otbhZs9uugSWkAbcQLLSmPB8jo5rzFL2/view?usp=sharing) | [Download (pwd:yp42)](https://pan.baidu.com/s/1S2-buTga9D4Nbrt6fevo8Q?pwd=yp42) |
| SAM-Med3D-turbo    | [Download](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing) | [Download (pwd:l6ol)](https://pan.baidu.com/s/1OEVtiDc6osG0l9HkQN4hEg?pwd=l6ol) |

Other checkpoints are available with their official link: [SAM](https://drive.google.com/file/d/1_U26MIJhWnWVwmI5JkGg2cd2J6MvkqU-/view?usp=drive_link) and [SAM-Med2D](https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link).

## 🗼 Method
<div align="center">
  <img src="assets/comparison.png">
</div>
<div align="center">
  <img src="assets/architecture.png">
</div>

## 🏆 Results
### 💡 Overall Performance
| **Model**    | **Prompt**   | **Resolution**                 | **Inference Time (s)** | **Overall Dice** |
|--------------|--------------|--------------------------------|------------------|------------------|
| SAM          | N points     | 1024×1024×N                    | 13               | 17.01            |
| SAM-Med2D    | N points     | 256×256×N                      | 4                | 42.75            |
| SAM-Med3D    | 1 point      | 128×128×128                    | 2                | 49.91            |
| SAM-Med3D    | 10 points    | 128×128×128                    | 6                | 60.94            |
| **SAM-Med3D-turbo** | 10 points | 128×128×128                | 6                | 77.60            |

> **Note:** Quantitative comparison of different methods on our evaluation dataset. Here, N denotes the count of slices containing the target object (10 ≤ N ≤ 200). Inference time is calculated with N=100, excluding the time for image processing and simulated prompt generation.



### 💡 Dice on Different Anatomical Architecture and Lesions
| **Model**    | **Prompt**   | **A&T** | **Bone** | **Brain** | **Cardiac** | **Gland** | **Muscle** | **Seen Lesion** | **Unseen Lesion** |
|--------------|--------------|---------|----------|-----------|-------------|-----------|------------|-----------------|-------------------|
| SAM          | N points     | 17.19   | 22.32    | 17.68     | 2.82        | 11.62     | 3.50       | 12.03           | 8.88              |
| SAM-Med2D    | N points     | 46.79   | 47.52    | 19.24     | 32.23       | 43.55     | 35.57      | 26.08           | 44.87             |
| SAM-Med3D    | 1 point      | 46.80   | 54.77    | 34.48     | 46.51       | 57.28     | 53.28      | 42.02           | 40.53             |
| SAM-Med3D    | 10 points    | 55.81   | 69.13    | 40.71     | 52.86       | 65.01     | 67.28      | 50.52           | **48.44**            |
| **SAM-Med3D-brain** | 10 points | 51.71   | -        | **62.77** | 37.93    | 62.95     | 43.70      | 45.89           | 20.51             |
| **SAM-Med3D-organ** | 10 points | 70.63   | -        | 46.49     | 63.14| **73.01** | 75.29      | 53.02           | 36.44             |
| **SAM-Med3D-turbo** | 10 points | **83.96**|**85.34**| 46.08 |	**69.90** |	**90.97**  |	**91.62**      |	**64.80** | **61.00**     |

> **Note:** Comparison from the perspective of anatomical structure and lesion. A&T represents Abdominal and Thorax targets. N denotes the count of slices containing the target object (10 ≤ N ≤ 200).


### 💡 Visualization
<div align="center">
  <img src="assets/vis_anat.png">
</div>
<div align="center">
  <img src="assets/vis_modal.png">
</div>


<!-- ## 🗓️ Ongoing 
- [ ] Dataset release
- [x] Train code release
- [x] [Feature] Evaluation on 3D data with 2D models (slice-by-slice)
- [x] Evaluation code release
- [x] Pre-trained model release
- [x] Paper release -->

## 📬 Citation
```
@misc{wang2023sammed3d,
      title={SAM-Med3D}, 
      author={Haoyu Wang and Sizheng Guo and Jin Ye and Zhongying Deng and Junlong Cheng and Tianbin Li and Jianpin Chen and Yanzhou Su and Ziyan Huang and Yiqing Shen and Bin Fu and Shaoting Zhang and Junjun He and Yu Qiao},
      year={2023},
      eprint={2310.15161},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE). 

## 💬 Discussion Group
<p align="center"><img width="100" alt="image" src="assets/QRCode.jpg"></p> 
(If the QRCode is expired, please contact the WeChat account: EugeneYonng or Small_dark8023.)

BTW, welcome to follow our [Zhihu official account](https://www.zhihu.com/people/gmai-38), we will share more information on medical imaging there.

## 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects:
  - [Segment Anything](https://github.com/facebookresearch/segment-anything) &#8194;
  - [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D/tree/main)

## 👋 Hiring & Global Collaboration
- **Hiring:** We are hiring researchers, engineers, and interns in General Vision Group, Shanghai AI Lab. If you are interested in Medical Foundation Models and General Medical AI, including designing benchmark datasets, general models, evaluation systems, and efficient tools, please contact us.
- **Global Collaboration:** We're on a mission to redefine medical research, aiming for a more universally adaptable model. Our passionate team is delving into foundational healthcare models, promoting the development of the medical community. Collaborate with us to increase competitiveness, reduce risk, and expand markets.
- **Contact:** Junjun He(hejunjun@pjlab.org.cn), Jin Ye(yejin@pjlab.org.cn), and Tianbin Li (litianbin@pjlab.org.cn).
