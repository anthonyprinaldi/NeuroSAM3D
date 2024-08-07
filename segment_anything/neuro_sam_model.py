from pathlib import Path
from typing import List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from utils.click_method import (get_next_click3D_torch_2,
                                get_next_click3D_torch_largest_blob)

from .build_sam3D import sam_model_registry3D


class NeuroSamModel(L.LightningModule):
    def __init__(self,
                 model_type: str,
                 work_dir: str,
                 task_name: str,
                 lr: float,
                 weight_decay: float,
                 lr_scheduler: str,
                 step_size: Union[int, List[int]],
                 gamma: float,
                 largest_first: bool,
                 click_type: str,
                 multi_click: bool,
                 img_size: int,
                 bbox_first: bool,
                 num_clicks: int,
                 logging_batches_idx: List[int],
                 checkpoint: Optional[str] = None,
                 ):
        """Initialize the trainer.

        :param model_type: String used to select the model from the registry.
        :type model_type: str
        :param work_dir: Directory to save the model and logs.
        :type work_dir: str
        :param task_name: Name of the run.
        :type task_name: str
        :param lr: Learning rate for optimizer.
        :type lr: float
        :param weight_decay: Weight decay for optimizer.
        :type weight_decay: float
        :param lr_scheduler: Learning rate scheduler to use.
        :type lr_scheduler: str
        :param step_size: Step size of the scheduler.
        :type step_size: Union[int, List[int]]
        :param gamma: Gamma of the scheduler.
        :type gamma: float
        :param largest_first: Whether to select the largest blob as the first point prompt.
        :type largest_first: bool
        :param click_type: Click method to use.
        :type click_type: str
        :param multi_click: Whether to stack previous clicks for the current prompt.
        :type multi_click: bool
        :param img_size: Size of the input image.
        :type img_size: int
        :param bbox_first: Whether to use bounding boxes as the first prompt.
        :type bbox_first: bool
        :param num_clicks: Number of clicks to use per training sample.
        :type num_clicks: int
        :param logging_batches_idx: List of batch indices to log images for.
        :type logging_batches_idx: List[int]
        :param checkpoint: Path to the checkpoint to load weights from.
        :type checkpoint: Optional[str]
        """
        super().__init__()

        self.model_type = model_type
        self.work_dir = work_dir
        self.task_name = task_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.largest_first = largest_first
        self.click_type = click_type
        self.multi_click = multi_click
        self.img_size = img_size
        self.bbox_first = bbox_first
        self.num_clicks = num_clicks
        self.logging_batches_idx = logging_batches_idx

        self.model = sam_model_registry3D[model_type](checkpoint=None, image_size=img_size)

        if checkpoint is not None:
            self = NeuroSamModel.load_from_checkpoint(checkpoint)
        # self.best_loss = np.inf
        # self.best_dice = 0.0
        # self.step_best_loss = np.inf
        # self.step_best_dice = 0.0
        # self.losses = []
        # self.dices = []
        # self.ious = []
        self.set_loss_fn()
        # self.set_optimizer()
        # self.set_lr_scheduler()
        # self.init_checkpoint(
        #     Path(self.args.work_dir) / self.args.task_name / "sam_model_latest.pth"
        # )
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

        self.click_methods = {
            "random": get_next_click3D_torch_2,
            "largest": get_next_click3D_torch_largest_blob,
        }

        self.LOG_OUT_DIR: Path = Path(work_dir) / task_name
        self.LOG_OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_SAVE_PATH: Path = Path(work_dir) / task_name
        self.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.image_encoder.parameters()
                },
                {
                    "params": self.model.prompt_encoder.parameters(),
                    "lr": self.lr * 0.1,
                },
                {
                    "params": self.model.mask_decoder.parameters(),
                    "lr": self.lr * 0.1,
                },
            ],
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )

        if self.lr_scheduler == "multisteplr":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.step_size, self.gamma
            )
        elif self.lr_scheduler == "steplr":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.step_size[0], self.gamma
            )
        elif self.lr_scheduler == "coswarm":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0 = 10,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1)

        return [optimizer], [lr_scheduler]
    

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")


    def batch_forward(
        self, sam_model, image_embedding, gt3D, low_res_masks, points=None, boxes=None,
    ):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=low_res_masks,
        )
        # in case sparse embeddings did not move devices if we
        # don't use points or boxes
        sparse_embeddings = sparse_embeddings.to(self.device)

        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(
            low_res_masks, size=gt3D.shape[-3:], mode="trilinear", align_corners=False
        )
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D, num_click):
        if num_click == 0 and self.largest_first:
            batch_points, batch_labels = self.click_methods["largest"](
                prev_masks, gt3D
            )
        else:
            batch_points, batch_labels = self.click_methods[self.click_type](
                prev_masks, gt3D
            )

        points_co = torch.cat(batch_points, dim=0)
        points_la = torch.cat(batch_labels, dim=0)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1)
        labels_multi = torch.cat(self.click_labels, dim=1)

        if self.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input
    

    def get_boxes(self, prev_masks, gt3D):
        boxes_coords = []
        
        for i in range(gt3D.shape[0]):
            x, y, z = torch.nonzero(gt3D[i, 0] > 0, as_tuple=True)
            start = torch.tensor([x.min(), y.min(), z.min()])
            end = torch.tensor([x.max(), y.max(), z.max()])

            boxes_coords.append(torch.stack([start, end], dim=0))

        boxes_coords = torch.stack(boxes_coords, dim=0)

        return boxes_coords.to(gt3D.device)


    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(self.device)
        low_res_masks = F.interpolate(
            prev_masks.float(),
            size=(self.img_size // 2, self.img_size // 2, self.img_size // 2),
        )

        boxes = self.get_boxes(prev_masks, gt3D) if self.bbox_first else None

        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            
            if num_click == 0 and self.bbox_first:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    gt3D,
                    low_res_masks,
                    boxes=boxes,
                    points=None,
                )
                continue
            
            points_input, labels_input = self.get_points(prev_masks, gt3D, num_click)

            # make sure we are not flowing gradients through multiple clicks
            points_input.detach()
            labels_input.detach()

            if num_click == random_insert: # TODO: why on last iter?
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model, image_embedding, gt3D, low_res_masks, points=None, boxes=None,
                )
            else:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model,
                    image_embedding,
                    gt3D,
                    low_res_masks,
                    points=[points_input, labels_input],
                    boxes=boxes,
                )
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss

    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = mask_pred > mask_threshold
            mask_gt = mask_gt > 0

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = prev_masks > 0.5
        true_masks = gt3D > 0
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()
    
    def prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        return batch["image"], batch["label"]
    
    def _step(self, batch, batch_idx, log_images=False, threshold=0.5):
        image, gt = self.prepare_batch(batch)

        image = self.norm_transform(image.squeeze(dim=1))  # (N, C, W, H, D)
        image = image.unsqueeze(dim=1)

        image_embedding = self.model.image_encoder(image)

        # reset click points and labels
        self.click_points = []
        self.click_labels = []

        mask_pred, loss = self.interaction(self.model, image_embedding, gt, num_clicks=self.num_clicks)

        dice = self.get_dice_score(mask_pred, gt)

        if log_images:
            self.logger.log_image(
                key=f"sample data epoch-{self.current_epoch} batch-{batch_idx}",
                images=[
                    image[0].cpu().detach().type(torch.float32).numpy()[:, image.shape[-2] // 2, :],
                    gt[0, 0, ...].cpu().detach().type(torch.float32).numpy()[:, gt.shape[-2] // 2, :],
                    mask_pred[0, 0, ...].cpu().detach().type(torch.float32).numpy()[:, mask_pred.shape[-2] // 2, :] > threshold,
                ],
                caption=["val_image", "val_gt", "val_pred"],
            )

        return loss, dice

    
    def training_step(self, batch, batch_idx):

        loss, dice = self._step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch["image"].shape[0], sync_dist=True)
        self.log("train_dice", dice, on_step=True, on_epoch=True, batch_size=batch["image"].shape[0], sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        loss, dice = self._step(batch, batch_idx, log_images=(batch_idx in self.logging_batches_idx))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch["image"].shape[0], sync_dist=True)
        self.log("val_dice", dice, on_epoch=True, batch_size=batch["image"].shape[0], sync_dist=True)

        return loss