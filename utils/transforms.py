import monai.transforms


def get_train_trainsforms(img_size: int) -> monai.transforms.Compose:
    return monai.transforms.Compose(
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
                spatial_size=[img_size, img_size, img_size],
                pos=1,
                neg=0,
                num_samples=1,
                allow_smaller=True,
            ),
            # ensure that we have the right shape in the end
            monai.transforms.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=[img_size, img_size, img_size],
            ),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )

def get_val_transforms(img_size: int) -> monai.transforms.Compose:
    return monai.transforms.Compose(
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