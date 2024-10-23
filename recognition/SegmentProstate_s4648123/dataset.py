from monai.transforms import (Compose, ToTensord, RandFlipd, Spacingd, RandScaleIntensityd, RandShiftIntensityd,
                              NormalizeIntensityd, DivisiblePadd, LoadImaged, EnsureChannelFirstd,
                              ScaleIntensityRanged, CropForegroundd, Orientationd, RandCropByPosNegLabeld)

#Transforms to be applied on training instances
train_transform = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)

#Transforms to be applied on validation instances
val_transform = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)

# test other transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        ToTensord(keys=["image", "label"], device="cuda")
        # can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),

    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ToTensord(keys=["image", "label"], device="cuda")
    ]
)