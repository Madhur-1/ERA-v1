import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train data transformations
train_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True, border_mode=0),
        A.RandomCrop(height=32, width=32, always_apply=True),
        # A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=64, min_width=64, always_apply=True, border_mode=0),
        A.CoarseDropout(
            p=0.2,
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=(0.4914, 0.4822, 0.4465),
            mask_fill_value=None,
        ),
        A.CenterCrop(height=32, width=32, always_apply=True),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ToTensorV2(),
    ]
)

# Test data transformations
test_transforms = A.Compose(
    [
        A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ToTensorV2(),
    ]
)
