from torchvision import transforms

# Train data transformations
train_transforms = transforms.Compose(
    [
        # transforms.RandomApply(
        #     [
        #         transforms.CenterCrop(28),
        #     ],
        #     p=0.1,
        # ),
        # transforms.Resize((32, 32)),
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomRotation((-7.0, 7.0), fill=255),
        # transforms.RandomApply(
        #     [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)], p=0.1
        # ),
        # transforms.TrivialAugmentWide(),
        transforms.RandomApply([transforms.RandAugment()], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

# Test data transformations
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)
