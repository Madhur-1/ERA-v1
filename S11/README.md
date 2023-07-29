# Session 11

## Introduction

This assignment is focussed towards using GradCAM to identify the parts of an input image that most impact the classification score.

### Target
1. ResNet 18
2. GradCAM usage

## Structure

<img width="397" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/9706b3c7-8d6b-4c28-a144-37268c320139">

### Metrics
| Train Acc | Test Acc | Train Loss | Test Loss |
|-----------|----------|------------|-----------|
| 98.93     | 91.26    | 0.04       | 0.34      |


## Performance Curve
![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/7f298fb5-c258-457c-9875-88d6fe0420ed)



## Confusion Matrix

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/5e8afe1a-5c3b-4b4f-9377-ded0d2cf00cb)



## Data Exploration

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/e0d4b688-0aff-4d43-80d9-9020451cbe5f)



```python
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
```

As seen above, three transforms from the Albumentations library RandomCrop and CourseDropout were used.

## LR Finder

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/e699899f-2179-4946-bacf-202f7fbb156c)


`LR suggestion: steepest gradient
Suggested LR: 1.91E-03`

From the above figure we can see that the optimal lr is found using the steepest gradient.

## Misclassified Images

Total Incorrect Preds = 859

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/0ade94d4-ccc4-4163-ac8e-64b751c43bd0)


We see that the misclassified images in all three models have classes very close to each other as misclassified. These misclassified images would be hard for a human to classify correctly too!

## GradCAM

### GradCAM for the same image for different grad targets

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/9a273b3c-ca32-4797-9dfa-1aa1dda7c32c)

We can see that for the given Bird image there are different parts of the image that can be used to support different images.
1. The birds wings and the sky support the Airplane class
2. The Bird class is supported by the body of the bird
3. The legs support Cat class
4. The neck area support Dog class
5. The neck and face area supports the Horse class

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/7fb38da7-c748-4e28-a29c-3a5958aa0b7a)

Similarly:
1. The Airplane class is supported the body
2. Automobile class is supported by the components like the body, the wings etc.
3. The bird class is supported by the beak of the plane.

### GradCAM for Misclassified images

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/0e89b0c4-f09d-4edc-b7d5-69308fc28668)

In the above set of pictures say for the first Truck we can see that the hood of the trunk supports the predicted automobile class while the whole body supports the target class - This seems fair. The same pattern of identifying objects using different parts of the image is depicted in the GradCAM images.

## Training Log

```
Epoch 1
Train: 100% Loss=1.3532 Batch_id=97 Accuracy=39.81
Test set: Average loss: 1.6397, Accuracy: 4259/10000 (42.59%)

Epoch 2
Train: 100% Loss=0.8659 Batch_id=97 Accuracy=59.58
Test set: Average loss: 1.1223, Accuracy: 6241/10000 (62.41%)

Epoch 3
Train: 100% Loss=0.9156 Batch_id=97 Accuracy=69.09
Test set: Average loss: 1.0250, Accuracy: 6755/10000 (67.55%)

Epoch 4
Train: 100% Loss=0.5482 Batch_id=97 Accuracy=75.17
Test set: Average loss: 1.5539, Accuracy: 5866/10000 (58.66%)

Epoch 5
Train: 100% Loss=0.5817 Batch_id=97 Accuracy=78.20
Test set: Average loss: 0.9223, Accuracy: 7010/10000 (70.10%)

Epoch 6
Train: 100% Loss=0.4804 Batch_id=97 Accuracy=81.42
Test set: Average loss: 0.7703, Accuracy: 7447/10000 (74.47%)

Epoch 7
Train: 100% Loss=0.4788 Batch_id=97 Accuracy=84.03
Test set: Average loss: 0.6842, Accuracy: 7647/10000 (76.47%)

Epoch 8
Train: 100% Loss=0.3154 Batch_id=97 Accuracy=86.15
Test set: Average loss: 0.5369, Accuracy: 8234/10000 (82.34%)

Epoch 9
Train: 100% Loss=0.3233 Batch_id=97 Accuracy=87.84
Test set: Average loss: 0.6117, Accuracy: 8015/10000 (80.15%)

Epoch 10
Train: 100% Loss=0.3392 Batch_id=97 Accuracy=89.26
Test set: Average loss: 0.4898, Accuracy: 8392/10000 (83.92%)

Epoch 11
Train: 100% Loss=0.3067 Batch_id=97 Accuracy=90.58
Test set: Average loss: 0.5270, Accuracy: 8374/10000 (83.74%)

Epoch 12
Train: 100% Loss=0.2675 Batch_id=97 Accuracy=91.58
Test set: Average loss: 0.4969, Accuracy: 8450/10000 (84.50%)

Epoch 13
Train: 100% Loss=0.1699 Batch_id=97 Accuracy=92.84
Test set: Average loss: 0.4825, Accuracy: 8514/10000 (85.14%)

Epoch 14
Train: 100% Loss=0.1933 Batch_id=97 Accuracy=93.79
Test set: Average loss: 0.4436, Accuracy: 8691/10000 (86.91%)

Epoch 15
Train: 100% Loss=0.1242 Batch_id=97 Accuracy=94.84
Test set: Average loss: 0.4924, Accuracy: 8696/10000 (86.96%)

Epoch 16
Train: 100% Loss=0.0934 Batch_id=97 Accuracy=95.74
Test set: Average loss: 0.3843, Accuracy: 8928/10000 (89.28%)

Epoch 17
Train: 100% Loss=0.0541 Batch_id=97 Accuracy=96.76
Test set: Average loss: 0.4266, Accuracy: 8853/10000 (88.53%)

Epoch 18
Train: 100% Loss=0.0710 Batch_id=97 Accuracy=97.70
Test set: Average loss: 0.3712, Accuracy: 9031/10000 (90.31%)

Epoch 19
Train: 100% Loss=0.0269 Batch_id=97 Accuracy=98.39
Test set: Average loss: 0.3610, Accuracy: 9094/10000 (90.94%)

Epoch 20
Train: 100% Loss=0.0440 Batch_id=97 Accuracy=98.76
Test set: Average loss: 0.3454, Accuracy: 9141/10000 (91.41%)
```
