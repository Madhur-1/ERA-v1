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

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/4621c76f-2f3c-432a-807b-2d81ed764149)

`LR suggestion: steepest gradient
Suggested LR: 2.00E-03`

From the above figure we can see that the optimal lr is found using the steepest gradient at the 2.00E-03 point. Please note the setting for the lr_finder was the following:

```python
from torch_lr_finder import LRFinder
model = Net(dropout_percentage=0.02, norm="bn").to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = F.cross_entropy

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```

## Misclassified Images

Total Incorrect Preds = 718

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/042dae8d-e5d6-452c-82e7-6f6082a20bd5)



We see that the misclassified images in all three models have classes very close to each other as misclassified. These misclassified images would be hard for a human to classify correctly too!

## Training Log

```
Epoch 1
Train: 100% Loss=1.3700 Batch_id=97 Accuracy=37.17
Test set: Average loss: 1.4286, Accuracy: 4981/10000 (49.81%)

Epoch 2
Train: 100% Loss=0.9111 Batch_id=97 Accuracy=60.19
Test set: Average loss: 0.9862, Accuracy: 6653/10000 (66.53%)

Epoch 3
Train: 100% Loss=0.7911 Batch_id=97 Accuracy=70.60
Test set: Average loss: 0.8489, Accuracy: 7104/10000 (71.04%)

Epoch 4
Train: 100% Loss=0.7288 Batch_id=97 Accuracy=75.67
Test set: Average loss: 0.7319, Accuracy: 7571/10000 (75.71%)

Epoch 5
Train: 100% Loss=0.6057 Batch_id=97 Accuracy=78.72
Test set: Average loss: 0.7781, Accuracy: 7429/10000 (74.29%)

Epoch 6
Train: 100% Loss=0.4106 Batch_id=97 Accuracy=80.97
Test set: Average loss: 0.4810, Accuracy: 8346/10000 (83.46%)

Epoch 7
Train: 100% Loss=0.4909 Batch_id=97 Accuracy=83.93
Test set: Average loss: 0.4627, Accuracy: 8418/10000 (84.18%)

Epoch 8
Train: 100% Loss=0.3857 Batch_id=97 Accuracy=85.09
Test set: Average loss: 0.4717, Accuracy: 8450/10000 (84.50%)

Epoch 9
Train: 100% Loss=0.4860 Batch_id=97 Accuracy=86.98
Test set: Average loss: 0.3980, Accuracy: 8678/10000 (86.78%)

Epoch 10
Train: 100% Loss=0.4478 Batch_id=97 Accuracy=88.24
Test set: Average loss: 0.4256, Accuracy: 8609/10000 (86.09%)

Epoch 11
Train: 100% Loss=0.3112 Batch_id=97 Accuracy=89.44
Test set: Average loss: 0.5296, Accuracy: 8377/10000 (83.77%)

Epoch 12
Train: 100% Loss=0.3254 Batch_id=97 Accuracy=90.11
Test set: Average loss: 0.3613, Accuracy: 8841/10000 (88.41%)

Epoch 13
Train: 100% Loss=0.3124 Batch_id=97 Accuracy=90.87
Test set: Average loss: 0.3943, Accuracy: 8750/10000 (87.50%)

Epoch 14
Train: 100% Loss=0.2186 Batch_id=97 Accuracy=91.94
Test set: Average loss: 0.3726, Accuracy: 8779/10000 (87.79%)

Epoch 15
Train: 100% Loss=0.2087 Batch_id=97 Accuracy=92.40
Test set: Average loss: 0.4073, Accuracy: 8747/10000 (87.47%)

Epoch 16
Train: 100% Loss=0.1622 Batch_id=97 Accuracy=93.09
Test set: Average loss: 0.3162, Accuracy: 9003/10000 (90.03%)

Epoch 17
Train: 100% Loss=0.1884 Batch_id=97 Accuracy=93.73
Test set: Average loss: 0.2952, Accuracy: 9066/10000 (90.66%)

Epoch 18
Train: 100% Loss=0.1617 Batch_id=97 Accuracy=94.28
Test set: Average loss: 0.2963, Accuracy: 9053/10000 (90.53%)

Epoch 19
Train: 100% Loss=0.1487 Batch_id=97 Accuracy=94.99
Test set: Average loss: 0.3057, Accuracy: 9081/10000 (90.81%)

Epoch 20
Train: 100% Loss=0.1347 Batch_id=97 Accuracy=95.70
Test set: Average loss: 0.2795, Accuracy: 9151/10000 (91.51%)

Epoch 21
Train: 100% Loss=0.0768 Batch_id=97 Accuracy=96.08
Test set: Average loss: 0.2616, Accuracy: 9221/10000 (92.21%)

Epoch 22
Train: 100% Loss=0.0721 Batch_id=97 Accuracy=96.82
Test set: Average loss: 0.2476, Accuracy: 9244/10000 (92.44%)

Epoch 23
Train: 100% Loss=0.0667 Batch_id=97 Accuracy=97.18
Test set: Average loss: 0.2417, Accuracy: 9265/10000 (92.65%)

Epoch 24
Train: 100% Loss=0.0868 Batch_id=97 Accuracy=97.60
Test set: Average loss: 0.2367, Accuracy: 9282/10000 (92.82%)
```
