# Session 9

## Introduction

This assignment is focussed towards using Dilation and Depthwise separable convolutions to achieve the following target.

### Target
1. Accuracy > 85%
2. Number of Parameters < 200k
3. RF > 44

## Structure

<img width="310" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/b67961de-de6f-4767-aa47-010640854b8f">

1. Given the above structure we achieve a RF of 75.
2. We use 1 dilated kernel Conv2d-34 with a dilation of 2.
3. 3 Depthwise separable convolutions Conv2d-49,50,51 are used.


### Metrics
| Train Acc | Test Acc | Train Loss | Test Loss |
|-----------|----------|------------|-----------|
| 86.75     | 85.43    | 0.38       | 0.44      |


## Performance Curve
![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/b28d3855-1d60-443b-a7b3-ebcf46df50e1)

## Confusion Matrix

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/9c27e104-4cd7-43f8-8346-259ee755f38a)

## Data Exploration

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/ccfd4c54-b52a-4981-8026-e5f87ceefb3f)

```python
# Train data transformations
train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
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

As seen above, three transforms from the Albumentations library HoriznotalFlip, ShiftScaleRotate and CourseDropout were used. Note that for CourseDropout the strategy taught in class involving a pipeline Pad->CoarseDropout->Crop is followed.

## Misclassified Images

Total Incorrect Preds = 1457

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/c16bb413-761c-4f71-b970-562f7199d347)


We see that the misclassified images in all three models have classes very close to each other as misclassified. These misclassified images would be hard for a human to classify correctly too!

## Training Log

```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: 100% Loss=1.4956 Batch_id=781 Accuracy=39.97
Test set: Average loss: 1.2578, Accuracy: 5274/10000 (52.74%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: 100% Loss=0.7876 Batch_id=781 Accuracy=59.25
Test set: Average loss: 1.0200, Accuracy: 6448/10000 (64.48%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: 100% Loss=0.9778 Batch_id=781 Accuracy=67.66
Test set: Average loss: 0.8183, Accuracy: 7160/10000 (71.60%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: 100% Loss=0.9351 Batch_id=781 Accuracy=72.50
Test set: Average loss: 0.6751, Accuracy: 7659/10000 (76.59%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: 100% Loss=0.7785 Batch_id=781 Accuracy=75.87
Test set: Average loss: 0.6567, Accuracy: 7712/10000 (77.12%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: 100% Loss=0.8927 Batch_id=781 Accuracy=77.46
Test set: Average loss: 0.6362, Accuracy: 7852/10000 (78.52%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: 100% Loss=0.5647 Batch_id=781 Accuracy=79.15
Test set: Average loss: 0.5648, Accuracy: 8076/10000 (80.76%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 8
Train: 100% Loss=0.3892 Batch_id=781 Accuracy=80.39
Test set: Average loss: 0.5478, Accuracy: 8151/10000 (81.51%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 9
Train: 100% Loss=0.3857 Batch_id=781 Accuracy=83.91
Test set: Average loss: 0.4572, Accuracy: 8411/10000 (84.11%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 10
Train: 100% Loss=0.8276 Batch_id=781 Accuracy=84.67
Test set: Average loss: 0.4582, Accuracy: 8428/10000 (84.28%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 11
Train: 100% Loss=0.0763 Batch_id=781 Accuracy=85.11
Test set: Average loss: 0.4516, Accuracy: 8468/10000 (84.68%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 12
Train: 100% Loss=0.4092 Batch_id=781 Accuracy=85.17
Test set: Average loss: 0.4462, Accuracy: 8480/10000 (84.80%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 13
Train: 100% Loss=0.3290 Batch_id=781 Accuracy=85.51
Test set: Average loss: 0.4437, Accuracy: 8520/10000 (85.20%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 14
Train: 100% Loss=0.3941 Batch_id=781 Accuracy=85.66
Test set: Average loss: 0.4420, Accuracy: 8485/10000 (84.85%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 15
Train: 100% Loss=0.6335 Batch_id=781 Accuracy=86.00
Test set: Average loss: 0.4475, Accuracy: 8491/10000 (84.91%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 16
Train: 100% Loss=0.1034 Batch_id=781 Accuracy=86.14
Test set: Average loss: 0.4407, Accuracy: 8542/10000 (85.42%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 17
Train: 100% Loss=0.2676 Batch_id=781 Accuracy=86.70
Test set: Average loss: 0.4395, Accuracy: 8539/10000 (85.39%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 18
Train: 100% Loss=0.0958 Batch_id=781 Accuracy=86.41
Test set: Average loss: 0.4346, Accuracy: 8533/10000 (85.33%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 19
Train: 100% Loss=0.2886 Batch_id=781 Accuracy=86.75
Test set: Average loss: 0.4358, Accuracy: 8544/10000 (85.44%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 20
Train: 100% Loss=0.9737 Batch_id=781 Accuracy=86.82
Test set: Average loss: 0.4367, Accuracy: 8536/10000 (85.36%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 21
Train: 100% Loss=0.5213 Batch_id=781 Accuracy=86.70
Test set: Average loss: 0.4346, Accuracy: 8545/10000 (85.45%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 22
Train: 100% Loss=0.7300 Batch_id=781 Accuracy=86.70
Test set: Average loss: 0.4344, Accuracy: 8553/10000 (85.53%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 23
Train: 100% Loss=0.1984 Batch_id=781 Accuracy=86.84
Test set: Average loss: 0.4331, Accuracy: 8539/10000 (85.39%)

Adjusting learning rate of group 0 to 1.0000e-04.
Epoch 24
Train: 100% Loss=0.7466 Batch_id=781 Accuracy=86.83
Test set: Average loss: 0.4321, Accuracy: 8547/10000 (85.47%)

Adjusting learning rate of group 0 to 1.0000e-05.
Epoch 25
Train: 100% Loss=0.3672 Batch_id=781 Accuracy=86.75
Test set: Average loss: 0.4355, Accuracy: 8543/10000 (85.43%)

Adjusting learning rate of group 0 to 1.0000e-05.
```
