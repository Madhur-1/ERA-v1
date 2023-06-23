# Session 8

## Introduction

This assignment compares different normalization techniques: **Batch Norm, Layer Norm** and **Group Norm**.

We are presented with a multiclass classification problem on the CIFAR10 dataset.

### Target
1. Accuracy > 70%
2. Number of Parameters < 50k
3. Epochs <= 20

Use of Residual Connection is also advised.

## Implementation

<img src="https://github.com/Madhur-1/ERA-v1/assets/64495917/f1241563-94ca-4e63-ba68-32b0401741ca" width="300px" alt="image">

The above structure with two residual connections is used.

## Normalization Technique Comparison

### Metrics
|    | Train Acc | Test Acc | Train Loss | Test Loss |
|----|-----------|----------|------------|-----------|
| BN | 80.27     | 79.39    | 0.57       | 0.60      |
| GN | 76.18     | 74.84    | 0.68       | 0.72      |
| LN | 69.79     | 68.61    | 0.85       | 0.86      |

## Performance Curves
![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/6bb7621d-7ae9-4730-8a4e-0db4c563d03b)

## Confusion Matrices
**Batch Norm**

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/6cc20003-e120-4d4d-afbf-398512635fb6)

**Group Norm**

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/53d8861d-8b44-4e02-9788-d277cad72833)

**Layer Norm**

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/23cdaf9e-cfbd-4b34-a91a-219ae6687d9e)

## Misclassified Images
**Batch Norm**

Total Incorrect Preds = 2061

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/5f376c40-f1ad-4c04-8fda-a48f68e0750f)


**Group Norm**

Total Incorrect Preds = 2516

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/71a45708-534e-4bf6-a3b9-4956fae1dc51)


**Layer Norm**

Total Incorrect Preds = 3139

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/ffd45910-78ec-42ea-b9c3-f2bb271da12f)
