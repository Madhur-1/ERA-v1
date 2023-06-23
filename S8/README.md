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
_Note: We use GN with num_groups = 4_

### Metrics
|    | Train Acc | Test Acc | Train Loss | Test Loss |
|----|-----------|----------|------------|-----------|
| BN | 80.27     | 79.39    | 0.57       | 0.60      |
| GN | 76.18     | 74.84    | 0.68       | 0.72      |
| LN | 74.17     | 72.79    | 0.73       | 0.76      |

## Performance Curves
![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/26152e07-ae2a-495b-9f3c-82fb0c2cf0a4)

We see that the graphs portray BN > GN (4 groups) > LN consistently in all the training continues. We explore the reason for this in the next sections.

## Confusion Matrices

**Batch Norm | Group Norm | Layer Norm**
<div>
    <img src="https://github.com/Madhur-1/ERA-v1/assets/64495917/6cc20003-e120-4d4d-afbf-398512635fb6" width="325px" alt="image 1">
    <img src="https://github.com/Madhur-1/ERA-v1/assets/64495917/53d8861d-8b44-4e02-9788-d277cad72833" width="325px" alt="image 2">
    <img src="https://github.com/Madhur-1/ERA-v1/assets/64495917/615a69f9-35c3-4e3d-83bc-11e14dae36d1" width="325px" alt="image 3">
</div>


## Misclassified Images
**Batch Norm**

Total Incorrect Preds = 2061

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/5f376c40-f1ad-4c04-8fda-a48f68e0750f)


**Group Norm**

Total Incorrect Preds = 2516

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/71a45708-534e-4bf6-a3b9-4956fae1dc51)


**Layer Norm**

Total Incorrect Preds = 3139

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/597361a4-cfcf-412a-9e92-85f19a97d9f0)

We see that the misclassified images in all three models have classes very close to each other as misclassified. These misclassified images would be hard for a human to classify correctly too!

## Analysis
The following images have been taken from the paper:
Wu, Yuxin & He, Kaiming. (2020). Group Normalization. International Journal of Computer Vision. 128. 10.1007/s11263-019-01198-w. 


<img width="1030" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/0c107fc8-aafd-489e-92bf-a1b90c1ffc99">

The above shows the evolution of the feature distributions of the last layer of VGG-16 (can be trained w/o normalization).

We see that without normalization, the distributions tend to explode. GN and BN behave qualitatively similarly while being substantially different from the variant that uses no normalization; this phenomenon is also observed for all other convolutional layers.

<img width="557" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/8e36ce1c-ba3d-455e-a06c-259b5819f2ac">

Despite its great success, BN exhibits drawbacks that are also caused by its distinct behaviour of normalizing along the batch dimension. In particular, it is required for BN to work with a sufficiently large batch size (e.g., 32 per worker). A small batch leads to an inaccurate estimation of the batch statistics, and reducing BN’s batch size increases the model error dramatically (Figure 1).

With a batch size of 2 samples, GN has 10.6% lower error than its BN counterpart for ResNet-50 in ImageNet. With a regular batch size, GN is comparably good as BN (with a gap of ∼0.5%) and outperforms other normalization variants. Moreover, although the batch size may change, GN can naturally transfer from pre-training to fine-tuning.
