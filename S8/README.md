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
The following content have been taken from the paper:
Wu, Yuxin & He, Kaiming. (2020). Group Normalization. International Journal of Computer Vision. 128. 10.1007/s11263-019-01198-w. 

<img width="1108" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/9afaece1-dbcc-416d-93b6-174388728e9b">

### BN
- Batch Normalization (BN) is a milestone technique in the development of deep learning, enabling various networks to train. However, normalizing along the batch dimension introduces problems — BN’s error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation. This limits BN’s usage for training larger models and transferring features to computer vision tasks including detection, segmentation, and video, which require small batches constrained by memory consumption.
- BN normalizes the features by the mean and variance computed within a (mini)batch. This has been shown by many practices to ease optimization and enable very deep networks to converge. The stochastic uncertainty of the batch statistics also acts as a regularizer that can benefit generalization.
- But the concept of “batch” is not always present, or it may change from time to time. For example, batch-wise normalization is not legitimate at inference time, so the mean and variance are pre-computed from the training set, often by running average; consequently, there is no normalization performed when testing. The pre-computed statistics may also change when the target data distribution changes. These issues lead to in-consistency at training, transferring, and testing time.

### Reasoning for GN
- The channels of visual representations are not entirely independent.
- It is not necessary to think of deep neural network features as unstructured vectors. For example, for conv1 (the first convolutional layer) of a network, it is reasonable to expect a filter and its horizontal flipping to exhibit similar distributions of filter responses on natural images. If conv1 happens to approximately learn this pair of filters, or if the horizontal flipping (or other transformations) is made into the architectures by design, then the corresponding channels of these filters can be normalized together.
- For e.g. if the layer learns Horizontal and Vertical edge detectors, they could be grouped together.
![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/0a2112b6-743f-47e9-be2a-43f7648b3df6)

<img width="548" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/b15bc75d-028d-4757-ab87-1048b69f1080">

-Specifically, the pixels in the same group are normalized together by the same μ and σ. GN also learns the per-channel γ and β.

### GN
**Relation to LN** `If we set the number of groups to 1, GN becomes LN. LN assumes all channels in a layer make similar contributions and thus restrict the system. GN tries to mitigate this becaquse only each group shares common mean and variance.`

**Relation to IN** `GN becomes IN if we set the number of groups to C (1 group per channel). But IN only relies on the spatial dimension for computing the mean and variance and thus misses the opportunity of exploiting the channel dependence.`

**Effect of Group Number**
- In the extreme case of G = 1, GN is equivalent to LN, and its error rate is higher than all cases of G > 1 studied.
- In the extreme case of 1 channel per group, GN is equivalent to IN. Even if using as few as 2 channels per group, GN has substantially lower error than IN (25.6% vs. 28.4%). This result shows the effect of grouping channels when performing normalization.
<img width="470" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/2ce33587-bbac-4f78-9cba-aca48a51ccf8">


### Why Normalization?
<img width="1030" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/0c107fc8-aafd-489e-92bf-a1b90c1ffc99">

The above shows the evolution of the feature distributions of the last layer of VGG-16 (can be trained w/o normalization).

We see that without normalization, the distributions tend to explode. GN and BN behave qualitatively similarly while being substantially different from the variant that uses no normalization; this phenomenon is also observed for all other convolutional layers.

### Effect of Batch Size for BN
<img width="557" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/8e36ce1c-ba3d-455e-a06c-259b5819f2ac">

Despite its great success, BN exhibits drawbacks that are also caused by its distinct behaviour of normalizing along the batch dimension. In particular, it is required for BN to work with a sufficiently large batch size (e.g., 32 per worker). A small batch leads to an inaccurate estimation of the batch statistics, and reducing BN’s batch size increases the model error dramatically (Figure 1).

With a batch size of 2 samples, GN has 10.6% lower error than its BN counterpart for ResNet-50 in ImageNet. With a regular batch size, GN is comparably good as BN (with a gap of ∼0.5%) and outperforms other normalization variants. Moreover, although the batch size may change, GN can naturally transfer from pre-training to fine-tuning.

<img width="473" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/b10485d0-1abf-43ad-80b5-736e95c10e9f">

`We see that for larger batch sizes BN still wins the show which happens in our case as we use a batch size of 64.`

### Parameter Analysis
Please refer to `parameter_analysis.ipynb`.

