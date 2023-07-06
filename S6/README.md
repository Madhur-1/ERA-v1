# Session 6

## Introduction

This assignment is focussed towards grasping Backpropagation and Architectural Basics. The assignment is divided into 2 parts as described below.

## Part 1

This part involves using MS Excel to perform backprop on a arbitrary defined Neural network.

![Alt text](image.png)

The above screenshot depicts the training stat while varying the learning rates. We see that as we increase the learning rates we reach the saturation point faster.

The directory also contains `Screenshot 1.png` and `Screenshot 2.png` depicting the working of the backprop algorithm.

## Part 2

### Target
1. Accuracy > 99.4%
2. Number of Parameters < 20k
3. Num Epochs < 20

### Structure

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
            Conv2d-4           [-1, 16, 24, 24]           2,320
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
            Conv2d-7           [-1, 30, 22, 22]           4,350
              ReLU-8           [-1, 30, 22, 22]               0
       BatchNorm2d-9           [-1, 30, 22, 22]              60
           Conv2d-10           [-1, 30, 11, 11]             930
             ReLU-11           [-1, 30, 11, 11]               0
      BatchNorm2d-12           [-1, 30, 11, 11]              60
           Conv2d-13             [-1, 16, 9, 9]           4,336
             ReLU-14             [-1, 16, 9, 9]               0
      BatchNorm2d-15             [-1, 16, 9, 9]              32
           Conv2d-16             [-1, 16, 7, 7]           2,320
             ReLU-17             [-1, 16, 7, 7]               0
      BatchNorm2d-18             [-1, 16, 7, 7]              32
           Conv2d-19             [-1, 32, 5, 5]           4,640
             ReLU-20             [-1, 32, 5, 5]               0
      BatchNorm2d-21             [-1, 32, 5, 5]              64
        AvgPool2d-22             [-1, 32, 1, 1]               0
           Linear-23                   [-1, 10]             330
================================================================
Total params: 19,698
Trainable params: 19,698
Non-trainable params: 0
----------------------------------------------------------------
```

### Performance Curve
![Alt text](image-1.png)