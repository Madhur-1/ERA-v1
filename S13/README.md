# Session 13

## Introduction

This assignment is focused towards understanding the YOLOv3 and training it from strach while also using pytorch lightning and implementing grad-cam.

### Target
1. Port training code to Pytorch Lightning
2. Achieve:
   1. Class Acc >= 80
   2. No Obj Acc >= 98
   3. Object Acc >= 78
3. Use Gradio to create a simple app visualizing the bounding box output and GradCAM.
4. Host the app using HuggingFace Spaces.

**Link to HuggingFace Space:** `https://huggingface.co/spaces/madhurjindal/object-detection-yolov3-gradcam`

## Structure

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/367fbb1a-c284-4c8a-84f4-629d4a64e025)


### Metrics
| Class Acc | No Obj Acc |   Obj Acc  | MAP       | Train Loss | Test Loss |
|-----------|------------|------------|-----------|------------|-----------|
| 88.99     | 98.19      | 77.58      | 0.43      | 3.19       | 2.73      |

Note: The above loss values use lambda_class = 1, lambda_noobj = 5, lambda_obj = 1, lambda_box = 5.

## Data Exploration

<img width="451" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/865607f0-1640-4adb-aa28-79ef90e833c7">
<img width="437" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/662c852b-d821-4e44-b809-68b6fc797318">


```python
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ],
)
```

## LR Finder

<img width="568" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/a0d432c6-363c-4a4e-b383-61668fe1d322">



`LR suggestion: steepest gradient
Suggested LR: 0.003981071705534973`

From the above figure we can see that the optimal lr is found using the steepest gradient at the 0.003981071705534973 point. Please note the setting for the lr_finder was the following:


## Training Log

```
EPOCH: 0, Loss: 13.572253227233887
EPOCH: 1, Loss: 9.932060241699219
EPOCH: 2, Loss: 8.969964027404785

Class accuracy is: 44.560822%
No obj accuracy is: 93.487915%
Obj accuracy is: 58.498199%
EPOCH: 3, Loss: 8.279768943786621
```
<img width="437" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/eca87c33-6386-47ac-9aec-45061da93faa">

```
EPOCH: 4, Loss: 7.834333419799805
EPOCH: 5, Loss: 7.4485907554626465
EPOCH: 6, Loss: 7.1627092361450195

Class accuracy is: 48.725201%
No obj accuracy is: 96.308342%
Obj accuracy is: 61.208828%

+++ TEST ACCURACIES
Class accuracy is: 57.960655%
No obj accuracy is: 97.526146%
Obj accuracy is: 52.067055%

EPOCH: 7, Loss: 7.011684417724609
EPOCH: 8, Loss: 6.7238287925720215
```

<img width="444" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/fed8a3cf-0c99-46d0-be10-3e0ed754b154">

```
EPOCH: 9, Loss: 6.542272090911865
EPOCH: 10, Loss: 6.382115364074707

+++ TEST ACCURACIES
Class accuracy is: 63.951229%
No obj accuracy is: 96.849358%
Obj accuracy is: 66.733170%
EPOCH: 11, Loss: 6.283266067504883
EPOCH: 12, Loss: 6.148199081420898
EPOCH: 13, Loss: 6.106574058532715
```

<img width="442" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/962b03f2-5b11-497a-85f5-909498e24a58">

```
EPOCH: 14, Loss: 6.001038074493408
+++ TRAIN ACCURACIES
Class accuracy is: 56.644726%
No obj accuracy is: 96.247818%
Obj accuracy is: 69.856743%
+++ TEST ACCURACIES
Class accuracy is: 67.403717%
No obj accuracy is: 97.583992%
Obj accuracy is: 63.067329%
EPOCH: 15, Loss: 5.91851806640625
EPOCH: 16, Loss: 5.802373886108398
EPOCH: 17, Loss: 5.707857608795166
EPOCH: 18, Loss: 5.567074775695801
```

<img width="434" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/2a234a92-3f0f-41d6-afd0-a4d973a1ef4a">

```
+++ TEST ACCURACIES
Class accuracy is: 66.566917%
No obj accuracy is: 98.052994%
Obj accuracy is: 60.052647%
EPOCH: 19, Loss: 5.447572708129883
EPOCH: 20, Loss: 5.322601795196533
EPOCH: 21, Loss: 5.195068836212158
EPOCH: 22, Loss: 5.106668949127197
+++ TRAIN ACCURACIES
Class accuracy is: 67.076813%
No obj accuracy is: 95.820015%
Obj accuracy is: 77.327820%
+++ TEST ACCURACIES
Class accuracy is: 76.968689%
No obj accuracy is: 97.055359%
Obj accuracy is: 74.319756%
EPOCH: 23, Loss: 4.980020999908447
```

<img width="442" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/f9169ae8-9ed7-45e0-9d9e-44787b489fde">

```
EPOCH: 24, Loss: 4.837049961090088
EPOCH: 25, Loss: 4.764085292816162
EPOCH: 26, Loss: 4.589331150054932
+++ TEST ACCURACIES
Class accuracy is: 79.495705%
No obj accuracy is: 97.869995%
Obj accuracy is: 72.579666%
EPOCH: 27, Loss: 4.507960796356201
EPOCH: 28, Loss: 4.408302307128906
```

<img width="435" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/9b19b8c3-f1c9-4e4f-acd6-6cee7cbc603c">

```
EPOCH: 29, Loss: 4.3071184158325195
EPOCH: 30, Loss: 4.165201187133789
+++ TRAIN ACCURACIES
Class accuracy is: 76.161743%
No obj accuracy is: 96.402809%
Obj accuracy is: 81.299088%
+++ TEST ACCURACIES
Class accuracy is: 84.477692%
No obj accuracy is: 97.785561%
Obj accuracy is: 76.201164%
EPOCH: 31, Loss: 4.073633670806885
EPOCH: 32, Loss: 3.936971664428711
EPOCH: 33, Loss: 3.8076586723327637
```

<img width="440" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/e20eaeac-db83-4ea0-b02f-452ac564b9a2">

```
EPOCH: 34, Loss: 3.6981098651885986
+++ TEST ACCURACIES
Class accuracy is: 86.320312%
No obj accuracy is: 97.800079%
Obj accuracy is: 77.802719%
EPOCH: 35, Loss: 3.5974011421203613
EPOCH: 36, Loss: 3.4911625385284424
EPOCH: 37, Loss: 3.3553738594055176
EPOCH: 38, Loss: 3.260648250579834
```

<img width="436" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/09a0b520-a69b-410d-a5a8-5670cc5611d9">

```
+++ TRAIN ACCURACIES
Class accuracy is: 83.479492%
No obj accuracy is: 96.957603%
Obj accuracy is: 82.855469%
+++ TEST ACCURACIES
Class accuracy is: 88.994179%
No obj accuracy is: 98.185600%
Obj accuracy is: 77.581047%
+++ MAP:  0.4297582507133484
EPOCH: 39, Loss: 3.1944639682769775
```
