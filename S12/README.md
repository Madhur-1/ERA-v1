# Session 12

## Introduction

This assignment is focused towards getting acquainted with Pytorch Lightning, Gradio and HuggingFace🤗 Spaces.

### Target
1. Port training code to Pytorch Lightning
2. Use Gradio to create a simple app visualizing the classification output, GradCAM and the misclassified images.
3. Host the app using HuggingFace Spaces.

**Link to HuggingFace Space:** `https://huggingface.co/spaces/madhurjindal/image_classification_cifar10_gradcam`

## Structure

<img width="464" alt="image" src="https://github.com/Madhur-1/ERA-v1/assets/64495917/fd82959c-e02a-46ed-bc03-bd89ea858f96">



### Metrics
| Train Acc | Test Acc | Train Loss | Test Loss |
|-----------|----------|------------|-----------|
| 96.47     | 92.50    | 0.10       | 0.23      |


## Performance Curve
![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/dc30114c-a912-4d1c-9dd1-7568f008a2a9)



## Confusion Matrix

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/99f9bb9d-d907-41f5-b134-a214750b1c4b)



## Data Exploration

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/9bc426b3-c3cb-4307-8390-d725b434a22f)



```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train data transformations
train_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=48, min_width=48, always_apply=True, border_mode=0),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(p=0.5),
        # A.PadIfNeeded(min_height=64, min_width=64, always_apply=True, border_mode=0),
        A.CoarseDropout(
            p=0.2,
            max_holes=1,
            max_height=8,
            max_width=8,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=(0.4914, 0.4822, 0.4465),
            mask_fill_value=None,
        ),
        # A.CenterCrop(height=32, width=32, always_apply=True),
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

As seen above, three transforms from the Albumentations library RandomCrop, HoriznotalFlip and CourseDropout were used.

## LR Finder

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/9d5d49d6-83d6-4add-84d2-4b20cb9fdc3e)


`LR suggestion: steepest gradient
Suggested LR: 2.56E-03`

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

Total Incorrect Preds = 750

![image](https://github.com/Madhur-1/ERA-v1/assets/64495917/69211262-15a7-4d2a-806c-a9c7b3c76264)




We see that the misclassified images in all three models have classes very close to each other as misclassified. These misclassified images would be hard for a human to classify correctly too!

## Training Log

```
Epoch: 0, Val Loss: 1.477613925933838, Val Accuracy: 0.46369999647140503

Epoch: 0, Train Loss: 1.8622440099716187, Train Accuracy: 0.32736000418663025
Validation: 0it [00:00, ?it/s]
Epoch: 1, Val Loss: 1.0305286645889282, Val Accuracy: 0.6324999928474426

Epoch: 1, Train Loss: 1.2540448904037476, Train Accuracy: 0.5509399771690369
Validation: 0it [00:00, ?it/s]
Epoch: 2, Val Loss: 0.905449390411377, Val Accuracy: 0.6851999759674072

Epoch: 2, Train Loss: 0.9339127540588379, Train Accuracy: 0.6700199842453003
Validation: 0it [00:00, ?it/s]
Epoch: 3, Val Loss: 0.799048900604248, Val Accuracy: 0.7333999872207642

Epoch: 3, Train Loss: 0.7537699341773987, Train Accuracy: 0.7354999780654907
Validation: 0it [00:00, ?it/s]
Epoch: 4, Val Loss: 0.6794794201850891, Val Accuracy: 0.7656999826431274

Epoch: 4, Train Loss: 0.6541508436203003, Train Accuracy: 0.7729399800300598
Validation: 0it [00:00, ?it/s]
Epoch: 5, Val Loss: 0.6306113004684448, Val Accuracy: 0.7867000102996826

Epoch: 5, Train Loss: 0.5723051428794861, Train Accuracy: 0.8019199967384338
Validation: 0it [00:00, ?it/s]
Epoch: 6, Val Loss: 0.5900896191596985, Val Accuracy: 0.7961000204086304

Epoch: 6, Train Loss: 0.5007141828536987, Train Accuracy: 0.8266400098800659
Validation: 0it [00:00, ?it/s]
Epoch: 7, Val Loss: 0.471587210893631, Val Accuracy: 0.8413000106811523

Epoch: 7, Train Loss: 0.4469413161277771, Train Accuracy: 0.8455600142478943
Validation: 0it [00:00, ?it/s]
Epoch: 8, Val Loss: 0.5326340198516846, Val Accuracy: 0.8228999972343445

Epoch: 8, Train Loss: 0.39900901913642883, Train Accuracy: 0.8606799840927124
Validation: 0it [00:00, ?it/s]
Epoch: 9, Val Loss: 0.40668997168540955, Val Accuracy: 0.8654000163078308

Epoch: 9, Train Loss: 0.3671680986881256, Train Accuracy: 0.8739399909973145
Validation: 0it [00:00, ?it/s]
Epoch: 10, Val Loss: 0.3456963002681732, Val Accuracy: 0.8837000131607056

Epoch: 10, Train Loss: 0.3388667702674866, Train Accuracy: 0.8831999897956848
Validation: 0it [00:00, ?it/s]
Epoch: 11, Val Loss: 0.35601386427879333, Val Accuracy: 0.8769999742507935

Epoch: 11, Train Loss: 0.30515751242637634, Train Accuracy: 0.89274001121521
Validation: 0it [00:00, ?it/s]
Epoch: 12, Val Loss: 0.31688356399536133, Val Accuracy: 0.8914999961853027

Epoch: 12, Train Loss: 0.28730571269989014, Train Accuracy: 0.8989400267601013
Validation: 0it [00:00, ?it/s]
Epoch: 13, Val Loss: 0.3391858637332916, Val Accuracy: 0.8871999979019165

Epoch: 13, Train Loss: 0.26656946539878845, Train Accuracy: 0.9071800112724304
Validation: 0it [00:00, ?it/s]
Epoch: 14, Val Loss: 0.3330417573451996, Val Accuracy: 0.8906999826431274

Epoch: 14, Train Loss: 0.25152936577796936, Train Accuracy: 0.9127799868583679
Validation: 0it [00:00, ?it/s]
Epoch: 15, Val Loss: 0.2939375936985016, Val Accuracy: 0.9016000032424927

Epoch: 15, Train Loss: 0.22532588243484497, Train Accuracy: 0.9213200211524963
Validation: 0it [00:00, ?it/s]
Epoch: 16, Val Loss: 0.29781222343444824, Val Accuracy: 0.9031000137329102

Epoch: 16, Train Loss: 0.2067444771528244, Train Accuracy: 0.9279400110244751
Validation: 0it [00:00, ?it/s]
Epoch: 17, Val Loss: 0.3049575090408325, Val Accuracy: 0.9053999781608582

Epoch: 17, Train Loss: 0.19144289195537567, Train Accuracy: 0.9330800175666809
Validation: 0it [00:00, ?it/s]
Epoch: 18, Val Loss: 0.2997421622276306, Val Accuracy: 0.9049000144004822

Epoch: 18, Train Loss: 0.17048843204975128, Train Accuracy: 0.9408599734306335
Validation: 0it [00:00, ?it/s]
Epoch: 19, Val Loss: 0.2569156587123871, Val Accuracy: 0.9153000116348267

Epoch: 19, Train Loss: 0.1569124311208725, Train Accuracy: 0.9463599920272827
Validation: 0it [00:00, ?it/s]
Epoch: 20, Val Loss: 0.25585222244262695, Val Accuracy: 0.9172000288963318

Epoch: 20, Train Loss: 0.13740618526935577, Train Accuracy: 0.9524999856948853
Validation: 0it [00:00, ?it/s]
Epoch: 21, Val Loss: 0.2459949254989624, Val Accuracy: 0.9215999841690063

Epoch: 21, Train Loss: 0.12286654114723206, Train Accuracy: 0.9581800103187561
Validation: 0it [00:00, ?it/s]
Epoch: 22, Val Loss: 0.24200274050235748, Val Accuracy: 0.923799991607666

Epoch: 22, Train Loss: 0.11110514402389526, Train Accuracy: 0.9627400040626526
Validation: 0it [00:00, ?it/s]
Epoch: 23, Val Loss: 0.23540237545967102, Val Accuracy: 0.925000011920929

Epoch: 23, Train Loss: 0.10367754846811295, Train Accuracy: 0.964739978313446
```
