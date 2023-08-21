import gradio as gr
import torch

from Utilities import config
from Utilities.gradio_utils import generate_gradcam_output, plot_bboxes
from Utilities.model import YOLOv3
from Utilities.transforms import resize_transforms

model = YOLOv3.load_from_checkpoint(
    config.MODEL_CHECKPOINT_PATH,
    map_location=torch.device("cpu"),
)
# model = YOLOv3.load_from_checkpoint(
#     "/Users/madhurjindal/WorkProjects/ERA-v1/S13/Gradio App/Store/epoch=39-step=16560.ckpt",
#     map_location=torch.device("cpu"),
# )

examples = [
    [config.EXAMPLE_IMG_PATH + "cat.jpeg", 1],
    [config.EXAMPLE_IMG_PATH + "horse.jpg", 1],
    [config.EXAMPLE_IMG_PATH + "000018.jpg", 2],
    [config.EXAMPLE_IMG_PATH + "bird.webp", 2],
    [config.EXAMPLE_IMG_PATH + "000022.jpg", 2],
    [config.EXAMPLE_IMG_PATH + "airplane.png", 0],
    [config.EXAMPLE_IMG_PATH + "shipp.jpg", 0],
    [config.EXAMPLE_IMG_PATH + "car.jpg", 1],
    [config.EXAMPLE_IMG_PATH + "000007.jpg", 1],
    [config.EXAMPLE_IMG_PATH + "000013.jpg", 2],
    [config.EXAMPLE_IMG_PATH + "000012.jpg", 2],
    [config.EXAMPLE_IMG_PATH + "000006.jpg", 1],
    [config.EXAMPLE_IMG_PATH + "000004.jpg", 1],
    [config.EXAMPLE_IMG_PATH + "000014.jpg", 0],
]

title = "Object Detection (YOLOv3) with GradCAM"
description = """Introducing the YOLOv3 Object Detection Explorer 🕵️‍♀️🔍
---
Are you curious about the world of computer vision and object detection? Look no further! Our interactive Gradio app powered by Hugging Face Spaces brings the excitement of object detection to your fingertips.

🎉 Key Features:
---
YOLOv3 at Your Fingertips: Our app is built around the YOLOv3 model, meticulously trained from scratch using the comprehensive Pascal VOC dataset comprising 20 diverse classes. This ensures accurate and robust object detection.

Precision with GradCAM: Experience the power of GradCAM (Gradient-weighted Class Activation Mapping), a cutting-edge technique that delves into the inner workings of the model. By harnessing gradients, it unveils the specific areas in an image that heavily influence the classification score. This level of insight is unprecedented and helps demystify the model's decision-making process.

Streamline Your Object Detection: With three different output streams providing sizes of 13x13, 26x26, and 52x52, you have the flexibility to focus on objects of varying sizes. Smaller outputs excel at capturing large objects, while larger ones excel at handling more intricate details. Tailor your approach based on the nature of your task.

📸 How It Works:
---
Simply upload an image that you'd like to subject to object detection.
Select the output stream that you believe is most appropriate for your task.
Sit back and watch as our YOLOv3 model deftly identifies and annotates objects within the image.
For an added layer of enlightenment, explore the GradCAM visualization. Uncover the regions that the model identifies as pivotal in its classification decision, all in real-time!

✅ Pascal VOC Classes:
---
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

🌟 Explore and Learn:
---
Our "Examples" tab is a treasure trove of visual insights. Explore pre-loaded images with varying complexities to witness the prowess of YOLOv3 in action. Study the GradCAM outputs to gain a deeper understanding of how different output streams affect the model's attention.

Ready to embark on an object detection journey like never before? Give our YOLOv3 Object Detection Explorer a try and discover the captivating world of computer vision today!
"""


def generate_gradio_output(
    input_img,
    gradcam_output_stream=0,
):
    input_img = resize_transforms(image=input_img)["image"]
    fig, processed_img = plot_bboxes(
        input_img=input_img,
        model=model,
        thresh=0.6,
        iou_thresh=0.5,
        anchors=model.scaled_anchors,
    )
    visualization = generate_gradcam_output(
        org_img=input_img,
        model=model,
        input_img=processed_img,
        gradcam_output_stream=gradcam_output_stream,
    )
    return fig, visualization


# generate_gradio_output(torch.zeros(416, 416, 3).numpy())

gr.Interface(
    fn=generate_gradio_output,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Slider(0, 2, step=1, label="GradCAM Output Stream (13, 26, 52)"),
    ],
    outputs=[
        gr.Plot(
            visible=True,
            label="Bounding Box Predictions",
        ),
        gr.Image(label="GradCAM Output").style(width=416, height=416),
    ],
    examples=examples,
    title=title,
    description=description,
).launch()
