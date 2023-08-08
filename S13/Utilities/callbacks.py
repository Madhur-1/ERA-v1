import pytorch_lightning as pl

from . import config
from .utils import (
    check_class_accuracy,
    get_evaluation_bboxes,
    mean_average_precision,
    plot_couple_examples,
)


class PlotTestExamplesCallback(pl.Callback):
    def __init__(self, every_n_epochs: int = 1) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            plot_couple_examples(
                model=pl_module,
                loader=trainer.datamodule.train_dataloader(),
                thresh=0.6,
                iou_thresh=0.5,
                anchors=pl_module.scaled_anchors,
            )


class CheckClassAccuracyCallback(pl.Callback):
    def __init__(
        self, train_every_n_epochs: int = 1, test_every_n_epochs: int = 3
    ) -> None:
        super().__init__()
        self.train_every_n_epochs = train_every_n_epochs
        self.test_every_n_epochs = test_every_n_epochs

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (trainer.current_epoch + 1) % self.train_every_n_epochs == 0:
            print("+++ TRAIN ACCURACIES")
            class_acc, no_obj_acc, obj_acc = check_class_accuracy(
                model=pl_module,
                loader=trainer.datamodule.train_dataloader(),
                threshold=config.CONF_THRESHOLD,
            )
            pl_module.log_dict(
                {
                    "train_class_acc": class_acc,
                    "train_no_obj_acc": no_obj_acc,
                    "train_obj_acc": obj_acc,
                },
                logger=True,
            )

        if (trainer.current_epoch + 1) % self.test_every_n_epochs == 0:
            print("+++ TEST ACCURACIES")
            class_acc, no_obj_acc, obj_acc = check_class_accuracy(
                model=pl_module,
                loader=trainer.datamodule.test_dataloader(),
                threshold=config.CONF_THRESHOLD,
            )
            pl_module.log_dict(
                {
                    "test_class_acc": class_acc,
                    "test_no_obj_acc": no_obj_acc,
                    "test_obj_acc": obj_acc,
                },
                logger=True,
            )


class MAPCallback(pl.Callback):
    def __init__(self, every_n_epochs: int = 3) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            pred_boxes, true_boxes = get_evaluation_bboxes(
                loader=trainer.datamodule.test_dataloader(),
                model=pl_module,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
                device=config.DEVICE,
            )

            map_val = mean_average_precision(
                pred_boxes=pred_boxes,
                true_boxes=true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print("+++ MAP: ", map_val.item())
            pl_module.log(
                "MAP",
                map_val.item(),
                logger=True,
            )
            pl_module.train()
