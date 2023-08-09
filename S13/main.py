# del model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.tuner import Tuner
from torchsummary import summary
from Utilities import config
from Utilities.callbacks import (
    CheckClassAccuracyCallback,
    MAPCallback,
    PlotTestExamplesCallback,
)
from Utilities.model import YOLOv3

model = YOLOv3(num_classes=config.NUM_CLASSES)

summary(model, input_size=(3, config.IMAGE_SIZE, config.IMAGE_SIZE))

from Utilities.dataset import YOLODataModule

data_module = YOLODataModule(
    train_csv_path=config.DATASET + "/train.csv",
    test_csv_path=config.DATASET + "/test.csv",
)


trainer = pl.Trainer(
    max_epochs=40,
    accelerator=config.DEVICE,
    callbacks=[
        ModelCheckpoint(
            dirpath=config.CHECKPOINT_PATH,
            verbose=True,
        ),
        PlotTestExamplesCallback(every_n_epochs=3),
        CheckClassAccuracyCallback(train_every_n_epochs=1, test_every_n_epochs=3),
        MAPCallback(every_n_epochs=40),
        LearningRateMonitor(logging_interval="step", log_momentum=True),
    ],
    default_root_dir="Store/",
    precision=16,
)


# tuner = Tuner(trainer=trainer)

# # Run learning rate finder
# lr_finder = tuner.lr_find(
#     model,
#     datamodule=data_module,
#     min_lr=1e-4,
#     max_lr=1,
#     num_training=trainer.max_epochs,
# )

trainer.fit(model, data_module)
