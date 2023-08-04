import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .visualize import plot_model_training_curves


class TrainingEndCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        # Perform actions at the end of the entire training process
        print("Training, validation, and testing completed!")

        logged_metrics = trainer.logged_metrics
        print(logged_metrics)
        plot_model_training_curves(
            train_accs=logged_metrics["train_acc_epoch"],
            test_accs=logged_metrics["val_acc_epoch"],
            train_losses=logged_metrics["train_loss_epoch"],
            test_losses=logged_metrics["val_loss_epoch"],
        )
