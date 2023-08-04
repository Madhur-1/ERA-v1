import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .visualize import plot_model_training_curves


class TrainingEndCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        # Perform actions at the end of the entire training process
        print("Training, validation, and testing completed!")

        logged_metrics = pl_module.log_store
        print(logged_metrics)
        plot_model_training_curves(
            train_accs=logged_metrics["train_acc_epoch"],
            test_accs=logged_metrics["val_acc_epoch"],
            train_losses=logged_metrics["train_loss_epoch"],
            test_losses=logged_metrics["val_loss_epoch"],
        )


class PrintLearningMetricsCallback(Callback):
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        print(
            f"\nEpoch: {trainer.current_epoch}, Train Loss: {trainer.logged_metrics['train_loss_epoch']}, Train Accuracy: {trainer.logged_metrics['train_acc_epoch']}"
        )
        pl_module.log_store.get("train_loss_epoch").append(
            trainer.logged_metrics["train_loss_epoch"].cpu().detach().item()
        )
        pl_module.log_store.get("train_acc_epoch").append(
            trainer.logged_metrics["train_acc_epoch"].cpu().detach().item()
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        print(
            f"\nEpoch: {trainer.current_epoch}, Val Loss: {trainer.logged_metrics['val_loss_epoch']}, Val Accuracy: {trainer.logged_metrics['val_acc_epoch']}"
        )
        pl_module.log_store.get("val_loss_epoch").append(
            trainer.logged_metrics["val_loss_epoch"].cpu().detach().item()
        )
        pl_module.log_store.get("val_acc_epoch").append(
            trainer.logged_metrics["val_acc_epoch"].cpu().detach().item()
        )
        print(pl_module.log_store)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        print(
            f"\nEpoch: {trainer.current_epoch}, Test Loss: {trainer.logged_metrics['test_loss_epoch']}, Test Accuracy: {trainer.logged_metrics['test_acc_epoch']}"
        )
        pl_module.log_store.get("test_loss_epoch").append(
            trainer.logged_metrics["test_loss_epoch"].cpu().detach().item()
        )
        pl_module.log_store.get("test_acc_epoch").append(
            trainer.logged_metrics["test_acc_epoch"].cpu().detach().item()
        )
