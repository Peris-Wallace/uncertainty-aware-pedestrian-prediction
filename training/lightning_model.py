# Import required libraries
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch import Tensor, nn, optim
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from torchmetrics.functional.classification import binary_f1_score, binary_recall, binary_precision

# Import custom modules
from training.loss import edl_digamma_loss, edl_log_loss, edl_mse_loss
from utils import one_hot_embedding, relu_evidence
from training.training_utils import update_uncertainty_statistics, log_uncertainty_statistics
from training.metrics import (
    binary_brier_score,
    expected_calibration_error,
    risk_coverage_curve,
    area_under_risk_coverage_curve,
)      

from evaluation.plots import (
    plot_validation_model_selection,
    plot_metric_history,
    plot_confusion_matrix,
    plot_reliability_diagram,
    plot_precision_recall_curve,
    plot_risk_coverage,
    plot_uncertainty_distribution,
    log_figure,
)


class PedestrianCrossingLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        uncertainty=False,
        evidential_loss="mse",
        learning_rate=1e-3,
        weight_decay=0.0,
        annealing_step=10,
        scheduler_t_max=64,
        scheduler_eta_min=0.0,
        tune_threshold=False,
        decision_threshold=0.5,
        threshold_min=0.01,
        threshold_max=0.99,
        threshold_steps=99
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.uncertainty = uncertainty
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.annealing_step = annealing_step

        self.scheduler_t_max = scheduler_t_max
        self.scheduler_eta_min = scheduler_eta_min
    
        self.num_classes = 2

        self.evidential_loss_name = evidential_loss

        if uncertainty:
            loss_functions = {
                "mse": edl_mse_loss,
                "log": edl_log_loss,
                "digamma": edl_digamma_loss,
            }

            if evidential_loss not in loss_functions:
                raise ValueError("loss function must be one of: 'mse', 'log', or 'digamma'.")

            self.loss_function = loss_functions[evidential_loss]
        else:
            self.loss_function = nn.CrossEntropyLoss()


        self.train_accuracy = Accuracy(task="binary")

        self.validation_class_probabilities = []
        self.validation_targets = []
        self.validation_uncertainties = []

        self.validation_history = {
            "epoch": [],
            "f1": [],
            "threshold": [],
            "ece": [],
            "brier": [],
            "uncertainty": [],
}
        self.test_class_probabilities = []
        self.test_targets = []
        self.test_uncertainties = []
        self.test_predictions = []

        self.tune_threshold = tune_threshold
        self.fixed_threshold = decision_threshold
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_steps = threshold_steps

        self.best_threshold = 0.5
        self.best_validation_f1 = float("-inf")
        self.best_validation_epoch = -1 

        self.val_accuracy = Accuracy(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")

        self.register_buffer("decision_threshold", torch.tensor(decision_threshold, dtype=torch.float32))
        self.register_buffer("val_uncertainty_sum", torch.tensor(0.0))
        self.register_buffer("val_uncertainty_count", torch.tensor(0, dtype=torch.long))

        self.test_accuracy = Accuracy(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_confusion_matrix = ConfusionMatrix(task="binary")

        self.register_buffer("test_uncertainty_sum", torch.tensor(0.0))
        self.register_buffer("test_uncertainty_count", torch.tensor(0, dtype=torch.long))



    @classmethod
    def from_config(cls, model, config):
        train_opts = config["train_opts"]
        uncertainty_opts = config.get("uncertainty_opts", {})
        scheduler_opts = config.get("scheduler", {})
        evaluation_opts = config.get("evaluation", {})

        return cls(
            model=model, 
            uncertainty=uncertainty_opts.get("enabled", False),
            evidential_loss=uncertainty_opts.get("loss", "mse"),
            annealing_step=uncertainty_opts.get("annealing_step", 10),
            learning_rate=train_opts.get("learning_rate", 0.005),
            weight_decay=train_opts.get("weight_decay", 0.0),
            scheduler_t_max=scheduler_opts.get("t_max", train_opts["epochs"]),
            scheduler_eta_min=scheduler_opts.get("eta_min", 0.0),
            tune_threshold=evaluation_opts.get("tune_threshold", False),
            decision_threshold=evaluation_opts.get("decision_threshold", 0.5),
            threshold_min=evaluation_opts.get("threshold_min", 0.01),
            threshold_max=evaluation_opts.get("threshold_max", 0.99),
            threshold_steps=evaluation_opts.get("threshold_steps", 99),
        )


    def forward(self, x):
        return self.model(x)

    def _prepare_batch(self, batch):
        x, y = batch

        model_dtype = next(self.model.parameters()).dtype

        x = x.to(dtype=model_dtype)
        y = y.reshape(-1).long()

        return x, y

    def _shared_step(self,batch):
        x, y = self._prepare_batch(batch)

        outputs = self.model(x)

        if self.uncertainty:
            # Convert binary class labels to one-hot vectors because the
            # evidential loss operates on class-wise targets.            
            y_one_hot = one_hot_embedding(
                y, num_classes=self.num_classes
                ).to(
                device=outputs.device,
                dtype=outputs.dtype,
            )

            # Compute the selected evidential loss (MSE, log, or digamma).
            # The annealing term gradually introduces the KL regularisation during training.
            loss = self.loss_function(
                outputs,
                y_one_hot,
                self.current_epoch,
                self.num_classes,
                self.annealing_step,
                self.device,
            )

            # Convert the network outputs into non-negative evidence.
            # Evidence represents the support the model has accumulated for each class.
            evidence = relu_evidence(outputs)

            # Dirichlet concentration parameters.
            # Adding 1 ensures alpha_k >= 1 even when no evidence exists for a particular class.
            alpha = evidence + 1.0

            total_evidence = alpha.sum(dim=1, keepdim=True)
            probabilities = alpha / total_evidence
    
            uncertainty = self.num_classes / total_evidence.squeeze(1)

        else:
            # Cross-entropy loss and softmax probabilities.
            loss = self.loss_function(outputs, y)
            probabilities = torch.softmax(outputs, dim=1)

            uncertainty = None

        positive_probability = probabilities[:, 1]

        return loss, probabilities, positive_probability, y, uncertainty
        

    def training_step(self, batch, batch_idx):
        loss, probabilities, positive_probability, targets, uncertainty = self._shared_step(batch)

        self.train_accuracy.update(positive_probability, targets)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=targets.size(0),
        )

        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=targets.size(0),
        )

        if uncertainty is not None:
            self.log(
                "train_mean_uncertainty",
                uncertainty.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=targets.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, probabilities, positive_probability, targets, uncertainty = self._shared_step(batch)

        self.val_accuracy.update(positive_probability, targets)
        self.val_precision.update(positive_probability, targets)
        self.val_recall.update(positive_probability, targets)
        self.val_f1.update(positive_probability, targets)
        self.val_auroc.update(positive_probability, targets)
        
        self.validation_class_probabilities.append(probabilities.detach().cpu())
        self.validation_targets.append(targets.detach().cpu())
        
        if uncertainty is not None:
            self.validation_uncertainties.append(uncertainty.detach().cpu())

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=targets.size(0),
        )
        
        if uncertainty is not None:
            update_uncertainty_statistics(
                module=self,
                uncertainty=uncertainty,
                stage="val",
            )

        return loss


    def on_validation_epoch_end(self):

        if self.trainer.sanity_checking:
            self.validation_class_probabilities.clear()
            self.validation_targets.clear()
            self.validation_uncertainties.clear()
            return

        if not self.validation_class_probabilities:
            return

        val_accuracy = self.val_accuracy.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_f1 = self.val_f1.compute()
        val_auroc = self.val_auroc.compute()

        validation_probabilities = torch.cat(self.validation_class_probabilities, dim=0)
        positive_probabilities = validation_probabilities[:, 1]
        validation_targets = torch.cat(self.validation_targets, dim=0)

        validation_uncertainties = None
        if self.uncertainty and self.validation_uncertainties:
            validation_uncertainties = torch.cat(self.validation_uncertainties, dim=0)
            
        val_ece = expected_calibration_error(
            class_probabilities=validation_probabilities,
            targets=validation_targets,
            n_bins=15,
        )

        val_brier = binary_brier_score(
            probabilities=positive_probabilities,
            targets=validation_targets,
        )

        # If threshold tuning is enabled thresholds are evaluated on the validation set. 
        # The threshold producing the highest validation F1 score is selected.
        if self.tune_threshold:
            thresholds = torch.linspace(
                self.threshold_min,
                self.threshold_max,
                self.threshold_steps,
                device=positive_probabilities.device,
            )
    
            best_threshold = float(self.fixed_threshold)
            best_val_f1 = 0.0

            for threshold in thresholds:
                threshold_predictions = (positive_probabilities >= threshold).long()
                threshold_f1 = binary_f1_score(threshold_predictions, validation_targets)

                if threshold_f1 > best_val_f1:
                    best_val_f1 = threshold_f1 
                    best_threshold = float(threshold.detach().cpu().item())
        
        else:
            # Use the configured fixed threshold if tuning is disabled
            best_threshold = self.fixed_threshold
    

        # Compute ALL validation metrics at selected threshold
        best_threshold_predictions = (positive_probabilities >= best_threshold).long()
        best_val_f1 = binary_f1_score(best_threshold_predictions, validation_targets)
        best_precision = binary_precision(best_threshold_predictions, validation_targets)
        best_recall = binary_recall(best_threshold_predictions, validation_targets)   

        
        current_best_f1 = float(best_val_f1.detach().cpu().item())

        if current_best_f1 > self.best_validation_f1:
            self.best_validation_f1 = current_best_f1

            # Track the best validation epoch across training
            self.best_validation_epoch = int(self.current_epoch + 1)

            print(
                "BEST_EPOCH "
                f"best_epoch={self.best_validation_epoch} "
                f"best_val_f1={self.best_validation_f1:.6f} "
                f"best_threshold={best_threshold:.4f}"
            )

        # Store this epoch's selected threshold in the model.
        self.best_threshold = best_threshold
        self.decision_threshold.fill_(best_threshold)

        self.log(
            "val_accuracy",
            val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "val_f1",
            val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_precision",
            val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val_recall",
            val_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val_auroc",
            val_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "val_best_threshold_f1",
            best_val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val_best_threshold_precision",
            best_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "val_best_threshold_recall",
            best_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "val_best_threshold",
            best_threshold,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        
        self.log(
            "val_ece",
            val_ece,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "val_brier",
            val_brier,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        uncertainty_stats = None

        if self.uncertainty:
            uncertainty_stats = log_uncertainty_statistics(module=self, stage="val")


        # Store epoch-level values for plots.
        self.validation_history["epoch"].append(int(self.current_epoch + 1))
        self.validation_history["f1"].append(float(best_val_f1.detach().cpu().item()))
        self.validation_history["ece"].append(float(val_ece.detach().cpu().item()))
        self.validation_history["brier"].append(float(val_brier.detach().cpu().item()))
        self.validation_history["threshold"].append(float(best_threshold))

        if uncertainty_stats is not None:
            self.validation_history["uncertainty"].append(
                float(uncertainty_stats["mean"].detach().cpu().item()))
        else:
            self.validation_history["uncertainty"].append(float("nan"))


        output = (
            f"\nEpoch {self.current_epoch}: "
            f"val_f1@0.5={val_f1.item():.4f}, "
            f"best_val_f1={best_val_f1.item():.4f}, "
            f"val_precision@0.5={val_precision.item():.4f}, "
            f"best_val_precision={best_precision.item():.4f}, "
            f"val_recall@0.5={val_recall.item():.4f}, "
            f"best_val_recall={best_recall.item():.4f}, "
            f"best_threshold={best_threshold:.2f}, "
            f"val_ece={val_ece.item():.4f}, "
            f"val_brier={val_brier.item():.4f}"
        )

        if uncertainty_stats is not None:
            output += (
                f", val_uncertainty="
                f"{uncertainty_stats['mean'].item():.4f}"
            )

        print(output)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()


        self.validation_class_probabilities.clear()
        self.validation_targets.clear()
        self.validation_uncertainties.clear()

    def on_fit_end(self):

        epochs = self.validation_history["epoch"]

        if not epochs:
            return

        # Validation F1
        fig, _ = plot_validation_model_selection(
            epochs=self.validation_history["epoch"],
            validation_f1=self.validation_history["f1"],
            selected_thresholds=self.validation_history["threshold"],
            best_epoch=self.best_validation_epoch,
        )

        log_figure(
            module=self,
            fig=fig,
            filename="validation_model_selection.png",
            artifact_path="plots/validation",
        )

        # Validation ECE
        fig, _ = plot_metric_history(
            epochs=epochs,
            values=self.validation_history["ece"],
            ylabel="Expected Calibration Error (ECE)",
            title="Validation Calibration Across Epochs",
            best_epoch=self.best_validation_epoch,
            label="ECE",
        )

        log_figure(
            module=self,
            fig=fig,
            filename="validation_ece.png",
            artifact_path="plots/validation",
        )

        # Validation Brier score
        fig, _ = plot_metric_history(
            epochs=epochs,
            values=self.validation_history["brier"],
            ylabel="Brier Score",
            title="Validation Brier Score Across Epochs",
            best_epoch=self.best_validation_epoch,
            label="Brier Score",
        )

        log_figure(
            module=self,
            fig=fig,
            filename="validation_brier.png",
            artifact_path="plots/validation",
        )

        # EDL uncertainty
        if self.uncertainty:
            fig, _ = plot_metric_history(
                epochs=epochs,
                values=self.validation_history["uncertainty"],
                ylabel="Mean Evidential Uncertainty",
                title="Validation Uncertainty Across Epochs",
                best_epoch=self.best_validation_epoch,
                label="Mean Uncertainty",
            )

            log_figure(
                module=self,
                fig=fig,
                filename="validation_uncertainty.png",
                artifact_path="plots/validation",
            )


    def test_step(self, batch, batch_idx):
        loss, probabilities, positive_probability, targets, uncertainty = self._shared_step(batch)

        # Convert positive-class probabilities into binary predictions
        # using the decision threshold selected from validation data.
        test_predictions = (positive_probability >= self.decision_threshold).long()

        self.test_class_probabilities.append(probabilities.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        self.test_predictions.append(test_predictions.detach().cpu())

        # Store samples required for risk-coverage analysis.
        if uncertainty is not None:
            self.test_uncertainties.append(uncertainty.detach().cpu())
        
        # Classification metrics
        self.test_accuracy.update(test_predictions, targets)
        self.test_precision.update(test_predictions, targets)
        self.test_recall.update(test_predictions, targets)
        self.test_f1.update(test_predictions, targets)

        # Evaluates discrimination across all possible classification thresholds.
        self.test_auroc.update(positive_probability, targets)

        # TP, TN, FP and FN counts for error analysis.
        self.test_confusion_matrix.update(test_predictions, targets)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=targets.size(0),
        )

        if uncertainty is not None:
            update_uncertainty_statistics(
                module=self,
                uncertainty=uncertainty,
                stage="test",
            )

        return loss
    
    def on_test_start(self):
        print(
            "\nTesting with decision threshold: "
            f"{self.decision_threshold.item():.2f}"
        )
        
    def on_test_epoch_end(self):

        test_accuracy = self.test_accuracy.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        test_f1 = self.test_f1.compute()
        test_auroc = self.test_auroc.compute()
        confusion_matrix = self.test_confusion_matrix.compute()

        tn, fp = confusion_matrix[0]
        fn, tp = confusion_matrix[1]

        specificity = tn.float() / (tn + fp).float().clamp_min(1)
        false_positive_rate = fp.float() / (fp + tn).float().clamp_min(1)
        false_negative_rate = fn.float() / (fn + tp).float().clamp_min(1)

        normalized_cm = (
            confusion_matrix.float()
            / confusion_matrix.float().sum(dim=1, keepdim=True).clamp_min(1)
        )

        # Stored test outputs
        test_probabilities = torch.cat(self.test_class_probabilities, dim=0)
        test_targets = torch.cat(self.test_targets, dim=0)
        test_predictions = torch.cat(self.test_predictions, dim=0)

        positive_test_probabilities = test_probabilities[:, 1]

        test_ece = expected_calibration_error(
            class_probabilities=test_probabilities,
            targets=test_targets,
            n_bins=15,
        )

        test_brier = binary_brier_score(
            probabilities=positive_test_probabilities,
            targets=test_targets,
        )

        # Uncertainty metrics
        test_aurc = None
        test_coverage = None
        test_risk = None
        test_uncertainties = None

        if self.uncertainty:
            test_uncertainties = torch.cat(
                self.test_uncertainties,
                dim=0,
            )

            test_coverage, test_risk = risk_coverage_curve(
                uncertainty=test_uncertainties,
                predictions=test_predictions,
                targets=test_targets,
            )

            test_aurc = area_under_risk_coverage_curve(
                coverage=test_coverage,
                risk=test_risk,
            )

        # Log metrics
        metrics = {
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_auroc": test_auroc,
            "test_ece": test_ece,
            "test_brier": test_brier,
            "test_tn": tn.float(),
            "test_fp": fp.float(),
            "test_fn": fn.float(),
            "test_tp": tp.float(),
            "test_specificity": specificity,
            "test_false_positive_rate": false_positive_rate,
            "test_false_negative_rate": false_negative_rate,
        }

        if test_aurc is not None:
            metrics["test_aurc"] = test_aurc

        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        # Test plots
        fig, _ = plot_confusion_matrix(
            confusion_matrix=confusion_matrix.detach().cpu().numpy(),
            normalize=True,
            title="Confusion Matrix",
        )

        log_figure(
            module=self,
            fig=fig,
            filename="test_confusion_matrix.png",
            artifact_path="plots/test",
        )

        fig, _ = plot_reliability_diagram(
            probabilities=test_probabilities.numpy(),
            targets=test_targets.numpy(),
            n_bins=15,
            ece=test_ece.item(),
            title="Reliability Diagram",
        )

        log_figure(
            module=self,
            fig=fig,
            filename="test_reliability_diagram.png",
            artifact_path="plots/test",
        )

        fig, _ = plot_precision_recall_curve(
            probabilities=positive_test_probabilities.numpy(),
            targets=test_targets.numpy(),
            selected_threshold=self.decision_threshold.item(),
            selected_precision=test_precision.item(),
            selected_recall=test_recall.item(),
            title="Precision–Recall Curve",
        )

        log_figure(
            module=self,
            fig=fig,
            filename="test_precision_recall_curve.png",
            artifact_path="plots/test",
        )

        if self.uncertainty:
            fig, _ = plot_risk_coverage(
                coverage=test_coverage.numpy(),
                risk=test_risk.numpy(),
                aurc=test_aurc.item(),
                title="Risk–Coverage Curve",
            )

            log_figure(
                module=self,
                fig=fig,
                filename="test_risk_coverage.png",
                artifact_path="plots/test",
            )

            fig, _ = plot_uncertainty_distribution(
                uncertainties=test_uncertainties.numpy(),
                title="Distribution of Uncertainty",
            )

            log_figure(
                module=self,
                fig=fig,
                filename="test_uncertainty_distribution.png",
                artifact_path="plots/test",
            )

            log_uncertainty_statistics(
                module=self,
                stage="test",
            )

        # Summary
        print("\nTest Confusion Matrix")
        print("--------------------------------")
        print(f"TN={tn.item()}  FP={fp.item()}")
        print(f"FN={fn.item()}  TP={tp.item()}")

        print("\nNormalized Test Confusion Matrix")
        print(normalized_cm.cpu().numpy())

        print(f"\nSpecificity: {specificity.item():.4f}")
        print(f"False Positive Rate: {false_positive_rate.item():.4f}")
        print(f"False Negative Rate: {false_negative_rate.item():.4f}")

        # Reset
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_confusion_matrix.reset()

        self.test_class_probabilities.clear()
        self.test_targets.clear()
        self.test_predictions.clear()
        self.test_uncertainties.clear()


    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = (
            optim.lr_scheduler
            .CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_t_max,
                eta_min=self.scheduler_eta_min,
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            },
        }