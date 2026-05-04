from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    report: Dict[str, Dict[str, float]]
    report_path: Path
    plot_loss_path: Path
    confusion_matrix_path: Path


class Evaluator:
    @staticmethod
    def evaluate(
        model,
        X_test,
        Y_test,
        class_names: List[str],
        output_dir: Path,
        history=None,
    ) -> EvaluationResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        y_pred_raw = model.predict(X_test, verbose=0)
        if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] == 1:
            y_pred = np.round(y_pred_raw).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_raw, axis=1)

        y_true = np.asarray(Y_test).reshape(-1)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            output_dict=True,
        )

        report_path = output_dir / "relatorio_metricas.txt"
        Evaluator._save_text_report(report, accuracy, class_names, report_path)

        confusion_matrix_path = output_dir / "matriz_confusao.png"
        Evaluator._save_confusion_matrix(y_true, y_pred, class_names, confusion_matrix_path)

        plot_loss_path = output_dir / "avaliacao_loss_accuracy.png"
        Evaluator._save_loss_accuracy_plot(history, plot_loss_path)

        return EvaluationResult(
            accuracy=accuracy,
            report=report,
            report_path=report_path,
            plot_loss_path=plot_loss_path,
            confusion_matrix_path=confusion_matrix_path,
        )

    @staticmethod
    def _save_text_report(report: Dict[str, Dict[str, float]], accuracy: float, class_names: List[str], path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("RELATORIO DE METRICAS\n")
            handle.write("=" * 60 + "\n\n")
            handle.write(f"Acuracia global: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
            handle.write(f"{'Classe':<22} {'Precisao':>10} {'Recall':>10} {'F1':>10} {'Suporte':>10}\n")
            handle.write("-" * 62 + "\n")
            for class_name in class_names:
                row = report[class_name]
                handle.write(
                    f"{class_name:<22} {row['precision']:>10.4f} {row['recall']:>10.4f}"
                    f" {row['f1-score']:>10.4f} {int(row['support']):>10}\n"
                )
            handle.write("\n")
            for summary in ("macro avg", "weighted avg"):
                row = report[summary]
                handle.write(
                    f"{summary:<22} {row['precision']:>10.4f} {row['recall']:>10.4f}"
                    f" {row['f1-score']:>10.4f}\n"
                )

    @staticmethod
    def _save_confusion_matrix(y_true, y_pred, class_names: List[str], path: Path) -> None:
        cm = confusion_matrix(y_true, y_pred)
        figure = plt.figure(figsize=(max(6, len(class_names) * 1.5), max(5, len(class_names) * 1.5)))
        sns.heatmap(
            pd.DataFrame(cm, index=class_names, columns=class_names),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 12},
        )
        plt.ylabel("Real")
        plt.xlabel("Predito")
        plt.title("Matriz de Confusao")
        plt.tight_layout()
        figure.savefig(str(path), dpi=150)
        plt.close(figure)

    @staticmethod
    def _save_loss_accuracy_plot(history, path: Path) -> None:
        figure, axes = plt.subplots(1, 2, figsize=(16, 8))
        if history is not None and hasattr(history, "history"):
            data = history.history
            axes[0].plot(data.get("loss", []), color="b", label="Training loss")
            axes[0].plot(data.get("val_loss", []), color="r", label="Validation loss")
            axes[0].set_title("Loss")
            axes[0].legend()

            axes[1].plot(data.get("accuracy", []), color="b", label="Training accuracy")
            axes[1].plot(data.get("val_accuracy", []), color="r", label="Validation accuracy")
            axes[1].set_title("Accuracy")
            axes[1].legend()
        plt.tight_layout()
        figure.savefig(str(path), dpi=150)
        plt.close(figure)
