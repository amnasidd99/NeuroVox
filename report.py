"""
Generate a downloadable text/markdown research report for a single analysis.
"""

from __future__ import annotations

from datetime import datetime

from model import PRETTY_NAMES


def build_report(
    filename: str,
    probs: dict[str, float],
    predicted_class: str,
    threshold: float,
    is_real_model: bool,
    metrics: dict | None = None,
) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("# NeuroVox — Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Input file:** {filename}")
    lines.append(f"**Model:** ResNet-18 (transfer learning)")
    lines.append("")
    lines.append("## Prediction")
    lines.append("")
    lines.append(f"- **Predicted class:** {PRETTY_NAMES.get(predicted_class, predicted_class)}")
    lines.append(f"- **Decision threshold (tumor classes):** {threshold:.2f}")
    lines.append("")
    lines.append("## Class probabilities")
    lines.append("")
    lines.append("| Class | Probability |")
    lines.append("| --- | --- |")
    for cls, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {PRETTY_NAMES.get(cls, cls)} | {p * 100:.2f}% |")
    lines.append("")

    if metrics:
        lines.append("## Model performance (held-out test set)")
        lines.append("")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                lines.append(f"- **{k}:** {v:.3f}")
            else:
                lines.append(f"- **{k}:** {v}")
        lines.append("")

    if not is_real_model:
        lines.append("> NOTE: No trained weights were found, so this report was produced "
                     "with a randomly initialised placeholder model. Train the model "
                     "(see the Colab notebook) and add `weights/best_model.pth` for real predictions.")
        lines.append("")

    lines.append("## Disclaimer")
    lines.append("")
    lines.append(
        "This application is an educational research prototype demonstrating an "
        "end-to-end machine learning workflow and explainable AI for medical "
        "imaging. It is **not** intended for medical diagnosis or clinical "
        "decision-making."
    )
    return "\n".join(lines)
