from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def create_human_label_template(video_paths: list[str], out_csv: str) -> None:
    rows = []
    for v in video_paths:
        rows.append(
            {
                "video_path": v,
                "semantic_human": "",
                "logical_human": "",
                "decision_human": "",
                "unsafe_human": "",
                "note": "",
            }
        )

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compare_auto_vs_human(auto_csv: str, human_csv: str, out_dir: str) -> Dict[str, Dict[str, float]]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    auto_df = pd.read_csv(auto_csv)
    human_df = pd.read_csv(human_csv)

    merged = auto_df.merge(human_df, on="video_path", how="inner")

    for col in ["semantic_human", "logical_human", "decision_human", "unsafe_human"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    metrics: Dict[str, Dict[str, float]] = {}
    pairs = [
        ("semantic", "semantic_unsafe", "semantic_human"),
        ("logical", "logical_unsafe", "logical_human"),
        ("decision", "decision_unsafe", "decision_human"),
        ("overall", "unsafe", "unsafe_human"),
    ]

    for name, pred_col, gt_col in pairs:
        metrics[name] = _binary_metrics(merged[gt_col], merged[pred_col])

        report = classification_report(
            merged[gt_col],
            merged[pred_col],
            target_names=["safe", "unsafe"],
            zero_division=0,
            output_dict=False,
        )
        (out_path / f"classification_report_{name}.txt").write_text(str(report), encoding="utf-8")

    merged.to_csv(out_path / "merged_auto_human.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).T.to_csv(out_path / "metrics_summary.csv", encoding="utf-8-sig")

    return metrics


def analyze_failure_patterns(auto_csv: str, out_dir: str) -> Dict[str, float]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(auto_csv)

    total = len(df)
    if total == 0:
        stats = {
            "total": 0,
            "unsafe_ratio": 0.0,
            "semantic_unsafe_ratio": 0.0,
            "logical_unsafe_ratio": 0.0,
            "decision_unsafe_ratio": 0.0,
        }
        return stats

    for col in ["unsafe", "semantic_unsafe", "logical_unsafe", "decision_unsafe"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    def _safe_mean(col: str) -> float:
        if col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean())

    stats = {
        "total": float(total),
        "unsafe_ratio": float(df["unsafe"].mean()),
        "semantic_unsafe_ratio": float(df["semantic_unsafe"].mean()),
        "logical_unsafe_ratio": float(df["logical_unsafe"].mean()),
        "decision_unsafe_ratio": float(df["decision_unsafe"].mean()),
        "semantic_score_mean": _safe_mean("semantic_score"),
        "logical_score_mean": _safe_mean("logical_score"),
        "decision_score_mean": _safe_mean("decision_score"),
        "total_risk_mean": _safe_mean("total_risk"),
    }

    pd.DataFrame([stats]).to_csv(out_path / "pattern_summary.csv", index=False, encoding="utf-8-sig")

    # 统计三类问题组合出现频率
    combo = (
        df.groupby(["semantic_unsafe", "logical_unsafe", "decision_unsafe"])  # type: ignore[arg-type]
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    combo["ratio"] = combo["count"] / total
    combo.to_csv(out_path / "error_combinations.csv", index=False, encoding="utf-8-sig")

    return stats
