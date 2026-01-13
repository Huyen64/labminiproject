from __future__ import annotations

"""
Script chạy trực tiếp pipeline:
- Chuẩn bị dữ liệu + baseline supervised
- Tạo dataset bán giám sát
- Chạy self-training với nhiều giá trị tau
- Chạy co-training với nhiều giá trị tau

Ví dụ chạy:
    python -m src.semi_experiments --step all
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from typing import Iterable, List

import numpy as np
import pandas as pd

from .classification_library import Paths, run_prepare, run_train
from .semi_supervised_library import (
    SemiDataConfig,
    SelfTrainingConfig,
    CoTrainingConfig,
    mask_labels_time_aware,
    run_self_training,
    run_co_training,
    add_alert_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SemiExperimentConfig:
    cutoff: str = "2017-01-01"
    label_missing_fraction: float = 0.95
    random_state: int = 42
    taus_self: List[float] | None = None
    taus_co: List[float] | None = None
    alert_from_class: str = "Unhealthy"

    def __post_init__(self) -> None:
        if self.taus_self is None:
            self.taus_self = [0.8, 0.9, 0.95]
        if self.taus_co is None:
            self.taus_co = [0.9]


def get_paths() -> Paths:
    return Paths(project_root=PROJECT_ROOT)


def ensure_processed_dir(paths: Paths) -> Path:
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    return paths.data_processed


def step_prepare_and_baseline(
    use_ucimlrepo: bool = False,
    raw_zip_path: str | None = "data/raw/PRSA2017_Data_20130301-20170228.zip",
    lag_hours: Iterable[int] = (1, 3, 24),
    cutoff: str = "2017-01-01",
) -> None:
    """
    - Chuẩn bị cleaned.parquet từ dữ liệu gốc
    - Train baseline supervised, sinh metrics.json + predictions_sample.csv
    """
    paths = get_paths()
    ensure_processed_dir(paths)

    print("==> [1] Chuẩn bị dữ liệu cleaned.parquet (run_prepare)")
    cleaned_path = run_prepare(
        paths=paths,
        use_ucimlrepo=use_ucimlrepo,
        raw_zip_path=raw_zip_path,
        lag_hours=tuple(lag_hours),
    )
    print(f"    -> {cleaned_path}")

    print("==> [2] Train baseline supervised (run_train)")
    out = run_train(paths=paths, cutoff=cutoff)
    metrics_path = paths.data_processed / "metrics.json"
    pred_path = paths.data_processed / "predictions_sample.csv"
    print(f"    -> metrics baseline: {metrics_path}")
    print(f"    -> predictions sample: {pred_path}")
    print(
        f"    Accuracy={out['metrics']['accuracy']:.4f}, "
        f"F1-macro={out['metrics']['f1_macro']:.4f}"
    )


def step_build_semi_dataset(cfg: SemiExperimentConfig) -> Path:
    """
    Tạo dataset bán giám sát:
    - Đọc cleaned.parquet
    - Ẩn bớt nhãn trên TRAIN (time-based) bằng mask_labels_time_aware
    - Ghi data/processed/dataset_for_semi.parquet
    """
    paths = get_paths()
    ensure_processed_dir(paths)
    cleaned_path = paths.data_processed / "cleaned.parquet"
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {cleaned_path}. Hãy chạy bước baseline trước (step=baseline hoặc step=all)."
        )

    print("==> [3] Tạo dataset bán giám sát (dataset_for_semi.parquet)")
    df = pd.read_parquet(cleaned_path)
    data_cfg = SemiDataConfig(cutoff=cfg.cutoff, random_state=cfg.random_state)
    df_semi = mask_labels_time_aware(
        df=df,
        cfg=data_cfg,
        missing_fraction=cfg.label_missing_fraction,
    )

    semi_path = paths.data_processed / "dataset_for_semi.parquet"
    df_semi.to_parquet(semi_path, index=False)
    print(
        f"    -> {semi_path} "
        f"(LABEL_MISSING_FRACTION={cfg.label_missing_fraction:.2f})"
    )
    return semi_path


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def step_self_training_sweep(cfg: SemiExperimentConfig) -> None:
    """
    Chạy Self-Training cho nhiều giá trị tau, lưu:
    - self_history_tau_*.csv
    - metrics_self_training_tau_*.json
    Đồng thời chọn tau tốt nhất (F1-macro test) và lưu:
    - metrics_self_training.json
    - predictions_self_training_sample.csv
    - alerts_self_training_sample.csv
    """
    paths = get_paths()
    semi_path = paths.data_processed / "dataset_for_semi.parquet"
    if not semi_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {semi_path}. Hãy chạy step=semi_dataset hoặc step=all trước."
        )

    print("==> [4] Self-Training sweep theo tau")
    df = pd.read_parquet(semi_path)
    data_cfg = SemiDataConfig(cutoff=cfg.cutoff, random_state=cfg.random_state)

    best_f1 = -np.inf
    best_result = None
    best_tau: float | None = None

    comparison_rows = []

    for tau in cfg.taus_self or []:
        print(f"    -> Đang chạy Self-Training với tau={tau:.2f}")
        st_cfg = SelfTrainingConfig(
            tau=float(tau),
            max_iter=10,
            min_new_per_iter=20,
            val_frac=0.20,
        )
        result = run_self_training(df=df, data_cfg=data_cfg, st_cfg=st_cfg)

        history = pd.DataFrame(result["history"])
        metrics = result["test_metrics"]

        hist_path = paths.data_processed / f"self_training_history_tau_{tau:.2f}.csv"
        history.to_csv(hist_path, index=False)

        metrics_path = (
            paths.data_processed / f"metrics_self_training_tau_{tau:.2f}.json"
        )
        _save_json(metrics, metrics_path)

        comparison_rows.append(
            {
                "tau": float(tau),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            }
        )

        print(
            f"       Test Accuracy={metrics['accuracy']:.4f}, "
            f"F1-macro={metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_result = result
            best_tau = float(tau)

    if best_result is None or best_tau is None:
        print("    Không chạy được Self-Training nào (danh sách tau rỗng?)")
        return

    # Lưu bảng tổng hợp so sánh theo tau
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = paths.data_processed / "self_training_comparison_by_tau.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"    -> Bảng so sánh theo tau: {comp_path}")

    # Lưu kết quả tốt nhất ra file chuẩn (phục vụ dashboard/báo cáo)
    print(f"    -> Chọn tau tốt nhất cho Self-Training: tau={best_tau:.2f}")
    best_metrics = best_result["test_metrics"]
    _save_json(
        best_metrics,
        paths.data_processed / "metrics_self_training.json",
    )

    pred_df = best_result["pred_df"]
    pred_sample_path = (
        paths.data_processed / "predictions_self_training_sample.csv"
    )
    pred_df.head(5000).to_csv(pred_sample_path, index=False)

    alerts_df = add_alert_columns(
        pred_df, pred_col="y_pred", severe_from=cfg.alert_from_class
    )
    alerts_sample_path = (
        paths.data_processed / "alerts_self_training_sample.csv"
    )
    alerts_df.head(5000).to_csv(alerts_sample_path, index=False)

    print(
        f"    -> metrics_self_training.json, predictions_self_training_sample.csv, "
        f"alerts_self_training_sample.csv được tạo dựa trên tau={best_tau:.2f}"
    )


def step_co_training_sweep(cfg: SemiExperimentConfig) -> None:
    """
    Chạy Co-Training cho nhiều giá trị tau, lưu:
    - co_training_history_tau_*.csv
    - metrics_co_training_tau_*.json
    Đồng thời chọn tau tốt nhất (F1-macro test) và lưu:
    - metrics_co_training.json
    - predictions_co_training_sample.csv
    - alerts_co_training_sample.csv
    """
    paths = get_paths()
    semi_path = paths.data_processed / "dataset_for_semi.parquet"
    if not semi_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {semi_path}. Hãy chạy step=semi_dataset hoặc step=all trước."
        )

    print("==> [5] Co-Training sweep theo tau")
    df = pd.read_parquet(semi_path)
    data_cfg = SemiDataConfig(cutoff=cfg.cutoff, random_state=cfg.random_state)

    best_f1 = -np.inf
    best_result = None
    best_tau: float | None = None

    comparison_rows = []

    for tau in cfg.taus_co or []:
        print(f"    -> Đang chạy Co-Training với tau={tau:.2f}")
        ct_cfg = CoTrainingConfig(
            tau=float(tau),
            max_iter=10,
            max_new_per_iter=500,
            min_new_per_iter=20,
            val_frac=0.20,
        )
        result = run_co_training(
            df=df,
            data_cfg=data_cfg,
            ct_cfg=ct_cfg,
            view1_cols=None,
            view2_cols=None,
        )

        history = pd.DataFrame(result["history"])
        metrics = result["test_metrics"]

        hist_path = paths.data_processed / f"co_training_history_tau_{tau:.2f}.csv"
        history.to_csv(hist_path, index=False)

        metrics_path = (
            paths.data_processed / f"metrics_co_training_tau_{tau:.2f}.json"
        )
        _save_json(metrics, metrics_path)

        comparison_rows.append(
            {
                "tau": float(tau),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            }
        )

        print(
            f"       Test Accuracy={metrics['accuracy']:.4f}, "
            f"F1-macro={metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_result = result
            best_tau = float(tau)

    if best_result is None or best_tau is None:
        print("    Không chạy được Co-Training nào (danh sách tau rỗng?)")
        return

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = paths.data_processed / "co_training_comparison_by_tau.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"    -> Bảng so sánh theo tau: {comp_path}")

    print(f"    -> Chọn tau tốt nhất cho Co-Training: tau={best_tau:.2f}")
    best_metrics = best_result["test_metrics"]
    _save_json(
        best_metrics,
        paths.data_processed / "metrics_co_training.json",
    )

    pred_df = best_result["pred_df"]
    pred_sample_path = (
        paths.data_processed / "predictions_co_training_sample.csv"
    )
    pred_df.head(5000).to_csv(pred_sample_path, index=False)

    alerts_df = add_alert_columns(
        pred_df, pred_col="y_pred", severe_from=cfg.alert_from_class
    )
    alerts_sample_path = (
        paths.data_processed / "alerts_co_training_sample.csv"
    )
    alerts_df.head(5000).to_csv(alerts_sample_path, index=False)

    print(
        f"    -> metrics_co_training.json, predictions_co_training_sample.csv, "
        f"alerts_co_training_sample.csv được tạo dựa trên tau={best_tau:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chạy pipeline Semi-Supervised AQI trực tiếp bằng Python (không cần notebook).\n"
            "Ví dụ: python -m src.semi_experiments --step all"
        )
    )
    parser.add_argument(
        "--step",
        choices=["baseline", "semi_dataset", "self", "co", "all"],
        default="all",
        help="Bước muốn chạy: baseline / semi_dataset / self / co / all (mặc định: all).",
    )
    parser.add_argument(
        "--label-missing-fraction",
        type=float,
        default=0.95,
        help="Tỉ lệ ẩn nhãn trên tập train cho semi-supervised (mặc định 0.95).",
    )
    parser.add_argument(
        "--taus-self",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 0.95],
        help="Danh sách tau cho Self-Training (vd: --taus-self 0.8 0.9 0.95).",
    )
    parser.add_argument(
        "--taus-co",
        type=float,
        nargs="+",
        default=[0.9],
        help="Danh sách tau cho Co-Training (vd: --taus-co 0.9 0.95).",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2017-01-01",
        help="Mốc thời gian tách train/test (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--alert-from-class",
        type=str,
        default="Unhealthy",
        help='Ngưỡng lớp AQI để tạo cảnh báo (vd: "Unhealthy").',
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed cho các bước bán giám sát.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SemiExperimentConfig(
        cutoff=args.cutoff,
        label_missing_fraction=args.label_missing_fraction,
        random_state=args.random_state,
        taus_self=list(args.taus_self),
        taus_co=list(args.taus_co),
        alert_from_class=args.alert_from_class,
    )

    if args.step in ("baseline", "all"):
        step_prepare_and_baseline(cutoff=cfg.cutoff)

    if args.step in ("semi_dataset", "all"):
        step_build_semi_dataset(cfg)

    if args.step in ("self", "all"):
        step_self_training_sweep(cfg)

    if args.step in ("co", "all"):
        step_co_training_sweep(cfg)


if __name__ == "__main__":
    main()

