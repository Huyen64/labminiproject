from __future__ import annotations

"""
Chạy:
    streamlit run dashboard.py
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st


DATA_DIR = Path("data/processed")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def show_overall_metrics() -> None:
    st.subheader("So sánh hiệu năng trên tập test")

    baseline_metrics = _load_json(DATA_DIR / "metrics.json")
    self_metrics = _load_json(DATA_DIR / "metrics_self_training.json")
    co_metrics = _load_json(DATA_DIR / "metrics_co_training.json")

    rows = []
    if baseline_metrics is not None:
        rows.append(
            {
                "model": "Baseline",
                "accuracy": baseline_metrics["accuracy"],
                "f1_macro": baseline_metrics["f1_macro"],
            }
        )
    if self_metrics is not None:
        rows.append(
            {
                "model": "Self-Training",
                "accuracy": self_metrics["accuracy"],
                "f1_macro": self_metrics["f1_macro"],
            }
        )
    if co_metrics is not None:
        rows.append(
            {
                "model": "Co-Training",
                "accuracy": co_metrics["accuracy"],
                "f1_macro": co_metrics["f1_macro"],
            }
        )

    if not rows:
        st.info("Chưa tìm thấy file metrics. Hãy chạy script semi_experiments trước.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    st.bar_chart(df.set_index("model")[["accuracy", "f1_macro"]])


def show_history_section() -> None:
    st.subheader("Diễn biến Self-Training / Co-Training theo vòng lặp")

    history_files = sorted(DATA_DIR.glob("*training_history_tau_*.csv"))
    if not history_files:
        st.info(
            "Chưa tìm thấy file history (_training_history_tau_*.csv). "
            "Hãy chạy self-training/co-training bằng script semi_experiments."
        )
        return

    file_labels = {f.name: f for f in history_files}
    choice = st.selectbox("Chọn file history", list(file_labels.keys()))
    df = pd.read_csv(file_labels[choice])

    st.write("Bảng history (một vài dòng đầu):")
    st.dataframe(df.head(20))

    if {"iter", "val_f1_macro", "val_accuracy"}.issubset(df.columns):
        st.line_chart(
            df.set_index("iter")[["val_accuracy", "val_f1_macro"]],
            use_container_width=True,
        )

    if {"iter", "new_pseudo", "unlabeled_pool"}.issubset(df.columns):
        st.line_chart(
            df.set_index("iter")[["new_pseudo", "unlabeled_pool"]],
            use_container_width=True,
        )


def show_alerts_section() -> None:
    st.subheader("Cảnh báo AQI theo trạm")

    model_choice = st.selectbox(
        "Chọn mô hình để xem cảnh báo",
        ["Self-Training", "Co-Training", "Baseline"],
        index=0,
    )

    alerts_file_map = {
        "Self-Training": DATA_DIR / "alerts_self_training_sample.csv",
        "Co-Training": DATA_DIR / "alerts_co_training_sample.csv",
        "Baseline": None,  # baseline không sinh file alerts sẵn
    }

    path = alerts_file_map[model_choice]
    if path is None or not path.exists():
        st.info(
            "Hiện tại chỉ Self-Training / Co-Training có file alerts mẫu. "
            "Hãy kiểm tra lại sau khi chạy semi_experiments."
        )
        return

    df = pd.read_csv(path, parse_dates=["datetime"])
    if "station" not in df.columns:
        st.warning("File alerts không có cột 'station'. Không thể lọc theo trạm.")
        st.dataframe(df.head(50))
        return

    stations = sorted(df["station"].dropna().unique().tolist())
    if not stations:
        st.info("Không tìm thấy trạm nào trong file alerts.")
        return

    station = st.selectbox("Chọn trạm", stations)
    df_s = df[df["station"] == station].copy()
    df_s = df_s.sort_values("datetime")

    st.write(f"Số bản ghi cho trạm **{station}**: {len(df_s)}")

    if "is_alert" in df_s.columns:
        st.line_chart(
            df_s.set_index("datetime")[["is_alert"]],
            use_container_width=True,
        )

    st.write("Một vài dòng dữ liệu:")
    st.dataframe(df_s.head(100))


def main() -> None:
    st.set_page_config(
        page_title="AQI Semi-Supervised Dashboard",
        layout="wide",
    )
    st.title("Air Quality – Baseline vs Self-Training vs Co-Training")
    st.markdown(
        "Dashboard này đọc trực tiếp các file trong `data/processed` "
        "được sinh bởi script `src/semi_experiments.py`."
    )

    show_overall_metrics()
    st.markdown("---")
    show_history_section()
    st.markdown("---")
    show_alerts_section()


if __name__ == "__main__":
    main()

