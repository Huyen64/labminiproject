from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA

def get_aqi_status(pm25):
    if pm25 <= 35.4: return "Good"
    elif pm25 <= 75.4: return "Moderate"
    elif pm25 <= 115.4: return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4: return "Unhealthy"
    elif pm25 <= 250.4: return "Very Unhealthy"
    return "Hazardous"

class AirGuardForecaster:
    """
    Hệ thống dự báo PM2.5 tích hợp cảnh báo AQI theo trạm (AIR GUARD)
    """
    def __init__(self, station_name: str, value_col: str = "PM2.5"):
        self.station = station_name
        self.value_col = value_col
        self.model_result = None
        self.best_order = (1, 1, 1)

    def prepare_data(self, df: pd.DataFrame, freq: str = "H"):
        """Gộp các bước xử lý chuỗi thời gian: Lọc trạm, Resample, Interpolate"""
        sdf = df[df["station"] == self.station].copy()
        sdf = sdf.sort_values("datetime")
        s = pd.to_numeric(sdf[self.value_col], errors="coerce")
        s.index = pd.DatetimeIndex(sdf["datetime"].values)
        s = s.resample(freq).mean()
        s = s.where((s.isna()) | (s >= 0), 0.0)
        self.series = s.interpolate(method="time", limit_direction="both")
        return self.series

    def find_best_params(self, p_max=3, q_max=3, d_max=2):
        """Tự động tìm p, d, q tối ưu (Grid Search)"""
        x = self.series.dropna()
        d = 0
        while d <= d_max:
            if adfuller(x.diff(d).dropna() if d>0 else x)[1] < 0.05:
                break
            d += 1
        
        best_aic = np.inf
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                try:
                    m = ARIMA(x, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        self.best_order = (p, d, q)
                except: continue
        return self.best_order

    def forecast_with_alerts(self, train_size=0.8, threshold=115.5):
        """Dự báo và sinh cảnh báo AQI (Mục tiêu chính của AIR GUARD)"""
        split_idx = int(len(self.series) * train_size)
        train, test = self.series[:split_idx], self.series[split_idx:]
        
        model = ARIMA(train, order=self.best_order).fit()
        forecast = model.get_forecast(steps=len(test))
        
        y_pred = forecast.predicted_mean
        results = pd.DataFrame({
            "actual": test.values,
            "predicted": y_pred.values,
            "status": [get_aqi_status(v) for v in y_pred.values],
            "is_alert": y_pred.values > threshold
        }, index=test.index)
        
        return results
def run_air_guard_pipeline(df_path, station="Aotizhongxin"):
    df = pd.read_parquet(df_path)
    guard = AirGuardForecaster(station)
    guard.prepare_data(df)
    guard.find_best_params(p_max=3, q_max=3) 
    report = guard.forecast_with_alerts()
    print(f"--- BÁO CÁO CẢNH BÁO TRẠM {station} ---")
    print(report[report['is_alert'] == True].head())
    return report