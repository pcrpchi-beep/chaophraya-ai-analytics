"""
ml_service.py
แบบจำลอง Machine Learning พยากรณ์ระดับน้ำล่วงหน้า 1-3 วัน

แนวคิด
- Target  = ระดับน้ำที่ "เปลี่ยนไป" ในอีก h วัน (h = 1, 2, 3) แล้วบวกกลับกับระดับน้ำวันนี้
- Features = การเปลี่ยนแปลงของระดับน้ำสถานีนั้น + การเปลี่ยนแปลงของสถานีต้นน้ำ
             + การเปลี่ยนแปลงของปริมาณน้ำไหลผ่าน (C2, C13)
  (ไม่ใช้ค่าระดับน้ำสัมบูรณ์ เพราะทำให้โมเดลแบบต้นไม้จำค่าเดิม และทำนายผิดเมื่อเจอระดับน้ำที่ไม่เคยเห็น)
- เปรียบเทียบ 4 โมเดล: Persistence (baseline), Ridge Regression, Random Forest, Gradient Boosting
- แบ่งข้อมูลตามเวลา: Train 80% แรก | Test 20% ล่าสุด (ไม่สุ่ม เพราะเป็นอนุกรมเวลา)
- เลือกโมเดลด้วย Time-Series Cross-Validation บนชุด Train แล้วรายงานผลบน Test ที่โมเดลไม่เคยเห็น
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from water_service import MAIN_STATIONS

MODEL_PATH = Path(__file__).parent / "model" / "water_model.pkl"
HORIZONS = (1, 2, 3)
STATION_ORDER = list(MAIN_STATIONS)


class Persistence(BaseEstimator, RegressorMixin):
    """Baseline: สมมติว่าระดับน้ำในอนาคตเท่ากับวันนี้ (การเปลี่ยนแปลง = 0)"""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


CANDIDATES = {
    "Persistence (Baseline)": Persistence(),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                                                   subsample=0.8, random_state=42),
}


def to_wide(history: pd.DataFrame) -> pd.DataFrame:
    """แปลงข้อมูลยาว -> ตารางรายวัน (index = วันที่, คอลัมน์ = wl_C2, q_C2, ...)"""
    h = history[history["station"].isin(STATION_ORDER)]
    wl = h.pivot_table(index="date", columns="station", values="water_level")
    wl.columns = [f"wl_{c}" for c in wl.columns]
    parts = [wl]
    if h["discharge"].notna().any():
        q = h.pivot_table(index="date", columns="station", values="discharge")
        q.columns = [f"q_{c}" for c in q.columns]
        parts.append(q)
    wide = pd.concat(parts, axis=1).sort_index().asfreq("D")
    return wide.interpolate(limit=3, limit_area="inside")


def upstream_of(station: str) -> list[str]:
    """สถานีต้นน้ำ (อยู่ก่อนหน้าในลำดับ MAIN_STATIONS)"""
    return STATION_ORDER[: STATION_ORDER.index(station)]


def build_features(wide: pd.DataFrame, station: str) -> pd.DataFrame:
    """คอลัมน์ 'wl' = ระดับน้ำวันนี้ (ใช้คำนวณ target ไม่ได้ใช้เป็น feature)"""
    wl = wide.get(f"wl_{station}")
    if wl is None:
        return pd.DataFrame()
    X = pd.DataFrame(index=wide.index)
    X["wl"] = wl
    for k in (1, 2, 3, 7):
        X[f"chg{k}"] = wl.diff(k)
    X["ma7_gap"] = wl - wl.rolling(7, min_periods=3).mean()
    if f"q_{station}" in wide:
        X["q_chg1"] = wide[f"q_{station}"].diff(1)
        X["q_chg3"] = wide[f"q_{station}"].diff(3)
    for up in upstream_of(station):
        if f"wl_{up}" in wide:
            X[f"up_{up}_chg1"] = wide[f"wl_{up}"].diff(1)
            X[f"up_{up}_chg3"] = wide[f"wl_{up}"].diff(3)
        if f"q_{up}" in wide:
            X[f"up_{up}_qchg1"] = wide[f"q_{up}"].diff(1)
    return X


def feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c != "wl"]


def _metrics(y_true, y_pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def train_all(history: pd.DataFrame, test_size: float = 0.2, verbose: bool = True) -> dict:
    """เทรน + เลือกโมเดล + ประเมินผล ทุกสถานี ทุก horizon"""
    wide = to_wide(history)
    bundle = {"models": {}, "features": {}, "selected": {}, "residual_q": {},
              "metrics": [], "importance": {}, "trained_until": str(wide.index.max().date())}
    for st in STATION_ORDER:
        X_all = build_features(wide, st)
        if X_all.empty:
            continue
        feats = feature_cols(X_all)
        for h in HORIZONS:
            y = X_all["wl"].shift(-h) - X_all["wl"]
            data = X_all.assign(y=y).dropna()
            if len(data) < 120:
                continue
            n_test = int(len(data) * test_size)
            train, test = data.iloc[:-n_test], data.iloc[-n_test:]

            # 1) เลือกโมเดลด้วย Time-Series Cross-Validation 4 รอบบนชุด Train
            #    (หลายช่วงเวลา ทั้งฤดูน้ำหลากและฤดูแล้ง จึงเลือกได้น่าเชื่อถือกว่าแบ่งครั้งเดียว)
            val_mae = {name: [] for name in CANDIDATES}
            for tr_idx, va_idx in TimeSeriesSplit(n_splits=4).split(train):
                tr, va = train.iloc[tr_idx], train.iloc[va_idx]
                for name, model in CANDIDATES.items():
                    m = clone(model).fit(tr[feats], tr["y"])
                    val_mae[name].append(mean_absolute_error(va["y"], m.predict(va[feats])))
            cv_mae = {k: float(np.mean(v)) for k, v in val_mae.items()}
            best = min(cv_mae, key=cv_mae.get)

            # 2) ประเมินบน Test (ระดับน้ำจริง vs ระดับน้ำที่ทำนาย)
            actual = test["wl"] + test["y"]
            row = {"station": st, "horizon": h, "n_train": len(train), "n_test": len(test), "selected": best}
            test_pred = {}
            for name, model in CANDIDATES.items():
                m = clone(model).fit(train[feats], train["y"])
                test_pred[name] = m.predict(test[feats])
                met = _metrics(actual, test["wl"] + test_pred[name])
                row[f"{name}|MAE"] = met["MAE"]
                if name == best:
                    row.update({"MAE": met["MAE"], "RMSE": met["RMSE"], "R2": met["R2"]})
                if name == "Random Forest" and h == 1:
                    bundle["importance"][st] = (pd.Series(m.feature_importances_, index=feats)
                                                .sort_values(ascending=False))
            bundle["metrics"].append(row)
            resid = test["y"].values - test_pred[best]
            bundle["residual_q"][(st, h)] = (float(np.quantile(resid, 0.1)), float(np.quantile(resid, 0.9)))

            # 3) เทรนโมเดลที่เลือกใหม่ด้วยข้อมูลทั้งหมด เพื่อใช้งานจริง
            bundle["models"][(st, h)] = clone(CANDIDATES[best]).fit(data[feats], data["y"])
            bundle["features"][(st, h)] = feats
            bundle["selected"][(st, h)] = best
            if verbose:
                print(f"{st:7s} h={h}: เลือก {best:22s} Test MAE={row['MAE']:.3f} ม. "
                      f"(Baseline {row['Persistence (Baseline)|MAE']:.3f})")
    return bundle


def save_bundle(bundle: dict, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:  # เวอร์ชัน scikit-learn ไม่ตรงกัน -> ให้แอปเทรนใหม่
        return None


def forecast(bundle: dict, history: pd.DataFrame, station: str) -> pd.DataFrame:
    """พยากรณ์ระดับน้ำ 1-3 วันข้างหน้า จากข้อมูลวันล่าสุดของสถานีนั้น"""
    wide = to_wide(history)
    X = build_features(wide, station)
    rows = []
    for h in HORIZONS:
        model = bundle["models"].get((station, h))
        if model is None or X.empty:
            continue
        feats = bundle["features"][(station, h)]
        last = X[feats + ["wl"]].dropna().tail(1)
        if last.empty:
            continue
        base_date = last.index[0]
        wl0 = float(last["wl"].iloc[0])
        pred = wl0 + float(model.predict(last[feats])[0])
        lo, hi = bundle["residual_q"].get((station, h), (0.0, 0.0))
        rows.append({"base_date": base_date, "date": base_date + pd.Timedelta(days=h), "horizon": h,
                     "forecast": pred, "lower": pred + lo, "upper": pred + hi,
                     "model": bundle["selected"][(station, h)]})
    return pd.DataFrame(rows)


def label_feature(f: str) -> str:
    """ชื่อ feature ภาษาไทยสำหรับแสดงในกราฟ"""
    fixed = {"ma7_gap": "ห่างจากค่าเฉลี่ย 7 วัน",
             "q_chg1": "ปริมาณน้ำเปลี่ยน 1 วัน", "q_chg3": "ปริมาณน้ำเปลี่ยน 3 วัน"}
    if f in fixed:
        return fixed[f]
    if f.startswith("chg"):
        return f"ระดับน้ำเปลี่ยนใน {f[3:]} วัน"
    if f.startswith("up_"):
        _, st, kind = f.split("_", 2)
        name = MAIN_STATIONS.get(st, {}).get("name", st)
        return {"chg1": f"ต้นน้ำ {st} {name} เปลี่ยน 1 วัน",
                "chg3": f"ต้นน้ำ {st} {name} เปลี่ยน 3 วัน",
                "qchg1": f"ปริมาณน้ำต้นน้ำ {st} เปลี่ยน 1 วัน"}.get(kind, f)
    return f
