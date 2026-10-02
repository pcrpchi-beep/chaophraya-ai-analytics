"""
stats_service.py
คำนวณสถิติระดับน้ำ: การเปลี่ยนแปลงรายวัน, MA7, MA30, Volatility,
ระยะห่างจากตลิ่ง, % ความจุลำน้ำ และจัดระดับสถานการณ์น้ำ
"""
import pandas as pd

# เกณฑ์ระดับน้ำตามผังน้ำของ สสน. (% ความจุลำน้ำ)
RISK_LEVELS = [
    (100, "น้ำล้นตลิ่ง", "#d62728"),
    (80, "น้ำมาก", "#ff7f0e"),
    (30, "น้ำปกติ", "#2ca02c"),
    (10, "น้ำน้อย", "#e6b800"),
    (float("-inf"), "น้ำน้อยวิกฤต", "#8c564b"),
]


def risk_level(percent=None, freeboard=None) -> tuple[str, str]:
    """จัดระดับสถานการณ์น้ำ: ถ้าน้ำสูงกว่าตลิ่งแล้ว ถือว่าล้นตลิ่งทันที"""
    if freeboard is not None and pd.notna(freeboard) and freeboard < 0:
        return RISK_LEVELS[0][1], RISK_LEVELS[0][2]
    if percent is None or pd.isna(percent):
        return "ไม่มีข้อมูล", "#999999"
    for th, label, color in RISK_LEVELS:
        if percent >= th:
            return label, color
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """เพิ่มคอลัมน์สถิติให้ข้อมูลของสถานีเดียว (เรียงตามวันที่)"""
    d = df.sort_values("date").copy()
    d["change"] = d["water_level"].diff()                       # ม./วัน
    d["MA7"] = d["water_level"].rolling(7, min_periods=1).mean()
    d["MA30"] = d["water_level"].rolling(30, min_periods=1).mean()
    d["freeboard"] = d["bank_level"] - d["water_level"]         # + = ต่ำกว่าตลิ่ง, - = ล้นตลิ่ง
    return d


def _r(x, n=2):
    return round(float(x), n) if x is not None and pd.notna(x) else None


def calculate_statistics(df: pd.DataFrame, period_days: int = 30) -> dict:
    """สรุปสถิติของสถานีเดียวในช่วงเวลาที่เลือก"""
    if df.empty:
        return {"risk": "ไม่มีข้อมูล", "risk_color": "#999999"}
    d = add_features(df)
    d = d[d["date"] >= d["date"].max() - pd.Timedelta(days=period_days)]
    last, first = d.iloc[-1], d.iloc[0]
    prev7 = d[d["date"] <= last["date"] - pd.Timedelta(days=7)]
    risk, color = risk_level(last["percent"], last["freeboard"])
    return {
        "date": last["date"].date().isoformat(),
        "water_level": _r(last["water_level"]),
        "bank_level": _r(last["bank_level"]),
        "freeboard": _r(last["freeboard"]),
        "percent": _r(last["percent"], 1),
        "discharge": _r(last.get("discharge"), 0),
        "daily_change": _r(last["change"]),
        "trend_7d": _r(last["water_level"] - prev7["water_level"].iloc[-1]) if not prev7.empty else None,
        "change_period": _r(last["water_level"] - first["water_level"]),
        "MA7": _r(last["MA7"]),
        "MA30": _r(last["MA30"]),
        "volatility": _r(d["change"].std(), 3) if d["change"].notna().sum() > 1 else None,
        "period_high": _r(d["water_level"].max()),
        "period_low": _r(d["water_level"].min()),
        "days": int(len(d)),
        "risk": risk,
        "risk_color": color,
    }
