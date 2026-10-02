"""
water_service.py
ดึงข้อมูลระดับน้ำแม่น้ำเจ้าพระยาจากเว็บ สสน. (HII)
https://tiwrm.hii.or.th/DATA/REPORT/php/chart/chaopraya/small/chaopraya.php

หน้าเว็บนี้ฝังข้อมูลทั้งหมดไว้ในตัวแปร JavaScript ชื่อ `json_data` ประกอบด้วย
  - tele      : สถานีโทรมาตรของ สสน. (ระดับน้ำ, ระดับตลิ่ง, % ความจุลำน้ำ)  -> มีข้อมูลย้อนหลังทุกวัน
  - itc_water : สถานีของกรมชลประทาน (C2, C13 มีปริมาณน้ำไหลผ่าน ลบ.ม./วินาที)
ดูข้อมูลวันที่ย้อนหลังได้โดยส่ง POST  datepicker=YYYY-MM-DD
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

SOURCE_URL = "https://tiwrm.hii.or.th/DATA/REPORT/php/chart/chaopraya/small/chaopraya.php"
HISTORY_CSV = Path(__file__).parent / "data" / "history.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (ChaoPhraya-AI student project)"}

# สถานีหลัก เรียงจากต้นน้ำ -> ปลายน้ำ (ลำดับนี้ใช้กำหนด "สถานีต้นน้ำ" ในโมเดล ML)
MAIN_STATIONS = {
    "PIN005": {"name": "เก้าเลี้ยว", "province": "นครสวรรค์", "river": "แม่น้ำปิง", "src": "tele"},
    "NAN008": {"name": "ชุมแสง", "province": "นครสวรรค์", "river": "แม่น้ำน่าน", "src": "tele"},
    "CPY001": {"name": "สะพานเดชาติวงศ์", "province": "นครสวรรค์", "river": "เจ้าพระยา", "src": "tele"},
    "C2":     {"name": "ค่ายจิรประวัติ (C.2)", "province": "นครสวรรค์", "river": "เจ้าพระยา", "src": "itc"},
    "C13":    {"name": "ท้ายเขื่อนเจ้าพระยา (C.13)", "province": "ชัยนาท", "river": "เจ้าพระยา", "src": "itc"},
    "CPY005": {"name": "สรรพยา", "province": "ชัยนาท", "river": "เจ้าพระยา", "src": "tele"},
    "CPY008": {"name": "เมืองอ่างทอง", "province": "อ่างทอง", "river": "เจ้าพระยา", "src": "tele"},
    "CPY014": {"name": "สะพานนวลฉวี", "province": "นนทบุรี", "river": "เจ้าพระยา", "src": "tele"},
}

COLUMNS = ["date", "station", "name", "water_level", "bank_level", "ground_level",
           "percent", "discharge", "qmax"]
_JSON_RE = re.compile(r"var json_data = (\[.*?\]);\s*\n", re.S)


def _f(x):
    if x in (None, "", "-"):
        return None
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def parse_page(html: str) -> pd.DataFrame:
    """แยกข้อมูลสถานีหลักออกจาก HTML ของหน้าเว็บ"""
    m = _JSON_RE.search(html)
    if not m:
        raise ValueError("ไม่พบตัวแปร json_data ในหน้าเว็บ (โครงสร้างเว็บอาจเปลี่ยน)")
    data = json.loads(m.group(1))[0]
    rows = []
    for v in (data.get("tele") or {}).values():
        code = v.get("code")
        if code in MAIN_STATIONS and v.get("water1") is not None:
            rows.append({
                "date": v.get("date"), "station": code, "name": MAIN_STATIONS[code]["name"],
                "water_level": _f(v.get("water1")),        # ม.รทก.
                "bank_level": _f(v.get("bank")),           # ระดับตลิ่ง ม.รทก.
                "ground_level": _f(v.get("ground_level")), # ระดับท้องน้ำ ม.รทก.
                "percent": _f(v.get("percent")),           # % ความจุลำน้ำ (สสน. คำนวณ)
                "discharge": None, "qmax": None,
            })
    itc = data.get("itc_water") or {}
    for code, meta in MAIN_STATIONS.items():
        v = itc.get(code)
        if meta["src"] == "itc" and v and v.get("water_l") is not None:
            q, qmax = _f(v.get("storage")), _f(v.get("qmax"))
            rows.append({
                "date": v.get("date"), "station": code, "name": meta["name"],
                "water_level": _f(v.get("water_l")), "bank_level": _f(v.get("r_bank")),
                "ground_level": _f(v.get("ground_level")),
                "percent": q / qmax * 100 if q is not None and qmax else None,
                "discharge": q,                             # ลบ.ม./วินาที
                "qmax": qmax,                               # ความจุลำน้ำ ลบ.ม./วินาที
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_snapshot(day: date | str | None = None, timeout: int = 30) -> pd.DataFrame:
    """ดึงข้อมูลสถานีหลักของวันที่กำหนด (ไม่ใส่ = วันนี้)"""
    day = str(day or date.today())
    resp = requests.post(SOURCE_URL, data={"datepicker": day}, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return parse_page(resp.text)


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """ใช้วันที่ที่สถานีรายงานจริง, ตัดข้อมูลซ้ำ และค่าที่ผิดปกติ"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["water_level", "bank_level", "ground_level", "percent", "discharge", "qmax"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "water_level"])
    # ระดับน้ำต่ำกว่าท้องน้ำมาก ๆ = เซนเซอร์ผิดพลาด
    bad = df["ground_level"].notna() & (df["water_level"] < df["ground_level"] - 1)
    df = df[~bad]
    df = df.drop_duplicates(subset=["station", "date"], keep="last").sort_values(["station", "date"])
    return df[COLUMNS].reset_index(drop=True)


def fetch_history(days: int = 30, end: date | None = None, workers: int = 6) -> pd.DataFrame:
    """ดึงข้อมูลย้อนหลังทีละวันจากเว็บ (ประมาณ 0.5–1 วินาทีต่อวัน)"""
    end = end or date.today()
    dates = [str(end - timedelta(days=k)) for k in range(days)]

    def one(d):
        try:
            return fetch_snapshot(d)
        except Exception:
            return pd.DataFrame(columns=COLUMNS)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        frames = [f for f in ex.map(one, dates) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return clean_history(pd.concat(frames, ignore_index=True))


def load_history_csv(path: Path = HISTORY_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["station", "date"]).reset_index(drop=True)


def get_history(live: bool = True, max_gap_days: int = 90) -> tuple[pd.DataFrame, str]:
    """
    ข้อมูลย้อนหลังจาก data/history.csv + ดึงวันที่ขาดจากเว็บ สสน. ให้ถึงวันนี้
    คืนค่า (DataFrame, คำอธิบายแหล่งข้อมูล)
    """
    base = load_history_csv()
    source = "ไฟล์ data/history.csv"
    if live:
        try:
            last = base["date"].max().date() if not base.empty else date.today() - timedelta(days=30)
            gap = max(1, min((date.today() - last).days + 1, max_gap_days))
            fresh = fetch_history(days=gap)
            if fresh.empty:
                raise RuntimeError("เว็บไม่ส่งข้อมูลกลับมา")
            base = clean_history(pd.concat([base, fresh], ignore_index=True))
            source = "เว็บ สสน. tiwrm.hii.or.th (ล่าสุด) + ไฟล์ย้อนหลัง"
        except Exception as e:  # เว็บล่ม / เน็ตหลุด -> ใช้ไฟล์อย่างเดียว
            source += f" (ดึงข้อมูลสดไม่สำเร็จ: {e})"
    return base, source


if __name__ == "__main__":
    print(fetch_snapshot())
