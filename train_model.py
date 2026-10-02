"""
train_model.py
เทรนโมเดลพยากรณ์ระดับน้ำจาก data/history.csv แล้วบันทึกเป็น model/water_model.pkl

รัน:  python train_model.py            (ใช้ไฟล์อย่างเดียว)
      python train_model.py --update   (ดึงข้อมูลล่าสุดจากเว็บก่อน แล้วบันทึก CSV ใหม่)
"""
import sys

import pandas as pd

from ml_service import MODEL_PATH, save_bundle, train_all
from water_service import HISTORY_CSV, get_history, load_history_csv

if __name__ == "__main__":
    if "--update" in sys.argv:
        hist, src = get_history(live=True, max_gap_days=365)
        hist.to_csv(HISTORY_CSV, index=False)
        print(f"อัปเดตข้อมูล: {src}")
    else:
        hist = load_history_csv()

    print(f"ข้อมูล {len(hist):,} แถว | {hist['date'].min().date()} ถึง {hist['date'].max().date()}")
    print(hist.groupby("station").size().to_string(), "\n")

    bundle = train_all(hist)
    save_bundle(bundle)

    m = pd.DataFrame(bundle["metrics"])
    cols = [c for c in m.columns if c.endswith("|MAE")] + ["MAE"]
    summary = m.groupby("horizon")[cols].mean().round(3)
    summary.columns = [c.replace("|MAE", "") for c in cols[:-1]] + ["โมเดลที่เลือก"]
    print("\nMAE เฉลี่ยทุกสถานีบนชุด Test (หน่วย: เมตร)")
    print(summary.to_string())
    m.round(4).to_csv(MODEL_PATH.parent / "metrics.csv", index=False)
    print(f"\nบันทึกโมเดลที่ {MODEL_PATH}")
