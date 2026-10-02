"""
ai_service.py
เชื่อมต่อ Gemini เพื่ออธิบายผลการวิเคราะห์ระดับน้ำเป็นภาษาไทย
"""
from __future__ import annotations

import json
import os

try:
    import streamlit as st
except ImportError:  # ใช้งานนอก Streamlit ได้
    st = None

from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str, default: str | None = None) -> str | None:
    """อ่านค่าจาก Streamlit Secrets (ตอน Deploy) หรือไฟล์ .env (ตอนรันในเครื่อง)
    ค่าว่างหรือ "..." (ค่าตัวอย่าง) ถือว่ายังไม่ได้ตั้งค่า"""
    value = None
    if st is not None:
        try:
            if name in st.secrets:
                value = st.secrets[name]
        except Exception:
            pass
    if value is None:
        value = os.getenv(name)
    return default if value in (None, "", "...") else value


SYSTEM_PROMPT = """คุณคือผู้ช่วยวิเคราะห์สถานการณ์น้ำในแม่น้ำเจ้าพระยา
- ตอบเป็นภาษาไทยที่ประชาชนทั่วไปเข้าใจง่าย กระชับ ไม่เกิน 8 บรรทัด หรือใช้หัวข้อย่อย
- อ้างอิงเฉพาะตัวเลขที่ให้มาในข้อมูลเท่านั้น ห้ามแต่งตัวเลขหรือเหตุการณ์ขึ้นเอง
- อธิบายความหมายของตัวเลข เช่น ระยะห่างจากตลิ่ง แนวโน้ม ผลพยากรณ์ และระดับความเสี่ยง
- ถ้าข้อมูลไม่พอ ให้บอกตรง ๆ
- ปิดท้ายด้วย: "ข้อมูลนี้ใช้ประกอบการติดตามสถานการณ์เท่านั้น โปรดติดตามประกาศจากกรมชลประทานและหน่วยงานในพื้นที่"
"""

PRESET_PROMPTS = {
    "📈 แนวโน้ม": "ระดับน้ำที่สถานีนี้มีแนวโน้มขึ้นหรือลง เร็วแค่ไหน เทียบกับค่าเฉลี่ย 7 และ 30 วัน",
    "⚠️ ความเสี่ยง": "สถานีนี้เสี่ยงน้ำล้นตลิ่งแค่ไหน ใน 1-3 วันข้างหน้า ประชาชนริมน้ำควรเตรียมตัวอย่างไร",
    "🌊 สรุปภาพรวม": "สรุปสถานการณ์น้ำของสถานีนี้ และสถานีอื่นตลอดลำน้ำเจ้าพระยาแบบสั้น ๆ",
}


def ask_water_ai(question: str, context: dict) -> str:
    api_key = get_secret("GEMINI_API_KEY")
    model = get_secret("GEMINI_MODEL", "gemini-3.6-flash")
    if not api_key:
        return "⚠️ ยังไม่ได้ตั้งค่า GEMINI_API_KEY ในไฟล์ .env หรือ Streamlit Secrets"

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = (
        f"ข้อมูลการวิเคราะห์ (JSON):\n{json.dumps(context, ensure_ascii=False, default=str, indent=1)}"
        f"\n\nคำถามของผู้ใช้: {question}"
    )
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.3),
        )
        return resp.text or "(AI ไม่ได้ตอบกลับ)"
    except Exception as e:
        return f"⚠️ เรียก Gemini ไม่สำเร็จ: {e}"
