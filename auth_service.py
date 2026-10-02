"""
auth_service.py
ระบบสมัครสมาชิก / เข้าสู่ระบบด้วย Firebase Authentication (Email/Password)
เรียกผ่าน REST API จึงใช้แค่ FIREBASE_API_KEY (Web API Key ในหน้า Project settings)
"""
import requests

from ai_service import get_secret

BASE = "https://identitytoolkit.googleapis.com/v1/accounts"

ERRORS_TH = {
    "EMAIL_EXISTS": "อีเมลนี้ถูกใช้สมัครแล้ว",
    "EMAIL_NOT_FOUND": "ไม่พบบัญชีนี้",
    "INVALID_PASSWORD": "รหัสผ่านไม่ถูกต้อง",
    "INVALID_LOGIN_CREDENTIALS": "อีเมลหรือรหัสผ่านไม่ถูกต้อง",
    "INVALID_EMAIL": "รูปแบบอีเมลไม่ถูกต้อง",
    "USER_DISABLED": "บัญชีนี้ถูกระงับ",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "พยายามหลายครั้งเกินไป กรุณาลองใหม่ภายหลัง",
}


def _call(endpoint: str, email: str, password: str) -> dict:
    key = get_secret("FIREBASE_API_KEY")
    if not key:
        raise RuntimeError("ยังไม่ได้ตั้งค่า FIREBASE_API_KEY")
    r = requests.post(f"{BASE}:{endpoint}?key={key}",
                      json={"email": email, "password": password, "returnSecureToken": True},
                      timeout=15)
    data = r.json()
    if r.status_code != 200:
        msg = data.get("error", {}).get("message", "UNKNOWN")
        code = msg.split(" ")[0]
        if code.startswith("WEAK_PASSWORD"):
            raise RuntimeError("รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
        raise RuntimeError(ERRORS_TH.get(code, msg))
    return data


def register(email: str, password: str) -> dict:
    return _call("signUp", email, password)


def login(email: str, password: str) -> dict:
    return _call("signInWithPassword", email, password)


def auth_enabled() -> bool:
    return bool(get_secret("FIREBASE_API_KEY"))
