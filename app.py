"""
app.py — ChaoPhraya AI Water Monitor
Web Application วิเคราะห์และพยากรณ์ระดับน้ำแม่น้ำเจ้าพระยาด้วย AI
รัน:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_service import PRESET_PROMPTS, ask_water_ai
from auth_service import auth_enabled, login, register
from ml_service import forecast, label_feature, load_bundle, save_bundle, train_all
from stats_service import add_features, calculate_statistics
from water_service import MAIN_STATIONS, SOURCE_URL, get_history

st.set_page_config(page_title="ChaoPhraya AI Water Monitor", page_icon="🌊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(120deg,#0b3d5c,#11698e); color:#fff; padding:18px 24px;
       border-radius:14px; margin-bottom:14px;}
.hero h1 {font-size:1.6rem; margin:0;}
.hero p {margin:4px 0 0; opacity:.85;}
.badge {display:inline-block; padding:4px 12px; border-radius:999px; color:#fff; font-weight:600;}
.st-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:10px 12px; height:100%;}
.st-card .code {font-size:.8rem; opacity:.7;}
.st-card .wl {font-size:1.35rem; font-weight:700;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- Login
def login_page():
    st.markdown('<div class="hero"><h1>🌊 ChaoPhraya AI Water Monitor</h1>'
                '<p>เข้าสู่ระบบหรือสมัครสมาชิกเพื่อใช้งาน</p></div>', unsafe_allow_html=True)
    tab_login, tab_reg = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])
    with tab_login, st.form("login"):
        email = st.text_input("Email", placeholder="name@example.com")
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Login", width="stretch"):
            try:
                user = login(email, pw)
                st.session_state.user = user["email"]
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with tab_reg, st.form("register"):
        email = st.text_input("Email ", placeholder="name@example.com")
        pw = st.text_input("Password (อย่างน้อย 6 ตัว)", type="password")
        if st.form_submit_button("สมัครสมาชิก", width="stretch"):
            try:
                register(email, pw)
                st.success("สมัครสำเร็จ กรุณาเข้าสู่ระบบที่แท็บ 'เข้าสู่ระบบ'")
            except Exception as e:
                st.error(str(e))


if auth_enabled() and "user" not in st.session_state:
    login_page()
    st.stop()


# ---------------------------------------------------------------- Data & model
@st.cache_data(ttl=1800, show_spinner="กำลังดึงข้อมูลระดับน้ำจาก สสน. ...")
def load_data():
    return get_history()


@st.cache_resource(show_spinner="กำลังเตรียมโมเดล AI ...")
def load_model(trained_hint: str):
    bundle = load_bundle()
    if bundle is None:  # ยังไม่มีไฟล์โมเดล -> เทรนจากข้อมูลที่มี
        hist, _ = load_data()
        bundle = train_all(hist, verbose=False)
        save_bundle(bundle)
    return bundle


history, data_source = load_data()
if history.empty:
    st.error("ไม่มีข้อมูลระดับน้ำ: ดึงจากเว็บไม่สำเร็จและไม่พบไฟล์ data/history.csv")
    st.stop()
bundle = load_model("v1")

# ---------------------------------------------------------------- Header & sidebar
st.markdown('<div class="hero"><h1>🌊 ChaoPhraya AI Water Monitor</h1>'
            '<p>วิเคราะห์และพยากรณ์ระดับน้ำแม่น้ำเจ้าพระยาด้วย Machine Learning + Generative AI</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    if "user" in st.session_state:
        st.caption(f"เข้าสู่ระบบเป็น **{st.session_state.user}**")
        if st.button("Logout", width="stretch"):
            st.session_state.clear()
            st.rerun()
        st.divider()
    labels = {k: f"{k} · {v['name']} ({v['province']})" for k, v in MAIN_STATIONS.items()}
    available = [s for s in MAIN_STATIONS if s in set(history["station"])]
    station = st.selectbox("เลือกสถานี", available, index=available.index("CPY008") if "CPY008" in available else 0,
                           format_func=lambda s: labels[s])
    period = st.radio("ช่วงเวลา", [7, 30, 90, 365], index=1, horizontal=True,
                      format_func=lambda d: f"{d} วัน")
    if st.button("🔄 ดึงข้อมูลล่าสุด", width="stretch"):
        load_data.clear()
        st.rerun()
    st.caption(f"แหล่งข้อมูล: {data_source}")
    st.caption(f"[เปิดผังน้ำ สสน.]({SOURCE_URL})")

st_hist = history[history["station"] == station]
stats = calculate_statistics(st_hist, period)
fc = forecast(bundle, history, station)
info = MAIN_STATIONS[station]

tab_over, tab_data, tab_stats, tab_fc, tab_ai = st.tabs(
    ["ภาพรวม", "ข้อมูลระดับน้ำ", "วิเคราะห์สถิติ", "พยากรณ์ (ML)", "AI Insights"])

# ---------------------------------------------------------------- Overview
with tab_over:
    st.subheader("สถานการณ์ตลอดลำน้ำ (ต้นน้ำ → ปลายน้ำ)")
    cols = st.columns(4)
    for i, s in enumerate(available):
        c = cols[i % 4]
        sstat = calculate_statistics(history[history["station"] == s], 7)
        chg = sstat.get("daily_change")
        arrow = "▲" if chg and chg > 0 else ("▼" if chg and chg < 0 else "•")
        c.markdown(
            f'<div class="st-card"><div class="code">{s} · {MAIN_STATIONS[s]["province"]}</div>'
            f'<div class="wl">{sstat.get("water_level")} <span style="font-size:.8rem">ม.รทก.</span></div>'
            f'<div>{arrow} {chg if chg is not None else "-"} ม./วัน</div>'
            f'<span class="badge" style="background:{sstat["risk_color"]};font-size:.75rem">{sstat["risk"]}</span>'
            f'</div><div style="height:10px"></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### {station} · {info['name']} จ.{info['province']}")
    st.markdown(f'<span class="badge" style="background:{stats["risk_color"]}">สถานะ: {stats["risk"]}</span>'
                f' &nbsp; ข้อมูลวันที่ {stats["date"]}', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ระดับน้ำ (ม.รทก.)", stats["water_level"], stats["daily_change"])
    m2.metric("ระดับตลิ่ง (ม.รทก.)", stats["bank_level"],
              f'{"ต่ำกว่า" if (stats["freeboard"] or 0) >= 0 else "สูงกว่า"}ตลิ่ง {abs(stats["freeboard"] or 0)} ม.',
              delta_color="off")
    m3.metric("% ความจุลำน้ำ", f'{stats["percent"]}%' if stats["percent"] is not None else "-",
              f'น้ำไหลผ่าน {stats["discharge"]:,.0f} ลบ.ม./วิ' if stats["discharge"] else None, delta_color="off")
    if not fc.empty:
        f3 = fc.iloc[-1]
        m4.metric(f"พยากรณ์ {int(f3['horizon'])} วัน (ML)", f"{f3['forecast']:.2f}",
                  f"{f3['forecast'] - stats['water_level']:+.2f} ม.")


def level_chart(d: pd.DataFrame, with_fc: bool = False, show_ma: bool = True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["water_level"], name="ระดับน้ำ", line=dict(color="#11698e", width=3)))
    if show_ma:
        fig.add_trace(go.Scatter(x=d["date"], y=d["MA7"], name="MA7", line=dict(color="#f39c12", dash="dot")))
        fig.add_trace(go.Scatter(x=d["date"], y=d["MA30"], name="MA30", line=dict(color="#8e44ad", dash="dot")))
    if d["bank_level"].notna().any():
        fig.add_hline(y=d["bank_level"].dropna().iloc[-1], line_color="#d62728", line_dash="dash",
                      annotation_text="ระดับตลิ่ง", annotation_position="top left")
    if with_fc and not fc.empty:
        x0 = [d["date"].iloc[-1]] + list(fc["date"])
        y0 = [d["water_level"].iloc[-1]]
        fig.add_trace(go.Scatter(x=x0 + x0[::-1], y=y0 + list(fc["upper"]) + (list(fc["lower"]) + y0)[::-1],
                                 fill="toself", fillcolor="rgba(39,174,96,.18)", line=dict(width=0),
                                 name="ช่วงความไม่แน่นอน", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x0, y=y0 + list(fc["forecast"]), name="พยากรณ์ (ML)",
                                 line=dict(color="#27ae60", width=3, dash="dash"), mode="lines+markers"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title="ระดับน้ำ (ม.รทก.)", legend=dict(orientation="h", y=1.1))
    return fig


feat = add_features(st_hist)
feat_p = feat[feat["date"] >= feat["date"].max() - pd.Timedelta(days=period)]

with tab_over:
    st.plotly_chart(level_chart(feat_p, with_fc=True, show_ma=False), width="stretch", key="chart_overview")

# ---------------------------------------------------------------- Data
with tab_data:
    st.subheader(f"ข้อมูลรายวัน · {station} {info['name']}")
    names = {"date": "วันที่", "water_level": "ระดับน้ำ (ม.รทก.)", "bank_level": "ระดับตลิ่ง",
             "freeboard": "ห่างจากตลิ่ง (ม.)", "percent": "% ความจุลำน้ำ", "change": "เปลี่ยนแปลง (ม./วัน)",
             "discharge": "ปริมาณน้ำ (ลบ.ม./วิ)", "qmax": "ความจุลำน้ำ (ลบ.ม./วิ)"}
    cols_show = [c for c in names if c in feat_p and (feat_p[c].notna().any() or c == "date")]
    show = feat_p[cols_show].copy()
    show["date"] = show["date"].dt.date
    show = show.rename(columns=names).round(2)
    st.dataframe(show.iloc[::-1], width="stretch", hide_index=True)
    st.download_button("ดาวน์โหลด CSV", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"water_{station}.csv", mime="text/csv")

# ---------------------------------------------------------------- Statistics
with tab_stats:
    st.subheader(f"สถิติ {period} วันล่าสุด")
    a, b, c, d = st.columns(4)
    a.metric("MA7", stats["MA7"])
    b.metric("MA30", stats["MA30"])
    c.metric("Volatility (SD ของการเปลี่ยนแปลงรายวัน)", stats["volatility"])
    d.metric("เปลี่ยนแปลงในช่วงนี้ (ม.)", stats["change_period"])
    a.metric("สูงสุดในช่วง", stats["period_high"])
    b.metric("ต่ำสุดในช่วง", stats["period_low"])
    c.metric("แนวโน้ม 7 วัน (ม.)", stats["trend_7d"])
    d.metric("% ความจุลำน้ำ", stats["percent"])
    st.plotly_chart(level_chart(feat_p), width="stretch", key="chart_stats_level")

    bar = go.Figure(go.Bar(x=feat_p["date"], y=feat_p["change"],
                           marker_color=["#d62728" if v > 0 else "#2c7fb8" for v in feat_p["change"].fillna(0)]))
    bar.update_layout(height=260, title="การเปลี่ยนแปลงระดับน้ำรายวัน (ม.)", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(bar, width="stretch", key="chart_stats_change")

# ---------------------------------------------------------------- Forecast
with tab_fc:
    st.subheader("พยากรณ์ระดับน้ำล่วงหน้า 1–3 วัน ด้วย Machine Learning")
    if fc.empty:
        st.info("ข้อมูลของสถานีนี้ยังไม่พอสำหรับพยากรณ์")
    else:
        t = fc.copy()
        bank = stats["bank_level"]

        def fc_status(v):
            if not bank:
                return "-"
            fb = bank - v
            return f"🔴 สูงกว่าตลิ่ง {-fb:.2f} ม." if fb < 0 else f"ต่ำกว่าตลิ่ง {fb:.2f} ม."

        t["เทียบตลิ่ง"] = [fc_status(v) for v in t["forecast"]]
        t = t.assign(date=t["date"].dt.date)[["date", "horizon", "forecast", "lower", "upper", "เทียบตลิ่ง", "model"]]
        t.columns = ["วันที่", "ล่วงหน้า (วัน)", "พยากรณ์ (ม.รทก.)", "ต่ำสุดที่คาด", "สูงสุดที่คาด",
                     "เทียบตลิ่ง", "โมเดลที่ใช้"]
        st.dataframe(t.round(2), width="stretch", hide_index=True)
        st.caption("ช่วงต่ำสุด–สูงสุดที่คาด มาจากความคลาดเคลื่อนของโมเดลบนชุดทดสอบ (เปอร์เซ็นไทล์ที่ 10–90)")
        recent = feat[feat["date"] >= feat["date"].max() - pd.Timedelta(days=30)]
        st.plotly_chart(level_chart(recent, with_fc=True, show_ma=False), width="stretch", key="chart_forecast")

    m = pd.DataFrame(bundle["metrics"])
    m = m[m["station"] == station]
    if not m.empty:
        st.markdown("#### เปรียบเทียบโมเดล: MAE บนชุดทดสอบ 20% ล่าสุด (เมตร, ยิ่งต่ำยิ่งดี)")
        mae_cols = [c for c in m.columns if c.endswith("|MAE")]
        mt = m[["horizon"] + mae_cols + ["selected", "R2", "n_train", "n_test"]].copy()
        mt.columns = (["ล่วงหน้า (วัน)"] + [c.replace("|MAE", "") for c in mae_cols]
                      + ["โมเดลที่เลือก", "R² (โมเดลที่เลือก)", "จำนวน train", "จำนวน test"])
        st.dataframe(mt.round(3), width="stretch", hide_index=True)
        st.caption("Persistence = สมมติว่าระดับน้ำวันข้างหน้าเท่ากับวันนี้ (ใช้เป็นเกณฑ์เทียบ) · "
                   "เลือกโมเดลด้วย Time-Series Cross-Validation บนชุด Train · "
                   f"โมเดลเทรนด้วยข้อมูลถึง {bundle['trained_until']}")
    imp = bundle["importance"].get(station)
    if imp is not None:
        top = imp.head(10)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=[label_feature(f) for f in top.index], orientation="h",
                               marker_color="#11698e"))
        fig.update_layout(height=380, title="ปัจจัยที่มีผลต่อการพยากรณ์ 1 วันข้างหน้า (Random Forest Feature Importance)",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch", key="chart_importance")

# ---------------------------------------------------------------- AI Insights
with tab_ai:
    st.subheader("🤖 AI อธิบายสถานการณ์น้ำ (Gemini)")
    all_now = {s: {k: v for k, v in calculate_statistics(history[history["station"] == s], 7).items()
                   if k in ("date", "water_level", "bank_level", "freeboard", "percent", "daily_change", "trend_7d", "risk")}
               for s in available}
    context = {
        "station": station, "station_name": info["name"], "province": info["province"],
        "period_days": period, "statistics": stats,
        "forecast_ml": fc.assign(date=fc["date"].astype(str), base_date=fc["base_date"].astype(str)).round(2)
        .to_dict("records") if not fc.empty else [],
        "all_stations_latest_upstream_to_downstream": all_now,
        "risk_rule": "เกณฑ์ของ สสน. จาก % ความจุลำน้ำ: <10 น้ำน้อยวิกฤต, 10-30 น้ำน้อย, 30-80 น้ำปกติ, "
                     "80-100 น้ำมาก, >100 หรือระดับน้ำสูงกว่าตลิ่ง = น้ำล้นตลิ่ง; freeboard = ระดับตลิ่ง - ระดับน้ำ (ม.)",
    }
    if "q" not in st.session_state:
        st.session_state.q = ""
    bcols = st.columns(len(PRESET_PROMPTS))
    for col, (lbl, ptxt) in zip(bcols, PRESET_PROMPTS.items()):
        if col.button(lbl, width="stretch"):
            st.session_state.q = ptxt
    q = st.text_area("คำถาม", key="q", height=90)
    if st.button("Analyze", type="primary"):
        with st.spinner("AI กำลังวิเคราะห์ ..."):
            st.session_state.answer = ask_water_ai(q or "สรุปสถานการณ์น้ำ", context)
    if st.session_state.get("answer"):
        st.markdown(st.session_state.answer)

st.caption("ข้อมูล: คลังข้อมูลน้ำแห่งชาติ สถาบันสารสนเทศทรัพยากรน้ำ (สสน.) · "
           "ระบบนี้จัดทำเพื่อการศึกษา ไม่ใช่ประกาศเตือนภัยทางราชการ")