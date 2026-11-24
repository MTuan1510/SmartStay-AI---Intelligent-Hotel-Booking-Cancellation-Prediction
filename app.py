"""SmartStay AI — demo dự đoán hủy đặt phòng và hỗ trợ quyết định overbooking.

Chạy:  python -m streamlit run app.py     (từ thư mục gốc repo)

Quy ước theo reports/final_protocol.md (giữ nguyên, chỉ đổi cách trình bày):
  - dùng đúng file mô hình và ngưỡng đã khóa (frozen_thresholds.csv); app KHÔNG cho chỉnh ngưỡng;
  - không hiển thị xác suất thô (mục 5): chỉ hiển thị mức nguy cơ so với ngưỡng;
  - chi phí là giả định của nhóm (mục 6);
  - app KHÔNG đọc test.csv: số liệu test chỉ lấy từ các file kết quả đã lưu (reports/results/test_final*.csv).

Ghi chú: bảng "độ nhạy theo r" (test_final_sensitivity.csv) tạm không hiển thị vì cột `cost` trong file đó
được tính với c_walk = 2 cho mọi dòng r (xem notebooks/07_final_test.ipynb, ô tạo `sens`). Hiển thị lại
sau khi sinh lại file với c_walk = r.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.cost import baseline_costs, cost_at_threshold, cost_curve
from src.data import load_xy

RESULTS = ROOT / "reports" / "results"
MODELS_DIR = ROOT / "models"
MODELS = ["Tung_mlp_v2", "lightgbm_smotenc", "logreg", "tree"]
MODEL_LABEL = {"Tung_mlp_v2": "MLP v2 (mô hình chính)", "lightgbm_smotenc": "LightGBM + SMOTENC",
               "logreg": "Logistic Regression", "tree": "Decision Tree", "dummy": "Không dùng mô hình"}
RATIOS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
MAIN = "Tung_mlp_v2"

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DEPOSITS = ["No Deposit", "Non Refund", "Refundable"]
DEPOSIT_LABEL = {"No Deposit": "Không đặt cọc", "Non Refund": "Cọc không hoàn lại", "Refundable": "Cọc hoàn lại được"}
SEGMENT_LABEL = {"Online TA": "Đại lý du lịch online", "Offline TA/TO": "Đại lý / tour truyền thống",
                 "Direct": "Đặt trực tiếp", "Groups": "Đoàn", "Corporate": "Doanh nghiệp",
                 "Complementary": "Phòng tặng", "Aviation": "Hàng không"}
CUSTOMER_LABEL = {"Transient": "Khách lẻ", "Transient-Party": "Khách lẻ đi theo nhóm",
                  "Contract": "Theo hợp đồng", "Group": "Đoàn"}

# Bảng màu: mực đậm, xanh biển (màu chính), gạch (bất lợi), lục (có lợi), xám
INK, TEAL, MIST, BRICK, GREEN, GRAY = "#14262B", "#0B6E75", "#E8EFF1", "#C2453A", "#2E7D5B", "#7A8A8F"

st.set_page_config(page_title="SmartStay AI", page_icon="🏨", layout="wide")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
html, body, .stApp, p, li, label, h1, h2, h3, h4, button, input, textarea,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {{
    font-family: 'Be Vietnam Pro', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
.block-container {{ max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem; }}
h1 {{ font-size: 2.1rem !important; font-weight: 700 !important; letter-spacing: -0.02em; line-height: 1.2 !important; }}
h2, h3 {{ font-weight: 600 !important; letter-spacing: -0.01em; }}
.lede {{ font-size: 1.12rem; line-height: 1.65; max-width: 46rem; color: {INK}; }}
.verdict {{ border-left: 6px solid {GRAY}; background: #fff; padding: 1.2rem 1.5rem; margin: 1rem 0 1.4rem 0; border-radius: 2px; }}
.verdict.warn {{ border-left-color: {BRICK}; }}
.verdict.good {{ border-left-color: {GREEN}; }}
.verdict .v-head {{ font-size: 1.55rem; font-weight: 600; line-height: 1.35; letter-spacing: -0.01em; margin: 0 0 .6rem 0; }}
.verdict .v-body {{ font-size: 1rem; line-height: 1.65; margin: 0; max-width: 48rem; }}
.step b {{ font-size: 1.02rem; }}
.step p {{ margin: .25rem 0 0 0; line-height: 1.55; }}
.risk {{ padding: 1.1rem 1.4rem; border-radius: 2px; background: #fff; border-left: 6px solid {GRAY}; margin-bottom: 1rem; }}
.risk.high {{ border-left-color: {BRICK}; }}
.risk.low {{ border-left-color: {GREEN}; }}
.risk .r-head {{ font-size: 1.45rem; font-weight: 700; margin: 0 0 .25rem 0; }}
.risk.high .r-head {{ color: {BRICK}; }}
.risk.low .r-head {{ color: {GREEN}; }}
.risk .r-sub {{ margin: 0; line-height: 1.55; }}
.rankbar {{ position: relative; display: flex; height: 2.1rem; margin: 1.1rem 0 .3rem 0; font-size: .78rem; font-weight: 500; }}
.rankbar .rb-low, .rankbar .rb-high {{ display: flex; align-items: center; justify-content: center; overflow: hidden; white-space: nowrap; }}
.rankbar .rb-low {{ background: #D6EBE2; color: {GREEN}; }}
.rankbar .rb-high {{ background: #F4D9D5; color: {BRICK}; }}
.rankbar .rb-me {{ position: absolute; top: -7px; bottom: -7px; width: 4px; background: {INK}; transform: translateX(-2px); }}
.rb-note {{ font-size: .82rem; color: {GRAY}; margin-bottom: 1rem; }}
.chips {{ display: flex; gap: .6rem; flex-wrap: wrap; margin: .4rem 0 .6rem 0; }}
.chip {{ padding: .5rem .8rem; background: #fff; border: 1px solid #D5DFE2; font-size: .9rem; border-radius: 2px; }}
.chip.on {{ border: 2px solid {INK}; }}
.chip b.hi {{ color: {BRICK}; }}
.chip b.lo {{ color: {GREEN}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- định dạng số theo kiểu Việt Nam (dấu chấm ngăn nghìn, dấu phẩy thập phân) ----------
def fmt_int(n) -> str:
    return f"{n:,.0f}".replace(",", ".")


def fmt_pct(x, d: int = 1) -> str:
    """x đã ở dạng phần trăm (19.5 -> '19,5%')."""
    return "—" if x != x else f"{x:.{d}f}%".replace(".", ",")


def fmt_dec(x, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


# ---------- đọc dữ liệu (có cache) ----------
@st.cache_data
def frozen() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "frozen_thresholds.csv")


@st.cache_data
def val_data():
    X, y = load_xy("val")  # CHỈ val
    return X, y.to_numpy()


@st.cache_data
def train_defaults() -> pd.Series:
    X, _ = load_xy("train")
    d = {}
    for c in X.columns:
        d[c] = X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0]
    return pd.Series(d)


@st.cache_data
def val_proba(model: str) -> np.ndarray:
    return np.load(MODELS_DIR / f"proba_val_{model}.npy")


@st.cache_resource
def load_model(model: str):
    return joblib.load(MODELS_DIR / f"{model}.joblib")


def locked_threshold(model: str, r: float) -> float:
    f = frozen()
    return float(f[(f.model == model) & (f.r == r)]["threshold"].iloc[0])


def read_result(name: str):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


COST_NOTE = ("Chi phí là **giả định của nhóm**, không phải số liệu thật của khách sạn: "
             "một khách bị đuổi (walk) tốn r đơn vị, một phòng trống do khách hủy tốn 1 đơn vị. "
             "Mỗi đơn bị dự đoán hủy thì overbook đúng một phòng.")


def verdict(tone: str, head: str, body: str):
    st.markdown(f'<div class="verdict {tone}"><p class="v-head">{head}</p><p class="v-body">{body}</p></div>',
                unsafe_allow_html=True)


def chart_style(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(labelColor=INK, titleColor=INK, gridColor="#DCE5E7", domainColor="#B9C7CB")
            .configure_legend(labelColor=INK))


# ---------- trang 0: giới thiệu ----------
def page_intro():
    st.title("Về dự án SmartStay AI")
    st.markdown('<p class="lede">Khoảng một phần ba đơn đặt phòng bị hủy, để lại phòng trống. '
                'SmartStay AI thử xem mô hình học máy có giúp khách sạn bán thêm phòng '
                '(<b>overbooking</b>) đúng lúc để bù lại không, và khi nào việc đó có lợi.</p>',
                unsafe_allow_html=True)

    st.subheader("Cách hoạt động")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><b>1. Chấm điểm đơn đặt</b><p>Mô hình xem thông tin lúc khách đặt phòng '
                '(đặt trước bao lâu, loại đặt cọc, nguồn đặt...) và cho điểm nguy cơ hủy.</p></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="step"><b>2. So với ngưỡng</b><p>Nếu điểm vượt <b>ngưỡng</b> đã chọn, '
                'khách sạn bán thêm đúng một phòng cho đơn đó.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="step"><b>3. Tính chi phí</b><p>Khách hủy thật thì phòng được lấp, có lợi. '
                'Khách vẫn đến thì phải chuyển khách đi nơi khác, tốn kém hơn phòng trống.</p></div>',
                unsafe_allow_html=True)

    st.subheader("Trong bản demo này")
    st.markdown("- **Dự đoán**: nhập một đơn đặt phòng và xem mô hình đánh giá nguy cơ hủy ra sao.\n"
                "- **Kết quả**: kết luận cuối cùng trên dữ liệu mô hình chưa từng thấy.\n"
                "- **Thử mức chi phí**: kéo thử các mức chi phí giả định khác nhau và xem ngưỡng ảnh hưởng thế nào.\n"
                "- **Giới hạn**: mô hình có thể sai ở đâu.")

    st.subheader("Dữ liệu")
    st.markdown("Bộ dữ liệu *Hotel Booking Demand*: 86.617 đơn sau làm sạch, tỷ lệ hủy 27,7%. "
                "Dữ liệu chia theo thời gian: huấn luyện (07/2015–08/2016), kiểm định (09–12/2016), "
                "kiểm tra (01–08/2017). Tập kiểm tra chỉ được mở một lần ở cuối.")
    st.caption(COST_NOTE)

    with st.expander("Giải thích các từ khó"):
        st.markdown(
            "- **Overbooking**: bán nhiều phòng hơn số phòng thật, vì dự đoán sẽ có khách hủy.\n"
            "- **Ngưỡng**: điểm nguy cơ tối thiểu để mô hình khuyên overbook. Ngưỡng được chọn trên tập kiểm định "
            "rồi khóa lại, không chỉnh sau khi xem kết quả.\n"
            "- **r (tỷ lệ chi phí)**: một khách bị đuổi tốn gấp r lần một phòng trống. r càng lớn thì càng nên thận trọng.\n"
            "- **Bắt được đơn hủy** (recall): trong tất cả đơn hủy thật, mô hình phát hiện được bao nhiêu phần trăm.\n"
            "- **Đoán hủy thì đúng** (precision): trong các đơn mô hình đoán sẽ hủy, bao nhiêu phần trăm hủy thật.\n"
            "- **Tập kiểm định / tập kiểm tra**: kiểm định dùng để chọn ngưỡng nên kết quả lạc quan hơn; "
            "kiểm tra là dữ liệu mô hình chưa từng thấy, nên đáng tin hơn.")


# ---------- trang 1: kết quả cuối trên test ----------
def page_results():
    st.title("Kết quả trên dữ liệu chưa từng thấy")
    st.caption("Đọc từ file kết quả của lần chạy duy nhất trên tập kiểm tra, ở mức r = 2 "
               "(đuổi nhầm một khách tốn gấp đôi một phòng trống).")

    tf = read_result("test_final.csv")
    need = {"model", "n", "positive_rate", "recall", "precision", "fp", "tp",
            "cost_r2_locked", "cost_never", "saving_vs_never_pct"}
    if tf is None or not need.issubset(tf.columns):
        st.info("Chưa có đủ dữ liệu trong `reports/results/test_final.csv`.")
        return

    mlp = tf[tf.model == MAIN].iloc[0]
    never = float(mlp["cost_never"])
    others = tf[~tf.model.isin(["dummy", MAIN])].copy()
    winners = others[others.saving_vs_never_pct > 0].sort_values("saving_vs_never_pct", ascending=False)
    mlp_overbooked = int(mlp["tp"] + mlp["fp"])

    s = float(mlp["saving_vs_never_pct"])
    if s < 0:
        head = f"Với r = 2, dùng MLP v2 để overbook tốn hơn {fmt_pct(abs(s))} so với không overbook."
        tone = "warn"
    else:
        head = f"Với r = 2, dùng MLP v2 để overbook tiết kiệm {fmt_pct(s)} so với không overbook."
        tone = "good"
    body = (f"Mô hình bắt được {fmt_pct(mlp['recall'] * 100, 0)} đơn hủy, nhưng cũng overbook nhầm "
            f"{fmt_int(mlp['fp'])} đơn mà khách vẫn đến, và mỗi lần như vậy tốn gấp đôi một phòng trống.")
    if len(winners):
        counts = [int(r_.tp + r_.fp) for r_ in winners.itertuples()]
        lo, hi = winners.saving_vs_never_pct.min(), winners.saving_vs_never_pct.max()
        body += (f" Các mô hình thận trọng hơn chỉ tiết kiệm được {fmt_pct(lo)}–{fmt_pct(hi)}, "
                 f"vì chỉ overbook {fmt_int(min(counts))}–{fmt_int(max(counts))} đơn thay vì {fmt_int(mlp_overbooked)}.")
    else:
        body += " Không mô hình nào tiết kiệm được so với việc không overbook."
    verdict(tone, head, body)

    k1, k2, k3 = st.columns(3)
    k1.metric("Đơn hủy mô hình bắt được", fmt_pct(mlp["recall"] * 100, 0),
              help="Recall của MLP v2 ở ngưỡng đã khóa.")
    k2.metric("Đoán hủy thì đúng", fmt_pct(mlp["precision"] * 100, 0),
              help="Precision của MLP v2 ở ngưỡng đã khóa.")
    k3.metric("Chi phí so với không overbook", f"{'+' if s < 0 else '−'}{fmt_pct(abs(s))}",
              help="Dấu + nghĩa là tốn hơn so với việc không overbook.")

    st.subheader("Tổng chi phí của từng mô hình")
    st.caption("Thanh càng ngắn càng tốt. Vạch đứt là chi phí nếu không overbook phòng nào.")
    cdf = tf[tf.model != "dummy"].copy()
    cdf["name"] = cdf["model"].map(lambda m: MODEL_LABEL.get(m, m))
    cdf["cost"] = cdf["cost_r2_locked"].astype(float)
    cdf["label"] = cdf["saving_vs_never_pct"].map(
        lambda v: f"tiết kiệm {fmt_pct(v)}" if v > 0 else f"tốn hơn {fmt_pct(abs(v))}")
    bars = alt.Chart(cdf).mark_bar().encode(
        y=alt.Y("name:N", sort="x", title=None),
        x=alt.X("cost:Q", title="Tổng chi phí (đơn vị giả định)"),
        color=alt.condition(alt.datum.cost <= never, alt.value(GREEN), alt.value(BRICK)),
        tooltip=[alt.Tooltip("name:N", title="Mô hình"), alt.Tooltip("cost:Q", title="Chi phí", format=",.0f"),
                 alt.Tooltip("label:N", title="So với không overbook")])
    text = alt.Chart(cdf).mark_text(align="left", dx=5, color=INK).encode(
        y=alt.Y("name:N", sort="x"), x="cost:Q", text="label:N")
    rule = alt.Chart(pd.DataFrame({"x": [never], "t": [f"Không overbook: {fmt_int(never)}"]})).mark_rule(
        color=INK, strokeDash=[5, 4]).encode(x="x:Q")
    st.altair_chart(chart_style((bars + text + rule).properties(height=220)), width="stretch")

    st.subheader("Mô hình phát hiện đơn hủy tốt đến đâu?")
    t = tf[tf.model != "dummy"].copy()
    view = pd.DataFrame({
        "Mô hình": t["model"].map(lambda m: MODEL_LABEL.get(m, m)),
        "Bắt được đơn hủy": (t["recall"] * 100).round(1),
        "Đoán hủy thì đúng": (t["precision"] * 100).round(1),
        "Số đơn bị overbook": (t["tp"] + t["fp"]).astype(int),
    })
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Bắt được đơn hủy": st.column_config.ProgressColumn(
            "Bắt được đơn hủy", format="%.0f%%", min_value=0, max_value=100,
            help="Trong mọi đơn hủy thật, mô hình phát hiện bao nhiêu phần trăm."),
        "Đoán hủy thì đúng": st.column_config.ProgressColumn(
            "Đoán hủy thì đúng", format="%.0f%%", min_value=0, max_value=100,
            help="Trong các đơn mô hình đoán sẽ hủy, bao nhiêu phần trăm hủy thật."),
        "Số đơn bị overbook": st.column_config.NumberColumn("Số đơn bị overbook", format="%d"),
    })
    st.caption(f"Tập kiểm tra có {fmt_int(mlp['n'])} đơn, tỷ lệ hủy thật {fmt_pct(mlp['positive_rate'] * 100)}. "
               "Chi phí test chỉ có ở mức r = 2 cho mọi mô hình.")

    with st.expander("Chi tiết kỹ thuật"):
        tech = tf.copy()
        tech["model"] = tech["model"].map(lambda m: MODEL_LABEL.get(m, m))
        cols = {"model": "Mô hình", "threshold": "Ngưỡng", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1",
                "mean_proba": "Xác suất TB mô hình dự đoán"}
        cols = {k: v for k, v in cols.items() if k in tech.columns}
        st.dataframe(tech[list(cols)].rename(columns=cols).round(4), hide_index=True, width="stretch")
        st.caption("Nếu xác suất trung bình thấp hơn tỷ lệ hủy thật, xác suất thô của mô hình chưa được hiệu chỉnh, "
                   "nên app chỉ hiển thị mức nguy cơ so với ngưỡng, không hiển thị xác suất.")


# ---------- trang 2: thử các mức chi phí trên val ----------
def page_cost_explorer():
    st.title("Thử các mức chi phí khác nhau")
    st.markdown('<p class="lede">Chọn mô hình và mức chi phí giả định, rồi xem tổng chi phí thay đổi ra sao. '
                'Ngưỡng ở mỗi mức đã được chọn và khóa từ trước, trang này không cho chỉnh lại.</p>',
                unsafe_allow_html=True)
    st.caption(COST_NOTE)

    c1, c2 = st.columns([2, 1])
    model = c1.selectbox("Mô hình", MODELS, format_func=MODEL_LABEL.get)
    r = c2.select_slider("Đuổi nhầm 1 khách tốn gấp mấy lần 1 phòng trống?", options=RATIOS, value=2.0,
                         format_func=lambda v: f"{v:g} lần")
    t = locked_threshold(model, r)

    X, y = val_data()
    proba = val_proba(model)
    hotel = st.radio("Khách sạn", ["Tất cả", "City Hotel", "Resort Hotel"], horizontal=True)
    mask = np.ones(len(y), bool) if hotel == "Tất cả" else (X["hotel"] == hotel).to_numpy()
    yv, pv = y[mask], proba[mask]

    res = cost_at_threshold(yv, pv, t, r, 1.0)
    never = baseline_costs(yv, r, 1.0)["never_overbook"]
    saving = (never - res["cost"]) / never * 100 if never else float("nan")

    st.subheader("Trên tập kiểm định")
    st.caption("Tập này đã được dùng để chọn ngưỡng nên kết quả lạc quan hơn kết quả ở trang Kết quả mô hình.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ngưỡng đã khóa", fmt_dec(t), help="Điểm nguy cơ tối thiểu để mô hình khuyên overbook.")
    m2.metric("Chi phí khi dùng mô hình", fmt_int(res["cost"]))
    m3.metric("Chi phí nếu không overbook", fmt_int(never))
    m4.metric("Tiết kiệm", fmt_pct(saving))
    st.caption(f"{fmt_int(len(yv))} đơn. Overbook nhầm (khách vẫn đến): {fmt_int(res['fp'])}. "
               f"Bỏ lỡ (khách hủy mà không overbook): {fmt_int(res['fn'])}.")

    curve = cost_curve(yv, pv, r, 1.0)
    line = alt.Chart(curve).mark_line(color=TEAL, strokeWidth=2.5).encode(
        x=alt.X("threshold:Q", title="Ngưỡng"),
        y=alt.Y("cost:Q", title="Tổng chi phí (đơn vị giả định)"),
        tooltip=[alt.Tooltip("threshold:Q", title="Ngưỡng"), alt.Tooltip("cost:Q", title="Chi phí", format=",.0f")])
    rule = alt.Chart(pd.DataFrame({"t": [t], "label": [f"Ngưỡng đã khóa {fmt_dec(t)}"]})).mark_rule(
        color=BRICK, strokeDash=[5, 4]).encode(x="t:Q")
    rule_txt = alt.Chart(pd.DataFrame({"t": [t], "label": [f"Ngưỡng đã khóa {fmt_dec(t)}"]})).mark_text(
        align="left", dx=6, dy=10, color=BRICK).encode(x="t:Q", y=alt.value(0), text="label:N")
    never_df = pd.DataFrame({"y": [never], "label": ["Không overbook"]})
    never_rule = alt.Chart(never_df).mark_rule(color=GRAY).encode(y="y:Q")
    never_txt = alt.Chart(never_df).mark_text(align="left", dx=6, dy=-8, color=GRAY).encode(
        y="y:Q", x=alt.value(0), text="label:N")
    st.altair_chart(chart_style((line + rule + rule_txt + never_rule + never_txt).properties(
        height=320, title=f"Tổng chi phí theo ngưỡng (đuổi nhầm tốn gấp {r:g} lần phòng trống)")), width="stretch")
    st.caption("Đường cong chỉ để tham khảo, không dùng để chọn lại ngưỡng.")


# ---------- trang 3: thử một đơn đặt phòng ----------
def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def init_form(d: pd.Series, segs, custs, countries):
    defaults = {
        "f_hotel": "City Hotel", "f_month": "July", "f_deposit": "No Deposit", "f_lead": 41,
        "f_wkend": _clip(int(d["stays_in_weekend_nights"]), 0, 20),
        "f_wk": _clip(int(d["stays_in_week_nights"]), 0, 50),
        "f_adults": _clip(int(d["adults"]), 0, 10),
        "f_children": _clip(int(d["children"]), 0, 3),
        "f_adr": float(_clip(float(d["adr"]), 0.0, 520.0)),
        "f_prev": _clip(int(d["previous_cancellations"]), 0, 30),
        "f_seg": "Online TA" if "Online TA" in segs else segs[0],
        "f_cust": "Transient" if "Transient" in custs else custs[0],
        "f_country": countries[0], "f_rep": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def apply_preset(values: dict, valid: dict):
    for k, v in values.items():
        if k in valid and v not in valid[k]:
            continue
        st.session_state[k] = v


PRESETS = {
    "Khách lẻ đặt sớm": {"f_hotel": "City Hotel", "f_month": "July", "f_deposit": "No Deposit", "f_lead": 200,
                         "f_seg": "Online TA", "f_cust": "Transient", "f_prev": 0, "f_rep": False},
    "Đặt sát ngày, khách quen": {"f_deposit": "No Deposit", "f_lead": 3, "f_seg": "Direct",
                                 "f_prev": 0, "f_rep": True},
    "Cọc không hoàn lại": {"f_deposit": "Non Refund", "f_lead": 120, "f_seg": "Offline TA/TO"},
}


def rank_bar(pct: float, pct_thr: float) -> str:
    return (f'<div class="rankbar"><div class="rb-low" style="width:{pct_thr:.1f}%">Giữ phòng</div>'
            f'<div class="rb-high" style="width:{100 - pct_thr:.1f}%">Cân nhắc overbook</div>'
            f'<div class="rb-me" style="left:{pct:.1f}%"></div></div>'
            '<div class="rb-note">Vạch đen là vị trí của đơn này khi xếp 100 đơn trong tập kiểm định '
            'từ ít đến nhiều nguy cơ hủy.</div>')


def page_single():
    st.title("Dự đoán hủy đặt phòng")
    st.markdown('<p class="lede">Nhập thông tin một đơn đặt phòng, mô hình sẽ cho biết đơn này có nguy cơ '
                'bị hủy <b>cao</b> hay <b>thấp</b>. Chọn một kịch bản mẫu để bắt đầu nhanh, rồi chỉnh từng thông tin '
                'để xem kết quả thay đổi.</p>', unsafe_allow_html=True)

    X_val, _ = val_data()
    d = train_defaults()
    segs = sorted(X_val["market_segment"].unique())
    custs = sorted(X_val["customer_type"].unique())
    countries = X_val["country"].value_counts().head(20).index.tolist()
    init_form(d, segs, custs, countries)
    valid = {"f_seg": segs, "f_cust": custs, "f_country": countries}

    st.markdown("**Kịch bản mẫu**")
    pcols = st.columns(len(PRESETS))
    for i, (name, vals) in enumerate(PRESETS.items()):
        pcols[i].button(name, key=f"preset_{i}", on_click=apply_preset, args=(vals, valid))

    with st.expander("Cài đặt nâng cao: mức chi phí giả định"):
        r = st.select_slider("Đuổi nhầm 1 khách tốn gấp mấy lần 1 phòng trống?",
                             options=RATIOS, value=2.0, format_func=lambda v: f"{v:g} lần")
        st.caption("Mức này quyết định ngưỡng nguy cơ. Mặc định 2 lần, là mức dùng trong kết quả cuối của nhóm.")
    t = locked_threshold(MAIN, r)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Về đơn đặt**")
        a = st.columns(3)
        hotel = a[0].selectbox("Khách sạn", ["City Hotel", "Resort Hotel"], key="f_hotel")
        month = a[1].selectbox("Tháng đến", MONTHS, key="f_month", format_func=lambda m: f"Tháng {MONTHS.index(m) + 1}")
        deposit = a[2].selectbox("Loại đặt cọc", DEPOSITS, key="f_deposit", format_func=DEPOSIT_LABEL.get)
        b = st.columns(3)
        lead = b[0].number_input("Đặt trước (ngày)", 0, 710, key="f_lead",
                                 help="Số ngày từ lúc đặt đến lúc nhận phòng (lead time).")
        wk_end = b[1].number_input("Đêm cuối tuần", 0, 20, key="f_wkend")
        wk = b[2].number_input("Đêm trong tuần", 0, 50, key="f_wk")
        c = st.columns(3)
        adr = c[0].number_input("Giá phòng TB mỗi đêm", 0.0, 520.0, key="f_adr",
                                help="Average daily rate (adr), giá trung bình mỗi đêm.")
        seg = c[1].selectbox("Nguồn đặt phòng", segs, key="f_seg", format_func=lambda v: SEGMENT_LABEL.get(v, v))
        cust = c[2].selectbox("Loại khách", custs, key="f_cust", format_func=lambda v: CUSTOMER_LABEL.get(v, v))

        st.markdown("**Về khách**")
        e = st.columns(4)
        adults = e[0].number_input("Người lớn", 0, 10, key="f_adults")
        children = e[1].number_input("Trẻ em", 0, 3, key="f_children")
        prev_c = e[2].number_input("Số lần hủy trước", 0, 30, key="f_prev")
        country = e[3].selectbox("Quốc gia", countries, key="f_country")
        repeated = st.checkbox("Khách đã từng ở lại", key="f_rep")

    base = d.to_dict()
    base.update({"hotel": hotel, "arrival_date_month": month, "deposit_type": deposit, "lead_time": lead,
                 "stays_in_weekend_nights": wk_end, "stays_in_week_nights": wk, "adults": adults,
                 "children": children, "adr": adr, "previous_cancellations": prev_c,
                 "market_segment": seg, "customer_type": cust, "country": country,
                 "is_repeated_guest": int(repeated)})
    rows = [{**base, "deposit_type": dep} for dep in DEPOSITS]
    X3 = pd.DataFrame(rows)[X_val.columns].astype(X_val.dtypes.to_dict())
    scores = load_model(MAIN).predict_proba(X3)[:, 1]
    cur = DEPOSITS.index(deposit)
    score = float(scores[cur])

    vp = val_proba(MAIN)
    pct = float((vp < score).mean() * 100)
    pct_thr = float((vp < t).mean() * 100)
    high = score >= t

    with right:
        st.markdown("**Kết quả**")
        if high:
            st.markdown(f'<div class="risk high"><p class="r-head">Nguy cơ hủy: CAO</p>'
                        f'<p class="r-sub">Vượt ngưỡng đã khóa ({fmt_dec(t)}). Có thể cân nhắc overbook phòng này.</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk low"><p class="r-head">Nguy cơ hủy: THẤP</p>'
                        f'<p class="r-sub">Dưới ngưỡng đã khóa ({fmt_dec(t)}). Nên giữ phòng cho khách này.</p></div>',
                        unsafe_allow_html=True)
        st.markdown(rank_bar(pct, pct_thr), unsafe_allow_html=True)
        st.caption(f"Điểm của đơn này cao hơn khoảng {pct:.0f}% số đơn trong tập kiểm định. "
                   "Đây là mức nguy cơ so với ngưỡng đã khóa, không phải xác suất hủy đã hiệu chỉnh "
                   "(MLP thô cho xác suất thấp hơn thực tế), nên app không hiển thị xác suất.")

        st.markdown("**Nếu đổi loại đặt cọc thì sao?**")
        chips = ""
        for i, dep in enumerate(DEPOSITS):
            lvl = ('<b class="hi">Cao</b>' if scores[i] >= t else '<b class="lo">Thấp</b>')
            chips += f'<div class="chip{" on" if i == cur else ""}">{DEPOSIT_LABEL[dep]}: {lvl}</div>'
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
        st.caption("Giữ nguyên các thông tin khác. Đây là mối liên hệ mô hình đã học từ dữ liệu, "
                   "không chứng minh nguyên nhân hủy.")

    st.caption("Các cột không có trong form (agent, meal, ...) dùng giá trị phổ biến nhất của tập huấn luyện.")


# ---------- trang 4: giới hạn ----------
def page_limits():
    st.title("Mô hình có thể sai ở đâu?")
    st.caption("Mọi số liệu test dưới đây đọc từ file kết quả đã lưu của lần chạy duy nhất (protocol-v1).")

    boot = read_result("test_final_bootstrap.csv")
    if boot is not None:
        b = boot.iloc[0]
        st.subheader("MLP có thật sự hơn LightGBM không?")
        rows = []
        for label, k in [("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc")]:
            diff, lo, hi = b[f"diff_{k}"], b[f"ci_{k}_lo"], b[f"ci_{k}_hi"]
            if lo <= 0 <= hi:
                kq = "chưa đủ bằng chứng để nói mô hình nào hơn"
            elif diff > 0:
                kq = "MLP nhỉnh hơn, nhưng mức chênh rất nhỏ"
            else:
                kq = "LightGBM nhỉnh hơn, nhưng mức chênh rất nhỏ"
            rows.append(f"- **{label}**: chênh {diff:+.4f}, khoảng tin cậy 95% [{lo:+.4f}; {hi:+.4f}], {kq}.")
        st.markdown("\n".join(rows))
        st.caption("Khoảng tin cậy chứa 0 nghĩa là chênh lệch có thể chỉ do ngẫu nhiên (bootstrap ghép cặp).")

    by_hotel = read_result("test_final_by_hotel.csv")
    if by_hotel is not None and {"group", "recall", "precision"}.issubset(by_hotel.columns):
        st.subheader("Khác nhau giữa hai khách sạn")
        rec = dict(zip(by_hotel["group"], by_hotel["recall"]))
        if "City Hotel" in rec and "Resort Hotel" in rec:
            st.markdown(f"MLP bắt được **{fmt_pct(rec['City Hotel'] * 100, 0)}** đơn hủy ở City Hotel nhưng chỉ "
                        f"**{fmt_pct(rec['Resort Hotel'] * 100, 0)}** ở Resort Hotel.")
        hv = pd.DataFrame({"Khách sạn": by_hotel["group"], "Số đơn": by_hotel["n"],
                           "Bắt được đơn hủy": (by_hotel["recall"] * 100).round(1),
                           "Đoán hủy thì đúng": (by_hotel["precision"] * 100).round(1)})
        st.dataframe(hv, hide_index=True, width="stretch", column_config={
            "Bắt được đơn hủy": st.column_config.ProgressColumn("Bắt được đơn hủy", format="%.0f%%", min_value=0, max_value=100),
            "Đoán hủy thì đúng": st.column_config.ProgressColumn("Đoán hủy thì đúng", format="%.0f%%", min_value=0, max_value=100),
            "Số đơn": st.column_config.NumberColumn("Số đơn", format="%d")})

    by_month = read_result("test_final_by_month.csv")
    if by_month is not None and {"group", "recall", "precision"}.issubset(by_month.columns):
        st.subheader("Khác nhau theo tháng đến")
        long = by_month.melt(id_vars=["group"], value_vars=["recall", "precision"], var_name="metric", value_name="value")
        long["metric"] = long["metric"].map({"recall": "Bắt được đơn hủy", "precision": "Đoán hủy thì đúng"})
        long["Tháng"] = long["group"].map(lambda m: f"T{MONTHS.index(m) + 1}" if m in MONTHS else m)
        order = [f"T{i}" for i in range(1, 13)]
        ch = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("Tháng:N", sort=order, title="Tháng đến (T1 = tháng 1)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", scale=alt.Scale(range=[TEAL, BRICK]), legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("Tháng:N"), alt.Tooltip("metric:N", title="Chỉ số"),
                     alt.Tooltip("value:Q", title="Tỷ lệ", format=".0%")])
        st.altair_chart(chart_style(ch.properties(height=280)), width="stretch")

    st.subheader("Giới hạn đã biết")
    st.markdown("""
- Chi phí chỉ là giả định của nhóm (r = 2) và mô hình overbook một phòng cho mỗi đơn dự đoán hủy.
- Ở ngưỡng 0,43 MLP bắt được gần một nửa đơn hủy nhưng overbook nhầm khá nhiều, nên tổng chi phí vẫn cao hơn không overbook.
- Điểm val của MLP lạc quan vì cấu hình được chọn trên val; ngưỡng gắn với đúng một file huấn luyện.
- Xác suất MLP thô thấp hơn thực tế ở cả hai khách sạn, nên app không hiển thị xác suất.
- Dữ liệu chia theo ngày đến, không phải ngày đặt; tỷ lệ hủy tăng theo thời gian (drift).
- Dòng trùng bị xóa nhưng dữ liệu không có mã đặt phòng nên không biết đó là bản ghi lặp hay đơn thật.
- Chỉ là mối liên hệ mô hình đã học, không chứng minh nguyên nhân hủy.
""")


# ---------- điều hướng ----------
nav = st.navigation([
    st.Page(page_single, title="Dự đoán", icon=":material/hotel:", url_path="du-doan", default=True),
    st.Page(page_results, title="Kết quả mô hình", icon=":material/fact_check:", url_path="ket-qua"),
    st.Page(page_cost_explorer, title="Thử mức chi phí", icon=":material/tune:", url_path="thu-muc-chi-phi"),
    st.Page(page_limits, title="Giới hạn", icon=":material/report:", url_path="gioi-han"),
    st.Page(page_intro, title="Về dự án", icon=":material/info:", url_path="ve-du-an"),
], position="top")
nav.run()