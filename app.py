"""SmartStay AI — demo dự đoán hủy đặt phòng và hỗ trợ quyết định overbooking.

Chạy:  python -m streamlit run app.py     (từ thư mục gốc repo)

Quy ước theo reports/final_protocol.md (giữ nguyên, chỉ đổi cách trình bày):
  - protocol-v1 dùng mô hình và ngưỡng đã khóa; RF E1 bổ sung dùng ngưỡng 0,50 tham khảo;
  - không hiển thị xác suất thô (mục 5): chỉ hiển thị mức nguy cơ so với ngưỡng;
  - chi phí là giả định của nhóm (mục 6);
  - app KHÔNG đọc test.csv: số liệu test chỉ lấy từ các file kết quả đã lưu (reports/results/test_final*.csv).
  - các trang mô phỏng (Chấm hàng loạt, Một tuần vận hành) chỉ dùng tập kiểm định hoặc file người dùng tự tải lên.

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

from src.cost import baseline_costs, cost_at_threshold, cost_curve, theoretical_threshold
from src.data import load_xy

RESULTS = ROOT / "reports" / "results"
MODELS_DIR = ROOT / "models"
MODELS = ["Tung_mlp_v2", "lightgbm_smotenc", "logreg", "tree", "rf_e1"]
MODEL_LABEL = {"Tung_mlp_v2": "MLP v2 (mô hình chính)", "lightgbm_smotenc": "LightGBM + SMOTENC",
               "logreg": "Logistic Regression", "tree": "Decision Tree", "rf_e1": "Random Forest E1", "dummy": "Không dùng mô hình"}
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
    if model == "rf_e1":
        return 0.5  # Ngưỡng của E1 mới, không thuộc protocol-v1, chưa tối ưu theo r.
    f = frozen()
    return float(f[(f.model == model) & (f.r == r)]["threshold"].iloc[0])


def read_result(name: str):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None



# ---------- hàm dùng chung cho các trang phân tích thêm ----------
@st.cache_data
def val_sorted(model: str) -> np.ndarray:
    return np.sort(val_proba(model))


def rank_pct(sorted_vp: np.ndarray, scores) -> np.ndarray:
    """Vị trí xếp hạng: % đơn trong tập kiểm định có điểm THẤP hơn điểm này (0-100)."""
    sc = np.atleast_1d(np.asarray(scores, dtype=float))
    return np.searchsorted(sorted_vp, sc, side="left") / len(sorted_vp) * 100


def try_load_model(model: str):
    """Tải mô hình; trả về None nếu thiếu file (ví dụ clone repo mà chưa có file .joblib)."""
    try:
        return load_model(model)
    except Exception:
        return None


@st.cache_data
def channel_map() -> dict:
    """Kênh phân phối phổ biến nhất của từng nguồn đặt phòng (tập huấn luyện), để hai cột này luôn khớp nhau."""
    X, _ = load_xy("train")
    return X.groupby("market_segment")["distribution_channel"].agg(lambda s: s.value_counts().index[0]).to_dict()


@st.cache_data
def train_profile() -> dict:
    """Khoảng thường gặp của các thông tin nhập tay, tính từ tập huấn luyện, để cảnh báo khi nhập giá trị hiếm."""
    X, y = load_xy("train")
    num = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "adr",
           "previous_bookings_not_canceled"]
    pc = (X["previous_cancellations"] >= 1).to_numpy()
    return {
        "hi": {c: float(X[c].quantile(0.99)) for c in num},
        "prev_c_share": float(pc.mean()),
        "prev_c_rate": float(y.to_numpy()[pc].mean()) if pc.any() else float("nan"),
        "cat": {c: X[c].value_counts(normalize=True).to_dict()
                for c in ["market_segment", "customer_type", "deposit_type"]},
    }


def unusual_inputs(base: dict) -> list:
    """Các giá trị nhập vào hiếm hoặc vô lý so với dữ liệu huấn luyện (kết quả có thể kém tin cậy)."""
    p = train_profile()
    msgs = []
    if float(base["adults"]) + float(base["children"]) == 0:
        msgs.append("Đơn không có khách nào (người lớn = 0 và trẻ em = 0).")
    if float(base["stays_in_weekend_nights"]) + float(base["stays_in_week_nights"]) == 0:
        msgs.append("Đơn có 0 đêm lưu trú.")
    if float(base["previous_cancellations"]) >= 1:
        msgs.append(f"Chỉ {fmt_pct(p['prev_c_share'] * 100)} đơn trong dữ liệu huấn luyện có khách từng hủy trước đó, và "
                    f"{fmt_pct(p['prev_c_rate'] * 100, 0)} trong số đó đã hủy. Mô hình coi đây là tín hiệu rất mạnh "
                    "nên chỉ cần 1 lần hủy trước đó là điểm tăng vọt.")
    for col, name, unit in [("lead_time", "Đặt trước", " ngày"), ("adr", "Giá phòng TB", ""),
                            ("stays_in_week_nights", "Đêm trong tuần", ""), ("stays_in_weekend_nights", "Đêm cuối tuần", ""),
                            ("adults", "Người lớn", ""), ("children", "Trẻ em", ""),
                            ("previous_bookings_not_canceled", "Số lần đã ở trước đó", "")]:
        hi = p["hi"][col]
        if float(base[col]) > hi:
            msgs.append(f"{name} = {fmt_int(float(base[col]))}{unit} cao hơn mức thường gặp (99% đơn huấn luyện không "
                        f"vượt {fmt_int(hi)}{unit}), mô hình có rất ít dữ liệu ở vùng này.")
    for col, name, lab in [("market_segment", "Nguồn đặt phòng", SEGMENT_LABEL), ("customer_type", "Loại khách", CUSTOMER_LABEL),
                           ("deposit_type", "Loại đặt cọc", DEPOSIT_LABEL)]:
        share = p["cat"][col].get(base[col], 0.0)
        if share < 0.005:
            msgs.append(f"{name} «{lab.get(base[col], base[col])}» rất hiếm trong dữ liệu huấn luyện ({share * 100:.1f}%).")
    return msgs


def align(df: pd.DataFrame, X_val: pd.DataFrame) -> pd.DataFrame:
    """Đưa DataFrame về đúng thứ tự cột và kiểu dữ liệu mà mô hình đã huấn luyện."""
    return df[list(X_val.columns)].astype(X_val.dtypes.to_dict())


# Các yếu tố trong form dùng để giải thích một đơn (cột, tên hiển thị)
FACTORS = [
    ("hotel", "Khách sạn"), ("arrival_date_month", "Tháng đến"), ("deposit_type", "Loại đặt cọc"),
    ("lead_time", "Đặt trước"), ("stays_in_weekend_nights", "Đêm cuối tuần"),
    ("stays_in_week_nights", "Đêm trong tuần"), ("adults", "Người lớn"), ("children", "Trẻ em"),
    ("adr", "Giá phòng TB"), ("previous_cancellations", "Số lần hủy trước"),
    ("market_segment", "Nguồn đặt"), ("customer_type", "Loại khách"), ("country", "Quốc gia"),
    ("previous_bookings_not_canceled", "Số lần đã ở trước đó"),
]
# Các cột luôn phải đổi cùng nhau, nếu không sẽ ra tổ hợp vô lý (ví dụ khách quen nhưng chưa từng ở lần nào)
GROUPS = {"market_segment": ["distribution_channel"], "previous_bookings_not_canceled": ["is_repeated_guest"]}
WEEKDAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


def show_value(col: str, v) -> str:
    if col == "arrival_date_month":
        return f"tháng {MONTHS.index(v) + 1}" if v in MONTHS else str(v)
    if col == "deposit_type":
        return DEPOSIT_LABEL.get(v, str(v)).lower()
    if col == "market_segment":
        return SEGMENT_LABEL.get(v, str(v))
    if col == "customer_type":
        return CUSTOMER_LABEL.get(v, str(v)).lower()
    if col == "is_repeated_guest":
        return "có" if int(v) else "không"
    if col == "lead_time":
        return f"{fmt_int(float(v))} ngày"
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return fmt_int(f) if f == int(f) else fmt_dec(f, 1)
    return str(v)


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
    st.markdown("- **Dự đoán**: nhập một đơn đặt phòng, xem mô hình đánh giá nguy cơ hủy ra sao, "
                "vì sao đơn được xếp như vậy và các mô hình có đồng ý với nhau không.\n"
                "- **Chấm hàng loạt**: tải lên danh sách đơn, nhận bảng xếp theo nguy cơ để tải về.\n"
                "- **Một tuần vận hành**: xem lại một tuần trong tập kiểm định, overbook theo mô hình thì lấp được "
                "bao nhiêu phòng và đuổi nhầm bao nhiêu khách.\n"
                "- **Kết quả**: kết luận cuối cùng trên dữ liệu mô hình chưa từng thấy.\n"
                "- **Thử mức chi phí**: kéo thử các mức chi phí giả định khác nhau, so ngưỡng đã khóa với ngưỡng lý thuyết.\n"
                "- **Giới hạn**: mô hình có thể sai ở đâu.")

    st.subheader("Dữ liệu")
    st.markdown("Bộ dữ liệu *Hotel Booking Demand*: 86.617 đơn sau làm sạch, tỷ lệ hủy 27,7%. "
                "Dữ liệu chia theo thời gian: huấn luyện (07/2015–08/2016), kiểm định (09–12/2016), "
                "kiểm tra (01–08/2017). Tập kiểm tra chỉ được mở một lần ở cuối.")
    st.caption(COST_NOTE)

    with st.expander("Giải thích các từ khó"):
        st.markdown(
            "- **Overbooking**: bán nhiều phòng hơn số phòng thật, vì dự đoán sẽ có khách hủy.\n"
            "- **Ngưỡng**: điểm nguy cơ tối thiểu để xếp một đơn vào nhóm nguy cơ cao (mô phỏng sẽ overbook các đơn này). Ngưỡng được chọn trên tập kiểm định "
            "rồi khóa lại, không chỉnh sau khi xem kết quả.\n"
            "- **r (tỷ lệ chi phí)**: một khách bị đuổi tốn gấp r lần một phòng trống. r càng lớn thì càng nên thận trọng.\n"
            "- **Bắt được đơn hủy** (recall): trong tất cả đơn hủy thật, mô hình phát hiện được bao nhiêu phần trăm.\n"
            "- **Đoán hủy thì đúng** (precision): trong các đơn mô hình đoán sẽ hủy, bao nhiêu phần trăm hủy thật.\n"
            "- **Tập kiểm định / tập kiểm tra**: kiểm định dùng để chọn ngưỡng nên kết quả lạc quan hơn; "
            "kiểm tra là dữ liệu mô hình chưa từng thấy, nên đáng tin hơn.")


# ---------- trang 1: kết quả cuối trên test ----------
def page_results():
    st.title("Kết quả trên dữ liệu chưa từng thấy")
    st.caption("Kết quả trên tập kiểm tra ở mức r = 2 (đuổi nhầm một khách tốn gấp đôi một phòng trống). "
               "Bốn mô hình dùng kết quả protocol-v1; Random Forest E1 được tính sau bằng ngưỡng 0,50, chưa tối ưu theo chi phí.")

    tf = read_result("test_final.csv")
    need = {"model", "n", "positive_rate", "recall", "precision", "fp", "tp",
            "cost_r2_locked", "cost_never", "saving_vs_never_pct"}
    if tf is None or not need.issubset(tf.columns):
        st.info("Chưa có đủ dữ liệu trong `reports/results/test_final.csv`.")
        return

    # RF được chấm hậu nghiệm; giữ nguyên file kết quả protocol-v1.
    rf_test = read_result("test_rf_e1_posthoc.csv")
    if rf_test is not None and need.issubset(rf_test.columns) and "rf_e1" in set(rf_test.model):
        tf = pd.concat([tf[tf.model != "rf_e1"], rf_test[rf_test.model == "rf_e1"]], ignore_index=True)
    else:
        st.warning("Không tìm thấy kết quả Random Forest: cần file reports/results/test_rf_e1_posthoc.csv trong bộ đầy đủ.")

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
               "Chi phí test ở mức r = 2. Random Forest được chấm sau protocol-v1 với ngưỡng 0,50.")

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
                'Các mô hình protocol dùng ngưỡng khóa; Random Forest dùng ngưỡng 0,50 tham khảo.</p>',
                unsafe_allow_html=True)
    st.caption(COST_NOTE)

    c1, c2 = st.columns([2, 1])
    model = c1.selectbox("Mô hình", MODELS, format_func=MODEL_LABEL.get)
    r = c2.select_slider("Đuổi nhầm 1 khách tốn gấp mấy lần 1 phòng trống?", options=RATIOS, value=2.0,
                         format_func=lambda v: f"{v:g} lần")
    t = locked_threshold(model, r)
    if model == "rf_e1":
        st.warning("Random Forest E1 dùng ngưỡng 0,50 của notebook cho mọi mức r; chưa tối ưu hoặc khóa ngưỡng theo chi phí, kết quả tính trên validation.")

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
    tt = theoretical_threshold(r, 1.0)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Ngưỡng tham khảo" if model == "rf_e1" else "Ngưỡng đã khóa", fmt_dec(t), help="Điểm nguy cơ tối thiểu để xếp đơn vào nhóm nguy cơ cao (mô phỏng sẽ overbook các đơn này).")
    m2.metric("Ngưỡng lý thuyết", fmt_dec(tt),
              help="r/(1+r): ngưỡng tối ưu nếu điểm của mô hình là xác suất chuẩn. Xem giải thích bên dưới biểu đồ.")
    m3.metric("Chi phí khi dùng mô hình", fmt_int(res["cost"]))
    m4.metric("Chi phí nếu không overbook", fmt_int(never))
    m5.metric("Tiết kiệm", fmt_pct(saving))
    st.caption(f"{fmt_int(len(yv))} đơn. Overbook nhầm (khách vẫn đến): {fmt_int(res['fp'])}. "
               f"Bỏ lỡ (khách hủy mà không overbook): {fmt_int(res['fn'])}.")

    curve = cost_curve(yv, pv, r, 1.0)
    line = alt.Chart(curve).mark_line(color=TEAL, strokeWidth=2.5).encode(
        x=alt.X("threshold:Q", title="Ngưỡng"),
        y=alt.Y("cost:Q", title="Tổng chi phí (đơn vị giả định)"),
        tooltip=[alt.Tooltip("threshold:Q", title="Ngưỡng"), alt.Tooltip("cost:Q", title="Chi phí", format=",.0f")])
    # hai vạch dọc: ngưỡng đã khóa (gạch) và ngưỡng lý thuyết (xanh lục); chữ đặt về hai phía để không đè nhau
    locked_right = t >= tt
    lock_df = pd.DataFrame({"t": [t], "label": [f"{'Tham khảo' if model == 'rf_e1' else 'Đã khóa'} {fmt_dec(t)}"]})
    theo_df = pd.DataFrame({"t": [tt], "label": [f"Lý thuyết {fmt_dec(tt)}"]})
    rule = alt.Chart(lock_df).mark_rule(color=BRICK, strokeDash=[5, 4], strokeWidth=2).encode(x="t:Q")
    rule_txt = alt.Chart(lock_df).mark_text(
        align="left" if locked_right else "right", dx=6 if locked_right else -6, dy=10, color=BRICK,
        fontWeight="bold").encode(x="t:Q", y=alt.value(0), text="label:N")
    theo_rule = alt.Chart(theo_df).mark_rule(color=GREEN, strokeWidth=2).encode(x="t:Q")
    theo_txt = alt.Chart(theo_df).mark_text(
        align="right" if locked_right else "left", dx=-6 if locked_right else 6, dy=10, color=GREEN,
        fontWeight="bold").encode(x="t:Q", y=alt.value(0), text="label:N")
    never_df = pd.DataFrame({"y": [never], "label": ["Không overbook"]})
    never_rule = alt.Chart(never_df).mark_rule(color=GRAY).encode(y="y:Q")
    never_txt = alt.Chart(never_df).mark_text(align="left", dx=6, dy=-8, color=GRAY).encode(
        y="y:Q", x=alt.value(0), text="label:N")
    st.altair_chart(chart_style((line + rule + rule_txt + theo_rule + theo_txt + never_rule + never_txt).properties(
        height=320, title=f"Tổng chi phí theo ngưỡng (đuổi nhầm tốn gấp {r:g} lần phòng trống)")), width="stretch")
    st.caption("Đường cong chỉ để tham khảo, không dùng để chọn lại ngưỡng.")

    if model == "rf_e1":
        st.caption("Ngưỡng 0,50 là ngưỡng đánh giá trong notebook RF, không phải ngưỡng tối ưu theo r; ngưỡng lý thuyết cần xác suất được hiệu chỉnh.")
    else:
        with st.expander("Vì sao ngưỡng đã khóa khác ngưỡng lý thuyết?"):
            mean_p, true_p = float(proba.mean()), float(y.mean())
            st.markdown(
                f"Overbook một đơn có lợi khi xác suất hủy p thỏa p × 1 > (1 − p) × {r:g}, tức p > r/(1+r) = "
                f"**{fmt_dec(tt)}**. Công thức này chỉ đúng nếu điểm của mô hình là **xác suất chuẩn**.\n\n"
                f"Trên tập kiểm định, điểm trung bình của {MODEL_LABEL[model]} là **{fmt_pct(mean_p * 100)}**, "
                f"còn tỷ lệ hủy thật là **{fmt_pct(true_p * 100)}**. "
                + ("Điểm bị thấp hơn thực tế, nên một đơn có điểm 0,43 thực ra đã nguy cơ hơn con số đó, "
                   "và ngưỡng phải hạ xuống thấp hơn mốc lý thuyết mới bắt được đủ đơn hủy."
                   if mean_p < true_p - 0.02 else
                   "Hai con số này khá gần nhau nên ngưỡng đã khóa không lệch nhiều so với mốc lý thuyết."
                   if abs(mean_p - true_p) <= 0.02 else
                   "Điểm cao hơn thực tế, nên ngưỡng phải cao hơn mốc lý thuyết mới tránh đuổi nhầm khách.")
                + "\n\nNgưỡng đã khóa được chọn bằng cách thử mọi ngưỡng trên tập kiểm định và lấy mức chi phí thấp nhất, "
                "nên nó tự bù độ lệch này. Cũng vì độ lệch đó mà app chỉ hiển thị mức nguy cơ so với ngưỡng, "
                "không hiển thị xác suất.")


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
        "f_country": countries[0],
        "f_prevok": _clip(int(d["previous_bookings_not_canceled"]), 0, 72),
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
                         "f_seg": "Online TA", "f_cust": "Transient", "f_prev": 0, "f_prevok": 0},
    "Đặt sát ngày, khách quen": {"f_deposit": "No Deposit", "f_lead": 3, "f_seg": "Direct",
                                 "f_prev": 0, "f_prevok": 2},
    "Cọc không hoàn lại": {"f_deposit": "Non Refund", "f_lead": 120, "f_seg": "Offline TA/TO"},
}


def rank_bar(pct: float, pct_thr: float) -> str:
    return (f'<div class="rankbar"><div class="rb-low" style="width:{pct_thr:.1f}%">Nhóm nguy cơ thấp</div>'
            f'<div class="rb-high" style="width:{100 - pct_thr:.1f}%">Nhóm nguy cơ cao</div>'
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
        g = st.columns(4)
        prev_ok = g[0].number_input("Số lần đã ở trước đó", 0, 72, key="f_prevok",
                                    help="Số đơn trước đây của khách này đã nhận phòng thật. Lớn hơn 0 nghĩa là khách quen.")

    base = d.to_dict()
    base.update({"hotel": hotel, "arrival_date_month": month, "deposit_type": deposit, "lead_time": lead,
                 "stays_in_weekend_nights": wk_end, "stays_in_week_nights": wk, "adults": adults,
                 "children": children, "adr": adr, "previous_cancellations": prev_c,
                 "market_segment": seg, "customer_type": cust, "country": country,
                 "is_repeated_guest": int(prev_ok > 0), "previous_bookings_not_canceled": prev_ok,
                 "distribution_channel": channel_map().get(seg, d["distribution_channel"])})
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
        warns = unusual_inputs(base)
        if warns:
            st.warning("**Thông tin nhập có giá trị hiếm, kết quả có thể kém tin cậy:**\n\n" + "\n".join(f"- {w}" for w in warns))
        st.markdown("**Kết quả**")
        if high:
            st.markdown(f'<div class="risk high"><p class="r-head">Nguy cơ hủy: CAO</p>'
                        f'<p class="r-sub">Đơn này thuộc nhóm nguy cơ hủy cao (vượt ngưỡng đã khóa {fmt_dec(t)}).</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk low"><p class="r-head">Nguy cơ hủy: THẤP</p>'
                        f'<p class="r-sub">Đơn này thuộc nhóm nguy cơ hủy thấp (dưới ngưỡng đã khóa {fmt_dec(t)}).</p></div>',
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
        rare = train_profile()["cat"]["deposit_type"].get("Refundable", 0.0)
        st.caption("Giữ nguyên các thông tin khác. Đây là mối liên hệ mô hình đã học từ dữ liệu, "
                   "không chứng minh nguyên nhân hủy."
                   + (f" Loại \"cọc hoàn lại được\" chỉ chiếm {rare * 100:.1f}% dữ liệu huấn luyện nên kết quả của nó kém tin cậy."
                      if rare < 0.005 else ""))

    st.caption("Các cột không có trong form (agent, meal, ...) dùng giá trị phổ biến nhất của tập huấn luyện. "
               "Kênh phân phối tự khớp với nguồn đặt phòng, và khách quen nghĩa là đã ở ít nhất một lần trước đó, "
               "để tránh các tổ hợp vô lý.")

    st.divider()
    section_why(load_model(MAIN), base, X_val, d, score, t)
    st.divider()
    section_compare(align(pd.DataFrame([base]), X_val), r)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def section_why(mdl, base: dict, X_val: pd.DataFrame, d: pd.Series, score: float, t: float):
    """Thay RIÊNG từng yếu tố bằng mức phổ biến, giữ nguyên phần còn lại, xem điểm của mô hình đổi bao nhiêu.

    Đo bằng chênh lệch log-odds (thang tương đối) chứ không dùng vị trí xếp hạng: với đơn nguy cơ rất cao hoặc rất thấp,
    vị trí xếp hạng bị 'chạm trần' nên mọi yếu tố trông như chỉ đổi 1-2 điểm.
    """
    st.subheader("Vì sao đơn này được xếp như vậy?")
    variants, meta = [], []
    for col, name in FACTORS:
        cols = [col] + GROUPS.get(col, [])
        if any(base[c] != d[c] for c in cols):
            variants.append({**base, **{c: d[c] for c in cols}})
            meta.append((col, name))
    if not variants:
        st.info("Mọi yếu tố trong form đều đang ở mức phổ biến, nên không có gì để so sánh.")
        return
    sc = mdl.predict_proba(align(pd.DataFrame(variants), X_val))[:, 1]
    delta = float(_logit(score)) - _logit(sc)  # dương: yếu tố này đang đẩy nguy cơ lên so với mức phổ biến
    cur_high = score >= t
    flips = (sc >= t) != cur_high
    df = pd.DataFrame({
        "label": [f"{n}: {show_value(c, base[c])} (phổ biến: {show_value(c, d[c])})" for c, n in meta],
        "name": [n for _, n in meta], "delta": delta, "flip": flips})
    df = df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(8).reset_index(drop=True)
    df["txt"] = df["delta"].map(lambda v: f"{v:+.1f}".replace("-", "−").replace(".", ","))

    now, then = ("cao", "thấp") if cur_high else ("thấp", "cao")
    flipped = [n.lower() for n in df.loc[df["flip"], "name"]]
    top = df.sort_values("delta", ascending=False).iloc[0]
    msgs = []
    if flipped:
        msgs.append(f"Chỉ cần đưa riêng **{', '.join(flipped)}** về mức phổ biến là đơn này chuyển từ nguy cơ "
                    f"**{now}** sang **{then}**.")
    if top["delta"] >= 0.3:
        msgs.append(f"Yếu tố **đẩy nguy cơ lên nhiều nhất** là **{top['name'].lower()}**.")
    elif not flipped:
        msgs.append("Không yếu tố nào trong form đang đẩy nguy cơ của đơn này cao hơn mức phổ biến.")
    st.markdown(" ".join(msgs))

    y_enc = alt.Y("label:N", sort=df.sort_values("delta", ascending=False)["label"].tolist(), title=None,
                  axis=alt.Axis(labelLimit=420))
    lim = max(1.0, float(df["delta"].abs().max()) * 1.3)  # trục đối xứng quanh 0 để thanh nhỏ trông nhỏ
    bars = alt.Chart(df).mark_bar().encode(
        y=y_enc,
        x=alt.X("delta:Q", title="Đẩy nguy cơ lên (+) hay kéo xuống (−), đơn vị tương đối",
                scale=alt.Scale(domain=[-lim, lim], nice=False)),
        color=alt.condition(alt.datum.delta >= 0, alt.value(BRICK), alt.value(GREEN)),
        tooltip=[alt.Tooltip("label:N", title="Yếu tố"), alt.Tooltip("delta:Q", title="Sức đẩy", format="+.2f"),
                 alt.Tooltip("flip:N", title="Đổi kết luận Cao/Thấp?")])
    pos = alt.Chart(df).mark_text(align="left", dx=4, color=INK).encode(
        y=y_enc, x=alt.X("delta:Q", scale=alt.Scale(domain=[-lim, lim], nice=False)), text="txt:N").transform_filter(alt.datum.delta >= 0)
    neg = alt.Chart(df).mark_text(align="right", dx=-4, color=INK).encode(
        y=y_enc, x=alt.X("delta:Q", scale=alt.Scale(domain=[-lim, lim], nice=False)), text="txt:N").transform_filter(alt.datum.delta < 0)
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK).encode(x="z:Q")
    st.altair_chart(chart_style((bars + pos + neg + zero).properties(height=max(120, 36 * len(df)))), width="stretch")
    st.caption("Cách đọc: thay riêng yếu tố đó bằng mức phổ biến của tập huấn luyện (trong ngoặc), giữ nguyên mọi thứ "
               "khác, rồi xem điểm của mô hình đổi bao nhiêu. Đơn vị là thay đổi log-odds của điểm, chỉ để so sánh các "
               "yếu tố với nhau (+1 nghĩa là odds hủy tăng khoảng 2,7 lần), không phải xác suất. Chỉ hiện tối đa "
               "8 yếu tố tác động lớn nhất; các yếu tố tác động qua lại nên các thanh không cộng lại thành tổng. "
               "Đây là mối liên hệ mô hình đã học từ dữ liệu, không chứng minh nguyên nhân hủy.")


def section_compare(row: pd.DataFrame, r: float):
    """So sánh các mô hình; riêng RF dùng ngưỡng tham khảo 0,50."""
    st.subheader("Các mô hình khác nói gì về đơn này?")
    out = []
    missing = []
    for m in MODELS:
        try:
            mdl = load_model(m)
            vs = val_sorted(m)
            thr = locked_threshold(m, r)
            sc = float(mdl.predict_proba(row)[0, 1])
            out.append({"m": m, "high": sc >= thr, "pct": float(rank_pct(vs, sc)[0]),
                        "flag_rate": 100 - float(rank_pct(vs, thr)[0])})
        except Exception as exc:
            missing.append((m, exc))
    if missing:
        for model, exc in missing:
            st.warning(f"Không tải được {MODEL_LABEL[model]}: {exc}. Kiểm tra file "
                       f"models/{model}.joblib, models/proba_val_{model}.npy và thư viện trong requirements.txt.")
    if len(out) < 2:
        st.info("Chưa tải được đủ file mô hình trong thư mục `models/` để so sánh.")
        return
    n_high = sum(o["high"] for o in out)
    if n_high == len(out):
        st.markdown(f"**Cả {len(out)} mô hình** đều xếp đơn này vào nhóm nguy cơ **cao**.")
    elif n_high == 0:
        st.markdown(f"**Cả {len(out)} mô hình** đều xếp đơn này vào nhóm nguy cơ **thấp**.")
    else:
        st.markdown(f"**{n_high}/{len(out)} mô hình** xếp đơn này vào nhóm nguy cơ cao, các mô hình còn lại "
                    "cho là thấp. Các mô hình chưa đồng thuận với nhau.")
    view = pd.DataFrame({
        "Mô hình": [MODEL_LABEL[o["m"]] for o in out],
        "Kết luận": ["Cao" if o["high"] else "Thấp" for o in out],
        "Vị trí xếp hạng": [round(o["pct"], 1) for o in out],
        "Tỷ lệ đơn bị đánh dấu Cao": [round(o["flag_rate"], 1) for o in out],
    })
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Vị trí xếp hạng": st.column_config.ProgressColumn(
            "Vị trí xếp hạng", format="%.0f", min_value=0, max_value=100,
            help="Điểm của đơn này cao hơn bao nhiêu phần trăm đơn trong tập kiểm định, theo từng mô hình."),
        "Tỷ lệ đơn bị đánh dấu Cao": st.column_config.NumberColumn(
            "Tỷ lệ đơn bị đánh dấu Cao", format="%.0f%%",
            help="Ở ngưỡng đang dùng, mô hình này đánh dấu Cao bao nhiêu phần trăm đơn trong tập kiểm định. "
                 "Số càng nhỏ, mô hình càng thận trọng.")})
    st.caption(f"Các mô hình protocol-v1 dùng ngưỡng đã khóa riêng ở r = {r:g}; Random Forest E1 dùng ngưỡng 0,50, chưa khóa theo chi phí. Cùng một đơn có thể được xếp khác nhau.")


# ---------- trang: chấm hàng loạt ----------
BATCH_MAX_ROWS = 20000
BATCH_SHOW = {"hotel": "Khách sạn", "arrival_date_month": "Tháng đến", "lead_time": "Đặt trước (ngày)",
              "deposit_type": "Loại cọc", "market_segment": "Nguồn đặt", "country": "Quốc gia",
              "adr": "Giá TB/đêm"}
CHECK_UNSEEN = ["hotel", "arrival_date_month", "deposit_type", "market_segment", "customer_type"]


def prepare_batch(raw: pd.DataFrame, X_val: pd.DataFrame, d: pd.Series):
    """Đưa file người dùng về đúng cột, đúng kiểu. Trả về (X, cột thiếu, số ô phải điền, cảnh báo giá trị lạ)."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in X_val.columns if c not in raw.columns]
    out = pd.DataFrame(index=raw.index)
    filled = 0
    for c in X_val.columns:
        if c not in raw.columns:
            out[c] = d[c]
            continue
        s_ = raw[c]
        s_ = pd.to_numeric(s_, errors="coerce") if pd.api.types.is_numeric_dtype(X_val[c]) else s_.astype("object")
        bad = s_.isna()
        filled += int(bad.sum())
        out[c] = s_.where(~bad, d[c])
    unseen = []
    for c in CHECK_UNSEEN:
        if c in raw.columns:
            k = int((~out[c].astype(str).isin(set(X_val[c].astype(str)))).sum())
            if k:
                unseen.append((c, k))
    return align(out, X_val), missing, filled, unseen


def page_batch():
    st.title("Chấm hàng loạt đơn đặt phòng")
    st.markdown('<p class="lede">Tải lên danh sách đơn đặt phòng sắp đến, nhận lại bảng xếp theo nguy cơ hủy: đơn nào '
                'vượt ngưỡng được xếp vào nhóm nguy cơ cao và lên đầu bảng. Có nên overbook hay không còn phụ thuộc chi phí giả định '
                '(xem trang Kết quả mô hình).</p>',
                unsafe_allow_html=True)
    X_val, y_val = val_data()
    d = train_defaults()

    c1, c2 = st.columns([2, 1])
    model = c1.selectbox("Mô hình", MODELS, format_func=MODEL_LABEL.get, key="b_model")
    r = c2.select_slider("Đuổi nhầm 1 khách tốn gấp mấy lần 1 phòng trống?", options=RATIOS, value=2.0,
                         format_func=lambda v: f"{v:g} lần", key="b_r")
    t = locked_threshold(model, r)
    mdl = try_load_model(model)
    if model == "rf_e1":
        st.warning("Random Forest E1 dùng ngưỡng 0,50 từ notebook; chưa được khóa hay tối ưu theo mức chi phí r.")
    if mdl is None:
        st.error(f"Chưa tìm thấy file mô hình `models/{model}.joblib`. Hãy chép file này vào thư mục `models/`.")
        return

    st.markdown("**Dữ liệu đầu vào**")
    u1, u2 = st.columns([1, 1])
    tmpl = X_val.sample(30, random_state=11).sort_index()
    u1.download_button("Tải file CSV mẫu (30 đơn)", tmpl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="smartstay_mau.csv", mime="text/csv",
                       help="Dùng làm khuôn: giữ nguyên tên cột, thay bằng đơn của bạn.")
    use_sample = u2.toggle("Dùng 200 đơn mẫu từ tập kiểm định", value=False,
                           help="Không cần tải file. Có sẵn cột is_canceled nên đối chiếu được với kết quả thật.")
    up = None if use_sample else st.file_uploader("Tải lên file CSV", type=["csv"])

    if use_sample:
        idx = np.sort(np.random.default_rng(7).choice(len(X_val), 200, replace=False))
        raw = X_val.iloc[idx].reset_index(drop=True)
        raw["is_canceled"] = y_val[idx]
    elif up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Không đọc được file CSV: {e}")
            return
    else:
        st.info("Tải lên một file CSV hoặc bật \"Dùng 200 đơn mẫu\" để xem thử.")
        with st.expander("File cần có những cột nào?"):
            st.markdown("Tên cột giống file mẫu: " + ", ".join(f"`{c}`" for c in X_val.columns) + ".\n\n"
                        "Cột nào thiếu hoặc bỏ trống sẽ được điền bằng giá trị phổ biến nhất của tập huấn luyện, "
                        "nên càng đủ cột thì kết quả càng đáng tin. Cột `is_canceled` (0/1) là tùy chọn, nếu có thì "
                        "app đối chiếu với kết quả thật.")
        return

    if len(raw) == 0:
        st.error("File không có dòng dữ liệu nào.")
        return
    if len(raw) > BATCH_MAX_ROWS:
        st.warning(f"File có {fmt_int(len(raw))} dòng, app chỉ chấm {fmt_int(BATCH_MAX_ROWS)} dòng đầu.")
        raw = raw.head(BATCH_MAX_ROWS).reset_index(drop=True)
    raw = raw.reset_index(drop=True)

    try:
        X, missing, filled, unseen = prepare_batch(raw, X_val, d)
        proba = mdl.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Không chấm được file này. Hãy kiểm tra kiểu dữ liệu từng cột. Chi tiết: {e}")
        return

    if missing:
        st.warning(f"File thiếu {len(missing)} cột ({', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}), "
                   "đã điền bằng giá trị phổ biến nhất của tập huấn luyện. Kết quả kém tin cậy hơn.")
    if filled:
        st.caption(f"{fmt_int(filled)} ô trống hoặc sai kiểu đã được điền bằng giá trị phổ biến nhất.")
    for c, k in unseen:
        st.caption(f"Cột `{c}` có {fmt_int(k)} dòng mang giá trị chưa từng xuất hiện trong dữ liệu kiểm định.")

    vs = val_sorted(model)
    high = proba >= t
    pct = rank_pct(vs, proba)
    res = pd.DataFrame({
        "Hạng": pd.Series(proba).rank(ascending=False, method="first").astype(int).to_numpy(),
        "Dòng": np.arange(1, len(raw) + 1),
        "Mức nguy cơ": np.where(high, "Cao", "Thấp"),
        "Vị trí xếp hạng": pct.round(1),
    })
    shown = [c for c in BATCH_SHOW if c in raw.columns]
    res = pd.concat([res, X[shown].rename(columns=BATCH_SHOW)], axis=1).sort_values("Hạng").reset_index(drop=True)

    st.subheader("Kết quả")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Số đơn đã chấm", fmt_int(len(raw)))
    k2.metric("Vượt ngưỡng (Cao)", fmt_int(int(high.sum())))
    k3.metric("Tỷ lệ vượt ngưỡng", fmt_pct(high.mean() * 100))
    k4.metric("Ngưỡng tham khảo" if model == "rf_e1" else "Ngưỡng đã khóa", fmt_dec(t))

    only_high = st.checkbox("Chỉ hiện đơn vượt ngưỡng", value=False)
    view = res[res["Mức nguy cơ"] == "Cao"] if only_high else res
    st.dataframe(view.head(1000), hide_index=True, width="stretch", column_config={
        "Vị trí xếp hạng": st.column_config.ProgressColumn(
            "Vị trí xếp hạng", format="%.0f", min_value=0, max_value=100,
            help="Điểm của đơn cao hơn bao nhiêu phần trăm đơn trong tập kiểm định. Không phải xác suất hủy.")})
    if len(view) > 1000:
        st.caption(f"Bảng chỉ hiện 1.000 dòng đầu. File tải về có đủ {fmt_int(len(view))} dòng.")

    out = raw.copy()
    r_idx = res.set_index("Dòng")
    out["hang"] = r_idx.loc[np.arange(1, len(raw) + 1), "Hạng"].to_numpy()
    out["muc_nguy_co"] = np.where(high, "Cao", "Thap")
    out["vi_tri_xep_hang"] = pct.round(1)
    out = out.sort_values("hang")
    st.download_button("Tải kết quả (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="smartstay_ket_qua.csv", mime="text/csv")
    st.caption("App không hiển thị xác suất thô: điểm của mô hình chưa được hiệu chỉnh nên chỉ có mức nguy cơ so với "
               "ngưỡng đang dùng và vị trí xếp hạng so với tập kiểm định. File chỉ được xử lý trong phiên chạy này, "
               "app không lưu lại.")

    if "is_canceled" in raw.columns:
        yy = pd.to_numeric(raw["is_canceled"], errors="coerce")
        if yy.isin([0, 1]).all():
            yy = yy.astype(int).to_numpy()
            rs = cost_at_threshold(yy, proba, t, r, 1.0)
            never = baseline_costs(yy, r, 1.0)["never_overbook"]
            sv = (never - rs["cost"]) / never * 100 if never else float("nan")
            with st.expander("Đối chiếu với kết quả thật (file có cột is_canceled)", expanded=use_sample):
                e1, e2, e3 = st.columns(3)
                e1.metric("Chi phí khi dùng mô hình", fmt_int(rs["cost"]))
                e2.metric("Chi phí nếu không overbook", fmt_int(never))
                e3.metric("Tiết kiệm", fmt_pct(sv))
                st.caption(f"Overbook nhầm (khách vẫn đến): {fmt_int(rs['fp'])}. Bỏ lỡ (khách hủy mà không overbook): "
                           f"{fmt_int(rs['fn'])}. " + COST_NOTE
                           + (" Đây là đơn lấy từ tập kiểm định đã dùng để chọn ngưỡng nên kết quả lạc quan." if use_sample else ""))


# ---------- trang: một tuần vận hành ----------
@st.cache_data
def val_dates() -> np.ndarray:
    X, _ = val_data()
    month = X["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    dt = pd.to_datetime(pd.DataFrame({"year": X["arrival_date_year"], "month": month,
                                      "day": X["arrival_date_day_of_month"]}))
    return dt.to_numpy()


def page_week():
    st.title("Một tuần vận hành thử")
    st.markdown('<p class="lede">Quay lại một tuần trong tập kiểm định và làm như một khách sạn: đơn nào vượt ngưỡng '
                'thì bán thêm một phòng. Sau đó xem sự thật: bao nhiêu phòng được lấp lại, bao nhiêu khách phải chuyển '
                'đi nơi khác.</p>', unsafe_allow_html=True)
    st.caption(COST_NOTE)

    X, y = val_data()
    dates = pd.DatetimeIndex(val_dates())
    wk_start = (dates - pd.to_timedelta(dates.dayofweek, unit="D")).normalize()
    weeks = sorted(wk_start.unique())
    counts = pd.Series(wk_start).value_counts()
    default_idx = weeks.index(counts.idxmax())

    a1, a2 = st.columns([2, 2])
    week = a1.selectbox("Tuần (theo ngày đến)", weeks, index=default_idx, key="w_week",
                        format_func=lambda w: f"{w:%d/%m/%Y} – {w + pd.Timedelta(days=6):%d/%m/%Y} "
                                              f"({fmt_int(int(counts[w]))} đơn)")
    hotel = a2.radio("Khách sạn", ["Tất cả", "City Hotel", "Resort Hotel"], horizontal=True, key="w_hotel")
    b1, b2 = st.columns([2, 2])
    model = b1.selectbox("Mô hình", MODELS, format_func=MODEL_LABEL.get, key="w_model")
    r = b2.select_slider("Đuổi nhầm 1 khách tốn gấp mấy lần 1 phòng trống?", options=RATIOS, value=2.0,
                         format_func=lambda v: f"{v:g} lần", key="w_r")
    t = locked_threshold(model, r)
    if model == "rf_e1":
        st.warning("Random Forest E1 dùng ngưỡng 0,50; không phải ngưỡng đã khóa theo chi phí.")

    mask = np.asarray(wk_start == week)
    if hotel != "Tất cả":
        mask &= (X["hotel"] == hotel).to_numpy()
    n = int(mask.sum())
    if n == 0:
        st.info("Không có đơn nào trong lựa chọn này.")
        return
    yv, pv, dv = y[mask], val_proba(model)[mask], dates[mask]
    flag = pv >= t
    is_tp, is_fp, is_fn = flag & (yv == 1), flag & (yv == 0), ~flag & (yv == 1)
    tp, fp, fn = int(is_tp.sum()), int(is_fp.sum()), int(is_fn.sum())
    n_cancel = int((yv == 1).sum())
    cost = r * fp + fn
    never = float(n_cancel)

    span = f"{week:%d/%m}–{(week + pd.Timedelta(days=6)):%d/%m}"
    if never == 0:
        verdict("good", f"Tuần {span}: không có đơn nào bị hủy.",
                f"Mô hình khuyên overbook {fmt_int(int(flag.sum()))} đơn, và cả {fmt_int(fp)} khách đều vẫn đến.")
    else:
        sv = (never - cost) / never * 100
        head = (f"Tuần {span}: overbook theo {MODEL_LABEL[model]} tiết kiệm {fmt_pct(sv)} so với không overbook."
                if sv > 0 else
                f"Tuần {span}: overbook theo {MODEL_LABEL[model]} tốn hơn {fmt_pct(abs(sv))} so với không overbook.")
        body = (f"Có {fmt_int(n)} đơn đến trong tuần, {fmt_int(n_cancel)} đơn trong đó bị hủy nhưng lúc đặt chưa ai biết. "
                f"Mô hình khuyên overbook {fmt_int(tp + fp)} đơn: {fmt_int(tp)} đơn khách hủy thật nên phòng được lấp lại, "
                f"còn {fmt_int(fp)} đơn khách vẫn đến nên phải chuyển khách đi nơi khác. "
                f"Có {fmt_int(fn)} đơn hủy mà mô hình không bắt được, phòng bị bỏ trống.")
        verdict("good" if sv > 0 else "warn", head, body)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Phòng được lấp lại", fmt_int(tp), help="Đơn vượt ngưỡng mà khách hủy thật.")
    m2.metric("Khách bị chuyển đi", fmt_int(fp), help="Đơn vượt ngưỡng mà khách vẫn đến: mỗi lần tốn r đơn vị.")
    m3.metric("Phòng trống bỏ lỡ", fmt_int(fn), help="Đơn hủy nhưng mô hình không bắt được: mỗi lần tốn 1 đơn vị.")
    m4.metric("Chi phí / không overbook", f"{fmt_int(cost)} / {fmt_int(never)}")

    days = pd.date_range(week, periods=7)
    labels = [f"{WEEKDAYS[i]} {dd:%d/%m}" for i, dd in enumerate(days)]
    dnorm = pd.DatetimeIndex(dv).normalize()
    rows = []
    for i, dd in enumerate(days):
        on = np.asarray(dnorm == dd)
        rows += [{"ngay": labels[i], "loai": "Phòng được lấp lại", "so": int((on & is_tp).sum())},
                 {"ngay": labels[i], "loai": "Khách bị chuyển đi", "so": int((on & is_fp).sum())},
                 {"ngay": labels[i], "loai": "Phòng trống bỏ lỡ", "so": int((on & is_fn).sum())}]
    ch = alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        x=alt.X("ngay:N", sort=labels, title="Ngày đến", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("so:Q", title="Số đơn"),
        color=alt.Color("loai:N", legend=alt.Legend(title=None, orient="top"),
                        scale=alt.Scale(domain=["Phòng được lấp lại", "Khách bị chuyển đi", "Phòng trống bỏ lỡ"],
                                        range=[GREEN, BRICK, GRAY])),
        tooltip=[alt.Tooltip("ngay:N", title="Ngày"), alt.Tooltip("loai:N", title="Loại"),
                 alt.Tooltip("so:Q", title="Số đơn")])
    st.altair_chart(chart_style(ch.properties(height=300, title="Kết quả từng ngày trong tuần")), width="stretch")
    st.caption("Mô phỏng đơn giản trên tập kiểm định: mỗi đơn vượt ngưỡng overbook đúng một phòng, không tính sức chứa "
               "thật của khách sạn. Tập này đã được dùng để chọn ngưỡng nên kết quả lạc quan hơn thực tế; kết luận cuối "
               "cùng nằm ở trang Kết quả mô hình.")


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
def check_required_files():
    """Báo rõ file nào còn thiếu (ví dụ clone repo mà .gitignore đã chặn file mô hình) thay vì để app sập."""
    need = [MODELS_DIR / f"{MAIN}.joblib", MODELS_DIR / f"proba_val_{MAIN}.npy",
            ROOT / "data" / "processed" / "train.csv", ROOT / "data" / "processed" / "val.csv",
            RESULTS / "frozen_thresholds.csv"]
    missing = [p.relative_to(ROOT).as_posix() for p in need if not p.exists()]
    if missing:
        st.error("Thiếu file cần cho bản demo: " + ", ".join(f"`{m}`" for m in missing)
                 + ". Kiểm tra lại repo (lệnh `git ls-files models data`) và `.gitignore`, rồi xem mục "
                   "\"Chạy bản demo\" trong README.")
        st.stop()


check_required_files()
nav = st.navigation([
    st.Page(page_single, title="Dự đoán", icon=":material/hotel:", url_path="du-doan", default=True),
    st.Page(page_batch, title="Chấm hàng loạt", icon=":material/upload_file:", url_path="cham-hang-loat"),
    st.Page(page_week, title="Một tuần vận hành", icon=":material/calendar_month:", url_path="mot-tuan-van-hanh"),
    st.Page(page_results, title="Kết quả mô hình", icon=":material/fact_check:", url_path="ket-qua"),
    st.Page(page_cost_explorer, title="Thử mức chi phí", icon=":material/tune:", url_path="thu-muc-chi-phi"),
    st.Page(page_limits, title="Giới hạn", icon=":material/report:", url_path="gioi-han"),
    st.Page(page_intro, title="Về dự án", icon=":material/info:", url_path="ve-du-an"),
], position="top")
nav.run()