import streamlit as st
import numpy as np
import pandas as pd

# =====================================================================
# CẤU HÌNH GIAO DIỆN TỐI ƯU CHO MOBILE
# =====================================================================
st.set_page_config(
    page_title="Keno Quantitative Engine",
    page_icon="🎯",
    layout="centered"
)

st.markdown("""
<style>
    .main-card {
        background-color: #1e293b;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 10px;
    }
    .number-highlight {
        font-size: 32px;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-text {
        font-size: 13px;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LỚP MÔ HÌNH TOÁN HỌC (EKF & LOCAL NON-ERGODICITY)
# =====================================================================
class KenoQuantitativeEngine:
    def __init__(self, num_numbers=80, window_size=3):
        self.num_numbers = num_numbers
        self.window_size = window_size
        
        # Extended Kalman Filter Parameters
        self.x = np.full(num_numbers, 0.25)
        self.P = np.eye(num_numbers) * 0.1
        self.Q = np.eye(num_numbers) * 0.01
        self.R = np.eye(num_numbers) * 0.2

    def update_ekf(self, observation_vector):
        x_pred = self.x
        P_pred = self.P + self.Q
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        y = observation_vector - x_pred
        self.x = x_pred + K @ y
        self.P = (np.eye(self.num_numbers) - K) @ P_pred
        return self.x

    def analyze_local_non_ergodicity(self, history_draws):
        recent_window = history_draws[-self.window_size:]
        matrix = np.zeros((len(recent_window), self.num_numbers))
        
        for idx, draw in enumerate(recent_window):
            for num in draw:
                if 1 <= num <= self.num_numbers:
                    matrix[idx, int(num) - 1] = 1
        
        prob_window = np.mean(matrix, axis=0)
        delta_p = prob_window - 0.25  # P_0 lý thuyết = 0.25
        return delta_p

# =====================================================================
# GIAO DIỆN CHÍNH
# =====================================================================
st.title("🎯 Keno Signal Engine")
st.caption("Mô hình định lượng khai thác kẽ hở ngắn hạn (Short-term Non-ergodicity)")

st.divider()

# Nhập liệu thủ công (Mặc định 3 kỳ gần nhất đúng theo yêu cầu)
st.subheader("📥 Nhập dữ liệu kỳ quay thủ công")
st.markdown("Mỗi dòng là một kỳ quay (các số cách nhau bằng dấu phẩy hoặc khoảng trắng):")

default_input = """37, 51, 35, 56, 42, 57, 57, 59, 58, 58, 64, 59, 65, 66, 62, 66, 68, 68, 67, 72
38, 53, 41, 56, 42, 57, 57, 58, 58, 59, 64, 59, 65, 66, 62, 66, 68, 68, 67, 72
53, 56, 42, 57, 57, 58, 58, 59, 64, 59, 65, 66, 62, 66, 68, 68, 67, 72, 74, 77"""

raw_text_input = st.text_area("Danh sách các kỳ quay:", value=default_input, height=130)

# Xử lý dữ liệu đầu vào
history_draws = []
for line in raw_text_input.strip().split("\n"):
    if line.strip():
        try:
            numbers = [int(n.strip()) for n in line.replace(",", " ").split() if n.strip()]
            if numbers:
                history_draws.append(numbers)
        except ValueError:
            continue

# Cố định hoặc cho phép chọn cửa sổ trượt W = 3 (hoặc tùy biến theo số kỳ nhập)
max_w = max(len(history_draws), 2)
window_size = st.slider("Cửa sổ trượt ngắn hạn W (kỳ quay):", min_value=2, max_value=max_w, value=min(len(history_draws), 3))

if len(history_draws) > 0:
    # Khởi tạo và chạy thuật toán
    engine = KenoQuantitativeEngine(num_numbers=80, window_size=window_size)

    for draw in history_draws:
        binary_vector = np.zeros(80)
        for num in draw:
            if 1 <= num <= 80:
                binary_vector[num - 1] = 1
        kalman_scores = engine.update_ekf(binary_vector)

    delta_p = engine.analyze_local_non_ergodicity(history_draws)

    # Lọc lấy ĐÚNG 3 BỘ SỐ / 3 TÍN HIỆU TỐT NHẤT (Cold Attractors tiềm năng nhất)
    cold_indices = np.argsort(delta_p)[:3]
    top_3_numbers = [idx + 1 for idx in cold_indices]

    st.divider()
    st.subheader("🔥 Top 3 Bộ số Khai thác Tối ưu (Kỳ T+1 & T+2)")
    
    # Hiển thị trực quan 3 bộ số gợi ý chính dạng thẻ lớn
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, num in enumerate(top_3_numbers):
        with cols[i]:
            score_val = kalman_scores[num-1]
            delta_val = delta_p[num-1]
            st.markdown(f"""
                <div class="main-card">
                    <div class="metric-text">GỢI Ý #{i+1}</div>
                    <div class="number-highlight">{num:02d}</div>
                    <div class="metric-text">Score: <b>{score_val:.3f}</b></div>
                    <div class="metric-text">ΔP: <span style="color: #f43f5e;">{delta_val:.2f}</span></div>
                </div>
            """, unsafe_allow_html=True)

    st.warning("⚠️ **Kỷ luật Tau_crit = 2:** Cửa sổ vàng chỉ áp dụng cho **kỳ T+1 và T+2**. Tự động hủy lệnh nếu quá thời gian tới hạn.")

    st.divider()
    tab1, tab2 = st.tabs(["📊 Biểu đồ Phân bổ ΔP", "📜 Dữ liệu đã nhập"])

    with tab1:
        st.write("#### Biểu đồ lệch pha cục bộ so với P_0 (0.25)")
        chart_data = pd.DataFrame({
            "Số Keno": [f"{i:02d}" for i in range(1, 81)],
            "Độ lệch ΔP": delta_p
        })
        st.bar_chart(chart_data.set_index("Số Keno"))

    with tab2:
        st.write("#### Chi tiết các kỳ quay đã nhập thủ công")
        df_history = pd.DataFrame({
            f"Kỳ quay {i+1}": sorted(draw) 
            for i, draw in enumerate(history_draws)
        })
        st.dataframe(df_history, use_container_width=True)
else:
    st.info("Vui lòng nhập ít nhất một dòng kết quả kỳ quay để hệ thống tiến hành phân tích.")
