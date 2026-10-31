import streamlit as st
import numpy as np
import pandas as pd

# =====================================================================
# CẤU HÌNH TRANG STREAMLIT (TỐI ƯU GIAO DIỆN DI ĐỘNG)
# =====================================================================
st.set_page_config(
    page_title="Keno Quantitative Engine",
    page_icon="🎯",
    layout="centered"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LỚP MÔ HÌNH TOÁN HỌC (MODULE 1, 2, 3)
# =====================================================================
class KenoQuantitativeEngine:
    """Động cơ định lượng Keno: EKF + Local Non-Ergodicity Scanning"""
    def __init__(self, num_numbers=80, window_size=5):
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
        delta_p = prob_window - 0.25  # P_0 = 0.25
        return delta_p

# =====================================================================
# GIAO DIỆN STREAMLIT & NHẬP LIỆU THỦ CÔNG
# =====================================================================
st.title("🎯 Keno Quantitative Engine")
st.caption("Nhập dữ liệu thủ công và phân tích kẽ hở xác suất ngắn hạn")

st.divider()

# Khu vực nhập liệu thủ công
st.subheader("📥 Nhập dữ liệu kỳ quay thủ công")
st.markdown("Mỗi dòng tương ứng với kết quả 1 kỳ quay (các số cách nhau bằng dấu phẩy hoặc khoảng trắng, ví dụ: `1, 5, 12, ...`)")

# Mẫu mặc định để người dùng dễ hình dung định dạng nhập
default_input = """1, 3, 5, 12, 15, 22, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 72, 75, 78, 80
2, 4, 6, 11, 16, 21, 29, 31, 36, 41, 46, 51, 56, 61, 66, 71, 73, 76, 79, 70
3, 7, 9, 14, 18, 23, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 74, 77, 68, 69
4, 8, 10, 13, 17, 24, 26, 33, 38, 43, 48, 53, 58, 63, 68, 73, 75, 78, 64, 65"""

raw_text_input = st.text_area("Danh sách các kỳ quay (Dán dữ liệu của bạn vào đây):", value=default_input, height=160)

# Xử lý chuỗi văn bản do người dùng nhập thủ công thành ma trận số nguyên
history_draws = []
for line in raw_text_input.strip().split("\n"):
    if line.strip():
        try:
            numbers = [int(n.strip()) for n in line.replace(",", " ").split() if n.strip()]
            if numbers:
                history_draws.append(numbers)
        except ValueError:
            continue

# Thanh trượt chọn cửa sổ ngắn hạn W
max_w = max(len(history_draws), 2)
window_size = st.slider("Cửa sổ trượt ngắn hạn W (kỳ quay):", min_value=2, max_value=max_w, value=min(len(history_draws), 5))

if len(history_draws) > 0:
    # Khởi tạo động cơ định lượng với dữ liệu thực tế từ người dùng
    engine = KenoQuantitativeEngine(num_numbers=80, window_size=window_size)

    # Chạy EKF qua từng kỳ quay
    for draw in history_draws:
        binary_vector = np.zeros(80)
        for num in draw:
            if 1 <= num <= 80:
                binary_vector[num - 1] = 1
        kalman_scores = engine.update_ekf(binary_vector)

    # Phân tích độ lệch cục bộ Delta P
    delta_p = engine.analyze_local_non_ergodicity(history_draws)

    # Lọc Top Cold Attractors (Vùng nén áp suất cao)
    cold_indices = np.argsort(delta_p)[:min(6, len(delta_p))]
    top_numbers = [idx + 1 for idx in cold_indices]

    st.divider()
    st.subheader("📍 Bộ số ưu tiên ngắn hạn (Cold Attractors)")
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3, col1, col2, col3]

    for i, num in enumerate(top_numbers):
        with cols[i]:
            st.metric(
                label=f"Số {num:02d}", 
                value=f"Score: {kalman_scores[num-1]:.3f}",
                delta=f"ΔP: {delta_p[num-1]:.2f}",
                delta_color="inverse"
            )

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
    st.info("Vui lòng nhập ít nhất một dòng kết quả kỳ quay vào ô bên trên để hệ thống tiến hành phân tích.")
