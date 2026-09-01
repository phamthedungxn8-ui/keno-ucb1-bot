import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from main import KenoQuantumEngine, optimize_bac_2_3

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="AlphaVietlott Direct Quantum Engine v4.0",
    page_icon="🎯",
    layout="wide"
)

# 2. Tự động làm mới trang mỗi 60 giây (60.000 ms) để bắt kỳ Keno mới nhất từ nhà cái
st_autorefresh(interval=60000, key="keno_realtime_refresher")

st.title("🎯 ALPHAVIETLOTT DIRECT QUANTUM ENGINE v4.0")
st.caption("Cửa sổ trượt thời gian thực (Real-time Sliding Window) | Tự động đồng bộ kỳ mới")

# 3. Thanh điều hướng cấu hình
st.sidebar.header("⚙️ CẤU HÌNH HỆ THỐNG")
n_draws_input = st.sidebar.slider("Độ rộng Cửa sổ trượt (Số kỳ phân tích):", min_value=100, max_value=1000, value=500, step=50)

# Khởi tạo Engine với số kỳ cấu hình
engine = KenoQuantumEngine(draw_count=n_draws_input)

# 4. Hàm nạp dữ liệu với Cache ngắn hạn (TTL = 30s) để luôn cập nhật theo thời gian thực
@st.cache_data(ttl=30)
def load_realtime_data(draws_count):
    return engine.fetch_live_data()

# Lấy ma trận dữ liệu nhị phân 500 kỳ mới nhất tính đến giây hiện tại
binary_matrix = load_realtime_data(n_draws_input)

st.success(f"⚡ Hệ thống đang kết nối trực tiếp: **{binary_matrix.shape[0]} kỳ quay mới nhất** đã sẵn sàng trong ma trận Quantum Tensor.")

# 5. Khu vực hiển thị kết quả phân tích
if st.button("🔥 CHẠY THUẬT TOÁN TỐI ƯU KẾT QUẢ", use_container_width=True):
    with st.spinner("Đang tính toán ma trận Markov & Bộ lọc Entropy trên dữ liệu thời gian thực..."):
        df_b2, df_b3 = optimize_bac_2_3(binary_matrix)
        
        col1, col2 = st.columns(2)
        
        # Bảng Top Bậc 2
        with col1:
            st.subheader("🔥 TOP BẬC 2 TỐI ƯU (Highest EV)")
            st.caption("Mô hình Markov Vector | Chỉ vào tiền khi EV > 0")
            
            display_b2 = df_b2.copy()
            display_b2['Cặp Số'] = display_b2.apply(lambda r: f"[{int(r['Num1']):02d} - {int(r['Num2']):02d}]", axis=1)
            display_b2['Xác Suất'] = (display_b2['Prob'] * 100).map("{:.2f}%".format)
            display_b2['Điểm EV'] = display_b2['EV'].map("{:+.3f}".format)
            
            st.dataframe(
                display_b2[['Cặp Số', 'Xác Suất', 'Lift', 'Điểm EV']],
                use_container_width=True,
                hide_index=True
            )
            
        # Bảng Top Bậc 3
        with col2:
            st.subheader("🚀 TOP BẬC 3 TỐI ƯU (Highest EV)")
            st.caption("Tối ưu hóa kỳ vọng trúng thưởng 3/3 và 2/3")
            
            display_b3 = df_b3.copy()
            display_b3['Bộ 3 Số'] = display_b3['Tuple'].apply(lambda t: f"[{t[0]:02d} - {t[1]:02d} - {t[2]:02d}]")
            display_b3['Trúng 3/3'] = (display_b3['Prob_3_3'] * 100).map("{:.2f}%".format)
            display_b3['Điểm EV'] = display_b3['EV'].map("{:+.3f}".format)
            
            st.dataframe(
                display_b3[['Bộ 3 Số', 'Trúng 3/3', 'Điểm EV']],
                use_container_width=True,
                hide_index=True
            )

st.caption("🔄 Trạng thái: Dữ liệu tự động quét và làm mới mỗi 60 giây.")
