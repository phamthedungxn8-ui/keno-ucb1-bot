import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from main import KenoQuantumEngine, optimize_bac_2_3

# 1. Cấu hình trang Streamlit
st.set_page_config(
    page_title="AlphaVietlott Direct Quantum Engine v4.0",
    page_icon="🎯",
    layout="wide"
)

# 2. Cập nhật nhịp Auto-Refresh theo kỳ Keno: 500,000 ms = 8.3 phút (Sai số 1-2 phút)
st_autorefresh(interval=500000, key="keno_kiquay_refresher")

st.title("🎯 ALPHAVIETLOTT DIRECT QUANTUM ENGINE v4.0")
st.caption("Cập nhật theo nhịp kỳ quay Keno (10 phút/kỳ) | Cửa sổ trượt 500 kỳ")

# 3. Thanh điều hướng cấu hình
st.sidebar.header("⚙️ CẤU HÌNH HỆ THỐNG")
n_draws_input = st.sidebar.slider("Số kỳ phân tích (Cửa sổ trượt):", min_value=100, max_value=1000, value=500, step=50)

engine = KenoQuantumEngine(draw_count=n_draws_input)

# Cache dữ liệu ngắn hạn theo đúng nhịp kỳ quay (300 giây = 5 phút)
@st.cache_data(ttl=300)
def load_realtime_data(draws_count):
    return engine.fetch_live_data()

# Nạp dữ liệu ma trận & Mã kỳ quay
binary_matrix, latest_draw_id = load_realtime_data(n_draws_input)

# 4. BẢNG THÔNG TIN TRẠNG THÁI THỜI GIAN THỰC
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric(label="📌 MÃ KỲ QUAY MỚI NHẤT", value=str(latest_draw_id))

with col_info2:
    st.metric(label="📊 DỮ LIỆU CỬA SỔ TRUỘT", value=f"{binary_matrix.shape[0]} Kỳ")

with col_info3:
    current_time = datetime.now().strftime("%H:%M:%S")
    st.metric(label="⏱️ LẦN CẬP NHẬT GẦN NHẤT", value=current_time)

st.divider()

# 5. ĐIỀU KHIỂN & CHẠY THUẬT TOÁN
col_btn1, col_btn2 = st.columns([1, 2])

with col_btn1:
    # Nút ép cập nhật thủ công nếu nhà cái trả kết quả sớm hơn nhịp tự động
    if st.button("🔄 TẢI KỲ MỚI NGAY", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col_btn2:
    btn_run = st.button("🔥 CHẠY THUẬT TOÁN TỐI ƯU KẾT QUẢ", use_container_width=True)

if btn_run:
    with st.spinner("Đang tính toán ma trận Markov & Entropy trên dữ liệu kỳ mới nhất..."):
        df_b2, df_b3 = optimize_bac_2_3(binary_matrix)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 TOP BẬC 2 TỐI ƯU (Highest EV)")
            display_b2 = df_b2.copy()
            display_b2['Cặp Số'] = display_b2.apply(lambda r: f"[{int(r['Num1']):02d} - {int(r['Num2']):02d}]", axis=1)
            display_b2['Xác Suất'] = (display_b2['Prob'] * 100).map("{:.2f}%".format)
            display_b2['Điểm EV'] = display_b2['EV'].map("{:+.3f}".format)
            
            st.dataframe(
                display_b2[['Cặp Số', 'Xác Suất', 'Lift', 'Điểm EV']],
                use_container_width=True,
                hide_index=True
            )
            
        with col2:
            st.subheader("🚀 TOP BẬC 3 TỐI ƯU (Highest EV)")
            display_b3 = df_b3.copy()
            display_b3['Bộ 3 Số'] = display_b3['Tuple'].apply(lambda t: f"[{t[0]:02d} - {t[1]:02d} - {t[2]:02d}]")
            display_b3['Trúng 3/3'] = (display_b3['Prob_3_3'] * 100).map("{:.2f}%".format)
            display_b3['Điểm EV'] = display_b3['EV'].map("{:+.3f}".format)
            
            st.dataframe(
                display_b3[['Bộ 3 Số', 'Trúng 3/3', 'Điểm EV']],
                use_container_width=True,
                hide_index=True
            )

st.caption("🔄 Trạng thái: Hệ thống tự động làm mới theo chu kỳ kỳ quay (~8 phút/lần).")
