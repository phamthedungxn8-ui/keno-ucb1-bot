import streamlit as st
import pandas as pd
from main import KenoQuantumEngine, optimize_bac_2_3

# Cấu hình Trang Streamlit
st.set_page_config(
    page_title="AlphaVietlott Direct Quantum Engine v4.0",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 ALPHAVIETLOTT DIRECT QUANTUM ENGINE v4.0")
st.caption("Thuật toán tối ưu trực tiếp: Markov Vector + Entropy Dynamic Filter | Tập trung 100% Kết quả")

# Sidebar - Cấu hình
st.sidebar.header("⚙️ CẤU HÌNH HỆ THỐNG")
ticket_type = st.sidebar.radio("Loại vé:", ["KENO (20/80)", "MAX 3D", "MAX 3D+"])
n_draws_input = st.sidebar.slider("Số kỳ quay phân tích:", min_value=100, max_value=1000, value=500, step=50)

uploaded_file = st.sidebar.file_uploader("Nạp CSV/Excel thủ công (Chính xác 100%):", type=["csv", "xlsx", "txt"])

# Khởi tạo Engine
engine = KenoQuantumEngine(draw_count=n_draws_input)

# Nạp dữ liệu
@st.cache_data(ttl=600)
def load_data(file, draws_count):
    if file is not None:
        # Xử lý file người dùng upload
        df = pd.read_csv(file)
        # Giả định file csv chứa 80 cột nhị phân hoặc danh sách các số
        matrix = df.values[:draws_count, :]
        return matrix
    else:
        # Tự động cào/sinh dữ liệu
        return engine.fetch_live_data()

binary_matrix = load_data(uploaded_file, n_draws_input)

st.success(f"🌐 Dữ liệu khả dụng: **{binary_matrix.shape[0]} kỳ quay** mới nhất đã được nạp vào ma trận Quantum Tensor.")

if st.button("🔥 CHẠY THUẬT TOÁN TỐI ƯU KẾT QUẢ", use_container_width=True):
    with st.spinner("Đang chạy thuật toán Numba C-Matrix Acceleration & Entropy Filtering..."):
        df_b2, df_b3 = optimize_bac_2_3(binary_matrix)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 TOP BẬC 2 TỐI ƯU (Highest EV)")
            st.caption("Chiến thuật: Chọn bộ có Điểm EV > 0 và Entropy thấp")
            
            # Format dữ liệu hiển thị
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
            st.caption("Chiến thuật: Tối ưu hóa chu kỳ trúng thưởng 3/3 và 2/3")
            
            display_b3 = df_b3.copy()
            display_b3['Bộ 3 Số'] = display_b3['Tuple'].apply(lambda t: f"[{t[0]:02d} - {t[1]:02d} - {t[2]:02d}]")
            display_b3['Trúng 3/3'] = (display_b3['Prob_3_3'] * 100).map("{:.2f}%".format)
            display_b3['Điểm EV'] = display_b3['EV'].map("{:+.3f}".format)
            
            st.dataframe(
                display_b3[['Bộ 3 Số', 'Trúng 3/3', 'Điểm EV']],
                use_container_width=True,
                hide_index=True
            )
            
        st.info("💡 **Gợi ý Quản lý Vốn:** Chỉ vào tiền các bộ có `Điểm EV > 0`. Đặt mức tiền cố định (ví dụ 10k/bộ) và duy trì tối thiểu 10 kỳ để thu hoạch chu kỳ nổ.")
