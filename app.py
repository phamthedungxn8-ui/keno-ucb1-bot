import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. THUẬT TOÁN ĐỊNH LƯỢNG TRIẾT LÝ & ĐỘNG LỰC HỌC
# ==========================================

@st.cache_data(ttl=600)
def process_philosophical_quant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tối ưu hóa tính toán ma trận Động lực học & Triết lý con số.
    Sử dụng vectorized operations của pandas/numpy để đạt hiệu năng tối đa.
    """
    data = df.copy()

    # ----------------------------------------------------
    # Hướng 1: Nhịp Thở & Trạng Thái Năng Lượng (Energy Index)
    # Energy Index = c_gap / a_gap
    # ----------------------------------------------------
    data['energy_index'] = (data['c_gap'] / data['a_gap']).round(2)
    
    # Phân loại trạng thái hô hấp
    conditions_energy = [
        (data['energy_index'] < 0.8),
        (data['energy_index'] >= 0.8) & (data['energy_index'] < 1.0),
        (data['energy_index'] >= 1.0) & (data['energy_index'] < 1.3),
        (data['energy_index'] >= 1.3)
    ]
    choices_energy = [
        "🔵 Hít vào (Thong thả tích lũy)",
        "🟡 Nén lồng ngực (Chuẩn bị điểm nổ)",
        "🔴 Thở ra (Vùng căng cứng / Ưu tiên Sniper)",
        "⚠️ Quá tải năng lượng (Quá nhịp nén)"
    ]
    data['respiration_state'] = np.select(conditions_energy, choices_energy, default="Không xác định")

    # ----------------------------------------------------
    # Hướng 2: Trọng Lực & Lực Hấp Dẫn Cặp Số (Gravitational Score)
    # Gravity = (hits / a_gap) * (1 / (1 + (c_gap / m_gap)^2))
    # ----------------------------------------------------
    base_magnetism = data['hits'] / data['a_gap']
    gap_resistance = 1 + (data['c_gap'] / data['m_gap']) ** 2
    data['gravity_score'] = (base_magnetism / gap_resistance).round(2)

    # ----------------------------------------------------
    # Hướng 3: Vòng Đời & Điểm Gãy Chuyển Mùa (Life Cycle & Seasons)
    # Tỷ lệ tiến trình vòng đời = c_gap / m_gap
    # ----------------------------------------------------
    data['lifecycle_ratio'] = (data['c_gap'] / data['m_gap']).round(2)
    
    conditions_season = [
        (data['lifecycle_ratio'] < 0.3),
        (data['lifecycle_ratio'] >= 0.3) & (data['lifecycle_ratio'] < 0.65),
        (data['lifecycle_ratio'] >= 0.65) & (data['lifecycle_ratio'] < 0.90),
        (data['lifecycle_ratio'] >= 0.90)
    ]
    choices_season = [
        "🌸 Mùa Xuân (Sinh - Khởi đầu nhịp mới)",
        "☀️ Mùa Hạ (Trưởng - Điểm rơi phong độ)",
        "🍂 Mùa Thu (Thu hoạch - Dồn nén tối đa)",
        "❄️ Mùa Đông Buốt Giá (Ranh giới Bình Minh / Điểm Gãy)"
    ]
    data['season'] = np.select(conditions_season, choices_season, default="Không xác định")

    # Điểm tổng hợp Chiêm Nghiệm Quant Score (0 - 100)
    # Kết hợp trọng số: 40% Energy Index + 40% Gravity + 20% Lifecycle Ratio
    data['quant_artistry_score'] = (
        (np.clip(data['energy_index'], 0, 1.5) / 1.5) * 40 +
        (np.clip(data['gravity_score'], 0, 3.0) / 3.0) * 40 +
        (np.clip(data['lifecycle_ratio'], 0, 1.0)) * 20
    ).round(1)

    return data


# ==========================================
# 2. GIAO DIỆN STREAMLIT & ĐỒ HỌA TRỰC QUAN
# ==========================================

st.set_page_config(page_title="Keno Quant Philosophy Engine", layout="wide", page_icon="🌌")

st.title("🌌 KENO QUANT PHILOSOPHY ENGINE V4.0")
st.caption("Bộ lọc Động lực học Tự nhiên & Chiêm nghiệm Nhịp điệu Xác suất")

# File Upload Sidebar
uploaded_file = st.sidebar.file_uploader("Nạp tệp Excel/CSV (Chuẩn V4.0)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read Data
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    try:
        # Process Engine
        df_processed = process_philosophical_quant(raw_df)
        
        # Sidebar Controls
        st.sidebar.header("🎯 Bộ Lọc Động Lực")
        selected_zone = st.sidebar.multiselect("Lọc Phân Vùng", options=df_processed['zone'].unique(), default=df_processed['zone'].unique())
        min_score = st.sidebar.slider("Ngưỡng Điểm Chiêm Nghiệm (Quant Artistry)", 0.0, 100.0, 50.0)

        filtered_df = df_processed[(df_processed['zone'].isin(selected_zone)) & (df_processed['quant_artistry_score'] >= min_score)]

        # --- TAB NAVIGATION ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🫁 1. Nhịp Thở Năng Lượng", 
            "🌌 2. Lực Hấp Dẫn Cặp Số", 
            "🔄 3. Vòng Đời Chuyển Mùa",
            "📊 Bảng Tổng Hợp Chiêm Nghiệm"
        ])

        # ----------------------------------------------------
        # TAB 1: NHỊP THỜ NĂNG LƯỢNG
        # ----------------------------------------------------
        with tab1:
            st.subheader("🫁 Phân Tích Mức Độ Nén Năng Lượng (Energy Index)")
            st.markdown("""
            * **Energy Index ($c\_gap / a\_gap$):** Đo lường mức độ nín thở của cặp số. 
            * Khi chỉ số vượt ngưỡng **1.0**, lồng ngực năng lượng đã căng đầy và tiến vào **vùng thở ra (nổ) bắt buộc**.
            """)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig_energy = px.bar(
                    filtered_df.sort_values(by="energy_index", ascending=False).head(15),
                    x="pair", y="energy_index",
                    color="energy_index",
                    color_continuous_scale="Reds",
                    title="Top 15 Cặp Số Có Độ Nén Năng Lượng Cao Nhất",
                    labels={"energy_index": "Tỷ lệ nén (Energy Index)", "pair": "Cặp số"}
                )
                fig_energy.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Ngưỡng thở ra chuẩn (1.0)")
                st.plotly_chart(fig_energy, use_container_width=True)

            with col2:
                st.write("### Trạng Thái Hô Hấp")
                st.dataframe(
                    filtered_df[['pair', 'energy_index', 'respiration_state']]
                    .sort_values(by='energy_index', ascending=False),
                    hide_index=True, use_container_width=True
                )

        # ----------------------------------------------------
        # TAB 2: LỰC HẤP DẪN CẶP SỐ
        # ----------------------------------------------------
        with tab2:
            st.subheader("🌌 Bản Đồ Trọng Lực & Hố Đen Xác Suất")
            st.markdown("""
            Bản đồ phản ánh lực hút hấp dẫn của các cặp số. Cặp số nằm ở **góc trên bên trái** (Tần suất nổ dày & Trọng lực cao) tạo ra trường hấp dẫn mạnh nhất.
            """)

            fig_gravity = px.scatter(
                filtered_df,
                x="a_gap", y="gravity_score",
                size="hits", color="zone",
                hover_name="pair",
                text="pair",
                title="Ma Trận Lực Hấp Dẫn (Gravity Score vs Bước Xổ)",
                labels={"a_gap": "Bước Xổ Trung Bình (a_gap)", "gravity_score": "Điểm Trọng Lực (Gravity Score)"}
            )
            fig_gravity.update_traces(textposition='top center')
            st.plotly_chart(fig_gravity, use_container_width=True)

        # ----------------------------------------------------
        # TAB 3: VÒNG ĐỜI CHUYỂN MÙA
        # ----------------------------------------------------
        with tab3:
            st.subheader("🔄 Chu Kỳ Vòng Đời & Điểm Gãy Chuyển Mùa")
            st.markdown("""
            Khi `c_gap` tiến sát `m_gap` (Tỷ lệ tiến trình $\rightarrow 1.0$), cặp số đang ở **Mùa Đông Buốt Giá**. Đây không phải cạm bẫy, mà là thời điểm **Bình Minh Hồi Sinh** theo quy luật tự nhiên.
            """)

            fig_season = px.sunburst(
                filtered_df,
                path=['season', 'zone', 'pair'],
                values='quant_artistry_score',
                color='quant_artistry_score',
                color_continuous_scale='Magma',
                title="Phân Bố Cặp Số Theo 4 Mùa Tự Nhiên"
            )
            st.plotly_chart(fig_season, use_container_width=True)

        # ----------------------------------------------------
        # TAB 4: BẢNG TỔNG HỢP CHIÊM NGHIỆM
        # ----------------------------------------------------
        with tab4:
            st.subheader("📋 Bảng Ma Trận Định Lượng Chiêm Nghiệm Hoàn Chỉnh")
            # Chuyển cột pair sang dạng chuỗi
filtered_df['pair'] = filtered_df['pair'].astype(str)

            # Format display
            display_df = filtered_df[[
                'pair', 'zone', 'quant_artistry_score', 
                'energy_index', 'respiration_state', 
                'gravity_score', 'season', 'c_gap', 'a_gap', 'm_gap'
            ]].sort_values(by='quant_artistry_score', ascending=False)

            st.dataframe(
                display_df.style.background_gradient(subset=['quant_artistry_score', 'energy_index'], cmap='YlOrRd'),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"❌ Lỗi cấu trúc dữ liệu tệp nạp vào: {str(e)}")
        st.info("💡 Đảm bảo tệp nạp vào có đầy đủ các cột: pair, zone, hits, c_gap, a_gap, m_gap")
else:
    st.info("👆 Vui lòng tải tệp `data_keno_exact_spec.csv` hoặc `.xlsx` ở bước trước lên thanh bên (Sidebar) để bắt đầu phân tích.")
