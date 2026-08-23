import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==============================================================================
# 1. THUẬT TOÁN ĐỊNH LƯỢNG TRIẾT LÝ & ĐỘNG LỰC HỌC
# ==============================================================================

@st.cache_data(ttl=600)
def process_philosophical_quant(df: pd.DataFrame):
    data = df.copy()
    
    # Chuẩn hóa tên cột
    data.columns = [str(c).strip().lower() for c in data.columns]
    
    # Chuẩn hóa cột pair
    if 'pair' in data.columns:
        data['pair'] = data['pair'].astype(str).str.replace(r'\.0$', '', regex=True)
        data['pair'] = data['pair'].apply(lambda x: x.zfill(2) if x.isdigit() and len(x) == 1 else x)
    else:
        data['pair'] = [f"P_{i+1}" for i in range(len(data))]

    # Tự tạo/xử lý an toàn cho các cột nếu tệp CSV thiếu
    if 'c_gap' not in data.columns: data['c_gap'] = 1.0
    if 'a_gap' not in data.columns: data['a_gap'] = 1.0
    if 'hits' not in data.columns: data['hits'] = 1.0
    
    if 'max_gap' not in data.columns:
        if 'm_gap' in data.columns:
            data['max_gap'] = data['m_gap']
        else:
            data['max_gap'] = (data['c_gap'] * 1.5).replace(0, 1)

    # Ép kiểu số
    for col in ['c_gap', 'a_gap', 'max_gap', 'hits']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(1.0)

    # Đảm bảo không chia cho 0
    data['a_gap'] = data['a_gap'].replace(0, 1.0)
    data['max_gap'] = data['max_gap'].replace(0, 1.0)

    # 1. Nhịp Thở Năng Lượng (Energy Index = c_gap / a_gap)
    data['energy_index'] = (data['c_gap'] / data['a_gap']).round(2)

    conditions_energy = [
        (data['energy_index'] < 0.8),
        (data['energy_index'] >= 0.8) & (data['energy_index'] < 1.0),
        (data['energy_index'] >= 1.0) & (data['energy_index'] < 1.3),
        (data['energy_index'] >= 1.3)
    ]
    choices_energy = [
        "🔵 Hít vào (Thong thả tích lũy)",
        "🟡 Nén lồng ngực (Chuẩn bị điểm nổ)",
        "🔴 Thở ra (Vùng căng cứng / Ưu tiên nổ)",
        "⚠️ Quá tải năng lượng (Quá nhịp nén)"
    ]
    data['respiration_state'] = np.select(conditions_energy, choices_energy, default="⚪ Chưa xác định")

    # 2. Trọng Lực Cặp Số (Gravity Score)
    base_magnetism = data['hits'] / data['a_gap']
    gap_resistance = 1 + (data['c_gap'] / data['a_gap'])
    data['gravity_score'] = (base_magnetism / gap_resistance).round(2)

    # 3. Vòng Đời Chuyển Mùa (Lifecycle Ratio = c_gap / max_gap)
    data['lifecycle_ratio'] = (data['c_gap'] / data['max_gap']).round(2)

    conditions_season = [
        (data['lifecycle_ratio'] < 0.3),
        (data['lifecycle_ratio'] >= 0.3) & (data['lifecycle_ratio'] < 0.65),
        (data['lifecycle_ratio'] >= 0.65) & (data['lifecycle_ratio'] < 0.90),
        (data['lifecycle_ratio'] >= 0.90)
    ]
    choices_season = [
        "🌸 Mùa Xuân (Sinh - Khởi đầu nhịp nén)",
        "☀️ Mùa Hạ (Trưởng - Điểm rơi phong độ)",
        "🍂 Mùa Thu (Thu hoạch - Dồn nén tối đa)",
        "❄️ Mùa Đông Buốt Giá (Ranh giới chuyển mùa / Bỏ qua)"
    ]
    data['season'] = np.select(conditions_season, choices_season, default="⚪ Chưa rõ")

    # Điểm tổng hợp Chiêm Nghiệm Quant Score
    data['quant_artistry_score'] = (
        (np.clip(data['energy_index'], 0, 2) / 2.0 * 40) +
        (np.clip(data['gravity_score'], 0, 5) / 5.0 * 35) +
        (np.clip(data['lifecycle_ratio'], 0, 1) * 25)
    ).round(1)

    return data


# ==============================================================================
# 2. GIAO DIỆN STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Keno Quant Philosophy Engine", layout="wide")

st.title("🌌 KENO QUANT PHILOSOPHY ENGINE")
st.caption("Bộ lọc Động lực học Tự nhiên & Định lượng Cặp số Keno")

uploaded_file = st.sidebar.file_uploader("Nạp tệp Excel/CSV (Chuẩn V4.0)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    try:
        df_processed = process_philosophical_quant(raw_df)

        st.sidebar.header("🎯 Bộ Lọc Động Lực")
        all_zones = df_processed['zone'].unique().tolist() if 'zone' in df_processed.columns else []
        selected_zone = st.sidebar.multiselect("Lọc Phân Vùng", options=all_zones, default=all_zones)
        min_score = st.sidebar.slider("Ngưỡng Điểm Chiêm Nghiệm", 0.0, 100.0, 0.0)

        filtered_df = df_processed.copy()
        if selected_zone and 'zone' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['zone'].isin(selected_zone)]
        filtered_df = filtered_df[filtered_df['quant_artistry_score'] >= min_score]

        tab1, tab2, tab3, tab4 = st.tabs([
            "🫁 1. Nhịp Thở Năng Lượng",
            "🌌 2. Lực Hấp Dẫn Cặp Số",
            "🔄 3. Vòng Đời Chuyển Mùa",
            "📊 Bảng Tổng Hợp Chiêm Nghiệm"
        ])

        with tab1:
            st.subheader("🫁 Phân Tích Mức Độ Nén Năng Lượng (Energy Index)")
            col1, col2 = st.columns([2, 1])
            with col1:
                top15_energy = filtered_df.sort_values(by="energy_index", ascending=False).head(15)
                fig_energy = px.bar(
                    top15_energy,
                    x="pair", y="energy_index",
                    color="energy_index",
                    color_continuous_scale="Magma",
                    title="Top 15 Cặp Số Có Độ Nén Năng Lượng Cao Nhất",
                    labels={"energy_index": "Tỷ lệ nén", "pair": "Cặp số"}
                )
                fig_energy.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Ngưỡng thở ra chuẩn (1.0)")
                fig_energy.update_xaxes(type='category')
                st.plotly_chart(fig_energy, use_container_width=True)

            with col2:
                st.write("### Trạng Thái Hô Hấp")
                cols_disp = [c for c in ['pair', 'respiration_state', 'energy_index'] if c in filtered_df.columns]
                st.dataframe(filtered_df[cols_disp].sort_values(by='energy_index', ascending=False), hide_index=True, use_container_width=True)

        with tab2:
            st.subheader("🌌 Bản Đồ Trọng Lực & Lực Hút Cặp Số (Gravity Score)")
            fig_gravity = px.scatter(
                filtered_df,
                x="a_gap", y="gravity_score",
                size="hits" if 'hits' in filtered_df.columns else None,
                color="zone" if 'zone' in filtered_df.columns else None,
                hover_name="pair", text="pair",
                title="Ma Trận Lực Hấp Dẫn Cặp Số",
                labels={"a_gap": "Bước Nhảy Trung Bình (a_gap)", "gravity_score": "Điểm Trọng Lực (Gravity)"}
            )
            fig_gravity.update_traces(textposition='top center')
            fig_gravity.update_xaxes(type='category')
            st.plotly_chart(fig_gravity, use_container_width=True)

        with tab3:
            st.subheader("🔄 Phân Bố Cặp Số Theo Vòng Đời Tự Nhiên")
            path_cols = [c for c in ['season', 'zone', 'pair'] if c in filtered_df.columns]
            if len(path_cols) >= 2:
                fig_season = px.sunburst(
                    filtered_df, path=path_cols,
                    values='quant_artistry_score', color='quant_artistry_score',
                    color_continuous_scale='Magma', title="Phân Bố Cặp Số Theo 4 Mùa Tự Nhiên"
                )
                st.plotly_chart(fig_season, use_container_width=True)

        with tab4:
            st.subheader("📋 Bảng Ma Trận Định Lượng Chiêm Nghiệm")
            req_cols = ['pair', 'zone', 'quant_artistry_score', 'energy_index', 'respiration_state', 'gravity_score', 'season', 'c_gap', 'a_gap']
            avail_cols = [c for c in req_cols if c in filtered_df.columns]
            display_df = filtered_df[avail_cols].sort_values(by='quant_artistry_score', ascending=False)
            
            # Hiển thị dataframe sạch không dùng background_gradient
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Lỗi xử lý dữ liệu: {str(e)}")

else:
    st.info("👉 Vui lòng tải tệp Excel/CSV lên thanh công cụ bên trái để bắt đầu phân tích.")
