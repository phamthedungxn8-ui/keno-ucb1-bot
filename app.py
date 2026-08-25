import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. THUẬT TOÁN ĐỊNH LƯỢNG NÂNG CẤP (Z-SCORE & REAL-TIME TIMING)
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

    # Xử lý an toàn các cột số
    if 'c_gap' not in data.columns: data['c_gap'] = 1.0
    if 'a_gap' not in data.columns: data['a_gap'] = 1.0
    if 'hits' not in data.columns: data['hits'] = 1.0
    if 'std_gap' not in data.columns: data['std_gap'] = 2.0  # Độ lệch chuẩn mặc định nếu thiếu
    
    if 'max_gap' not in data.columns:
        if 'm_gap' in data.columns:
            data['max_gap'] = data['m_gap']
        else:
            data['max_gap'] = (data['c_gap'] * 1.5).replace(0, 1)

    for col in ['c_gap', 'a_gap', 'max_gap', 'hits', 'std_gap']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(1.0)

    data['a_gap'] = data['a_gap'].replace(0, 1.0)
    data['max_gap'] = data['max_gap'].replace(0, 1.0)
    data['std_gap'] = data['std_gap'].replace(0, 1.0)

    # 1. Nhịp Thở Năng Lượng (Energy Index = c_gap / a_gap)
    data['energy_index'] = (data['c_gap'] / data['a_gap']).round(2)

    # 2. TÍNH CHỈ SỐ Z-SCORE GẦN ĐỈNH NÉN (Độ lệch chuẩn thống kê)
    data['z_score'] = ((data['c_gap'] - data['a_gap']) / data['std_gap']).round(2)

    # 3. Trạng Thái Hô Hấp & Khung Cửa Sổ Timing
    conditions_timing = [
        (data['z_score'] >= 1.8),
        (data['z_score'] >= 1.0) & (data['z_score'] < 1.8),
        (data['energy_index'] >= 0.8) & (data['z_score'] < 1.0),
        (data['energy_index'] < 0.8)
    ]
    choices_timing = [
        "🔥 ĐIỂM NỔ CỰC ĐẠI (Đánh 1-3 Kỳ / Ưu tiên số 1)",
        "🟡 Nén Căng Cứng (Vào Watchlist / Chờ Tín hiệu)",
        "🔵 Đang Tích Lũy (Bỏ qua - Chưa nên nuôi)",
        "⚪ An Toàn / Chu Kỳ Mới"
    ]
    data['timing_status'] = np.select(conditions_timing, choices_timing, default="⚪ Chưa xác định")

    # 4. Trọng Lực Cặp Số
    base_magnetism = data['hits'] / data['a_gap']
    gap_resistance = 1 + (data['c_gap'] / data['a_gap'])
    data['gravity_score'] = (base_magnetism / gap_resistance).round(2)

    # 5. Vòng Đời Chuyển Mùa
    data['lifecycle_ratio'] = (data['c_gap'] / data['max_gap']).round(2)

    # Quant Score tổng hợp
    data['quant_artistry_score'] = (
        (np.clip(data['energy_index'], 0, 2) / 2.0 * 30) +
        (np.clip(data['z_score'], -1, 3) / 3.0 * 40) +  # Trọng số Z-Score cao nhất (40%)
        (np.clip(data['gravity_score'], 0, 5) / 5.0 * 30)
    ).round(1)

    return data


# ==============================================================================
# 2. GIAO DIỆN STREAMLIT v6.0
# ==============================================================================

st.set_page_config(page_title="Keno Quant Engine v6.0 - Realtime Timing", layout="wide")

st.title("⚡ KENO QUANT TIMING ENGINE v6.0")
st.caption("Bộ lọc Định lượng Bắt Điểm Nổ Tức thì & Quản trị Khung Cửa Sổ 3 Kỳ")

uploaded_file = st.sidebar.file_uploader("Nạp tệp Excel/CSV", type=["csv", "xlsx"])
st.sidebar.header("📥 Nạp Dữ Liệu")
input_method = st.sidebar.radio("Chọn cách nhập dữ liệu:", ["Dán văn bản trực tiếp", "Tải file (CSV/XLSX)"])

raw_df = None

if input_method == "Dán văn bản trực tiếp":
    text_data = st.sidebar.text_area("Dán nội dung từ Ghi chú vào đây:", height=200)
    if text_data.strip():
        import io
        raw_df = pd.read_csv(io.StringIO(text_data))
else:
    uploaded_file = st.sidebar.file_uploader("Nạp tệp Excel/CSV", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

if raw_df is not None:
    try:
        df_processed = process_philosophical_quant(raw_df)

        # ----------------------------------------------------------------------
        # BỘ LỌC KẾT QUẢ KỲ VỪA RA (REAL-TIME TRIGGER STREAM)
        # ----------------------------------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Real-Time Trigger (Kỳ Vừa Ra)")
        last_draw_input = st.sidebar.text_input(
            "Nhập các số kỳ vừa ra (cách nhau dấu phẩy):", 
            value="", 
            help="Ví dụ: 04, 15, 37, 52, 69"
        )
        
        triggered_pairs = []
        if last_draw_input.strip():
            drawn_numbers = [n.strip().zfill(2) for n in last_draw_input.split(",") if n.strip()]
            st.sidebar.success(f"Đã ghi nhận {len(drawn_numbers)} số kỳ vừa ra.")
            
            # Lọc các cặp số có chứa ít nhất 1 số mồi vừa xuất hiện
            for idx, row in df_processed.iterrows():
                p_str = str(row['pair'])
                if any(num in p_str for num in drawn_numbers):
                    triggered_pairs.append(row['pair'])

        st.sidebar.header("🎯 Lọc Bổ Sung")
        all_zones = df_processed['zone'].unique().tolist() if 'zone' in df_processed.columns else []
        selected_zone = st.sidebar.multiselect("Lọc Phân Vùng", options=all_zones, default=all_zones)

        filtered_df = df_processed.copy()
        if selected_zone and 'zone' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['zone'].isin(selected_zone)]

        # --- GIAO DIỆN CHÍNH ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⚡ 1. Point-of-Impact (Bắt Điểm Nổ)",
            "🔗 2. Real-Time Trigger (Số Mồi)",
            "🛡️ 3. Quy Tắc Cửa Sổ 3 Kỳ",
            "🧮 4. Phân Bổ Vốn Kelly",
            "📊 Full Data Matrix"
        ])

        # --- TAB 1: BẮT ĐIỂM NỔ Z-SCORE ---
        with tab1:
            st.subheader("🎯 Ma Trận Timing Z-Score (Loại bỏ gồng nuôi dài)")
            st.markdown("""
            * **Z-Score $\ge$ 1.8 (Vùng Đỏ):** Cặp số đã vượt quá 95% khoảng cách nén lịch sử. **Đây là lúc xuất tiền, khung đánh 1-3 kỳ.**
            * **Z-Score < 1.0 (Vùng Xanh/Trắng):** Tuyệt đối **KHÔNG NUÔI**, dù c_gap có cao đến đâu.
            """)
            filtered_df["bubble_size"] = filtered_df["quant_artistry_score"].apply(lambda x: max(float(x), 0.1) * 10 + 5)
            fig_z = px.scatter(
                filtered_df,
                x="energy_index", y="z_score",
                color="timing_status",
                size="bubble_size",
                hover_name="pair", text="pair",
                title="Bản Đồ Điểm Điểm Bùng Nổ Tức Thời (Energy vs Z-Score)",
                labels={"energy_index": "Tỷ lệ nén (c_gap/a_gap)", "z_score": "Căng Cứng Thống Kê (Z-Score)"}
            )
            fig_z.add_hline(y=1.8, line_dash="dash", line_color="red", annotation_text="Ngưỡng Nổ Bắt Buộc (Z >= 1.8)")
            fig_z.add_vline(x=1.0, line_dash="dash", line_color="orange", annotation_text="Ngưỡng Thở Ra (1.0)")
            fig_z.update_traces(textposition='top center')
            st.plotly_chart(fig_z, use_container_width=True)

        # --- TAB 2: REALTIME TRIGGER ---
        with tab2:
            st.subheader("🔗 Lọc Số Mồi Từ Kỳ Vừa Quay (Trigger Catalyst)")
            if triggered_pairs:
                trigger_df = filtered_df[filtered_df['pair'].isin(triggered_pairs)].sort_values(by="quant_artistry_score", ascending=False)
                st.success(f"🔥 Tìm thấy **{len(trigger_df)}** cặp số hội tụ Tín Hiệu Kích Hoạch từ kỳ vừa ra!")
                
                disp_trig = [c for c in ['pair', 'quant_artistry_score', 'z_score', 'timing_status', 'c_gap', 'a_gap'] if c in trigger_df.columns]
                st.dataframe(trigger_df[disp_trig], use_container_width=True, hide_index=True)
            else:
                st.info("👈 Nhập kết quả các số ở kỳ vừa ra tại thanh bên trái để kích hoạt ma trận Số Mồi (Real-Time Catalyst).")

        # --- TAB 3: QUY TẮC CỬA SỔ 3 KỲ ---
        with tab3:
            st.subheader("🛡️ Khung Kỷ Luật 3 Kỳ & Cắt Lỗ Tự Động")
            st.warning("⚠️ **QUY TẮC BẮT BUỘC:** Chỉ giao dịch tối đa 3 kỳ cho 1 Tín hiệu. Kỳ thứ 3 không nổ ➔ CẮT LỖ NGAY!")

            top3_entry = filtered_df[filtered_df['z_score'] >= 1.5].sort_values(by="quant_artistry_score", ascending=False).head(5)

            if not top3_entry.empty:
                st.write("### 🚀 Danh Sách Cặp Số Đủ Điều Kiện Vào Tiền Ngay Kỳ Tế Báo:")
                for i, row in top3_entry.iterrows():
                    with st.expander(f"📌 Cặp số: **{row['pair']}** | Điểm Quant: **{row['quant_artistry_score']}** | Z-Score: **{row['z_score']}**"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Kỳ 1 (Xuất Vốn Ban Đầu)", "30% Ngân Sách", "Vào Ngay")
                        col_b.metric("Kỳ 2 (Nén Tiếp Nút)", "40% Ngân Sách", "Nổ ➔ Dừng")
                        col_c.metric("Kỳ 3 (Khung Cuối)", "30% Ngân Sách", "Không Nổ ➔ CUT LOSS")
            else:
                st.info("Hiện chưa có cặp số nào chạm ngưỡng Z-Score căng cứng (>= 1.5). Hãy kiên nhẫn đứng ngoài quan sát!")

        # --- TAB 4: QUẢN TRỊ VỐN KELLY ---
        with tab4:
            st.subheader("🧮 Phân Bổ Vốn Tối Ưu Tốc Độ")
            total_cap = st.number_input("Tổng vốn dành cho khung 3 kỳ (VNĐ):", min_value=100000, value=1000000, step=100000)
            
            top_k = filtered_df.sort_values(by="quant_artistry_score", ascending=False).head(3).copy()
            if not top_k.empty:
                top_k['alloc_percent'] = (top_k['quant_artistry_score'] / top_k['quant_artistry_score'].sum()).round(2)
                top_k['budget_vnd'] = (top_k['alloc_percent'] * total_cap).round(-3)
                
                st.dataframe(
                    top_k[['pair', 'quant_artistry_score', 'z_score', 'alloc_percent', 'budget_vnd']].rename(columns={
                        'pair': 'Cặp số',
                        'quant_artistry_score': 'Điểm Quant',
                        'z_score': 'Z-Score',
                        'alloc_percent': 'Tỷ lệ phân bổ',
                        'budget_vnd': 'Số tiền cược (VNĐ)'
                    }),
                    use_container_width=True, hide_index=True
                )

        # --- TAB 5: BẢNG DỮ LIỆU TỔNG HỢP ---
        with tab5:
            st.subheader("📋 Ma Trận Định Lượng Toàn Phần")
            show_cols = ['pair', 'quant_artistry_score', 'z_score', 'timing_status', 'energy_index', 'gravity_score', 'c_gap', 'a_gap']
            avail = [c for c in show_cols if c in filtered_df.columns]
            st.dataframe(filtered_df[avail].sort_values(by="quant_artistry_score", ascending=False), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Lỗi xử lý dữ liệu: {str(e)}")

else:
    st.info("👉 Vui lòng tải tệp Excel/CSV lên thanh công cụ bên trái để bắt đầu phân tích.")
