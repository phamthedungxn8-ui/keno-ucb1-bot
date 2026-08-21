import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

st.set_page_config(page_title="Keno Quant Engine V4.0", layout="wide", page_icon="🎯")

st.title("🎯 HỆ THỐNG KENO QUANT ENGINE V4.0")
st.caption("Bộ lọc Định lượng & Quản trị vốn Sniper tự động")

# Sidebar cấu hình Quản lý vốn
st.sidebar.header("⚙️ Cấu hình Quản trị Vốn")
nav_capital = st.sidebar.number_input("Tổng Vốn NAV (VNĐ)", value=50000000, step=5000000)
kelly_fraction = st.sidebar.slider("Hệ số Kelly", 0.05, 0.50, 0.10)

st.subheader("📥 Nạp Dữ Liệu Phân Tích")
uploaded_file = st.file_uploader("Tải tệp Excel / CSV dữ liệu Keno lên đây", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
            
        results = []
        for idx, row in df_input.iterrows():
            hits = int(row["hits"])
            c_gap = float(row["c_gap"])
            a_gap = float(row["a_gap"])
            m_gap = float(row["m_gap"])
            
            # Tính toán các chỉ số Quant
            gap_ratio = c_gap / a_gap if a_gap > 0 else 0
            breakout = gap_ratio * (1 - c_gap / m_gap) * np.log(hits) if m_gap > 0 else 0
            
            # Chiến lược Quản lý vốn Kelly
            p_emp = hits / 100.0
            b = 8.0  # Tỷ lệ thưởng Keno Bậc 2
            full_k = (p_emp * (b + 1) - 1) / b
            bet = max(50000, int(np.round((nav_capital * max(0, full_k * kelly_fraction)) / 50000.0)) * 50000)
            
            results.append({
                "Cặp Số": row["pair"],
                "Phân Vùng": row["zone"],
                "Số Lần Trúng (Hits)": hits,
                "Chờ Nổ (c_gap)": int(c_gap),
                "Gap Ratio": round(gap_ratio, 2),
                "Điểm Breakout": round(breakout, 2),
                "Tiền Cược Gợi Ý": bet,
                "Hiển Thị Tiền": f"{bet:,} VNĐ"
            })

        # Sắp xếp tự động theo Điểm Breakout từ cao xuống thấp
        df_res = pd.DataFrame(results).sort_values(by="Điểm Breakout", ascending=False).reset_index(drop=True)

        # ---------------------------------------------------------------------
        # HIỂN THỊ KẾT QUẢ TOP 1-2-3 (CARD NỔI BẬT)
        # ---------------------------------------------------------------------
        st.subheader("🔥 TOP BỘ SỐ SÁNG GIÁ NHẤT (SNIPER TARGETS)")
        col1, col2, col3 = st.columns(3)
        
        top1 = df_res.iloc[0]
        top2 = df_res.iloc[1]
        top3 = df_res.iloc[2]
        
        with col1:
            st.metric(label="🥇 TOP 1 (Breakout Cao Nhất)", value=str(top1["Cặp Số"]), delta=f"Score: {top1['Điểm Breakout']}")
            st.write(f"• Phân vùng: **{top1['Phân Vùng']}**")
            st.write(f"• Đã chờ: **{top1['Chờ Nổ (c_gap)']} kỳ**")
            st.write(f"• Cược gợi ý: **{top1['Hiển Thị Tiền']}**")

        with col2:
            st.metric(label="🥈 TOP 2", value=str(top2["Cặp Số"]), delta=f"Score: {top2['Điểm Breakout']}")
            st.write(f"• Phân vùng: **{top2['Phân Vùng']}**")
            st.write(f"• Đã chờ: **{top2['Chờ Nổ (c_gap)']} kỳ**")
            st.write(f"• Cược gợi ý: **{top2['Hiển Thị Tiền']}**")

        with col3:
            st.metric(label="🥉 TOP 3", value=str(top3["Cặp Số"]), delta=f"Score: {top3['Điểm Breakout']}")
            st.write(f"• Phân vùng: **{top3['Phân Vùng']}**")
            st.write(f"• Đã chờ: **{top3['Chờ Nổ (c_gap)']} kỳ**")
            st.write(f"• Cược gợi ý: **{top3['Hiển Thị Tiền']}**")

        # ---------------------------------------------------------------------
        # BẢNG DỮ LIỆU ĐẦY ĐỦ
        # ---------------------------------------------------------------------
        st.subheader("📊 Bảng Chi Tiết Tất Cả Các Cặp Số")
        df_display = df_res.drop(columns=["Tiền Cược Gợi Ý"])
        st.dataframe(df_display, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Cấu trúc file không hợp lệ. Lỗi: {e}")
else:
    st.info("💡 Vui lòng tải tệp data_keno.xlsx hoặc data_keno.csv lên để xem kết quả phân tích.")
