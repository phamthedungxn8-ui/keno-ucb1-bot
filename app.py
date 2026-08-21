import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

st.set_page_config(page_title="Keno Quant V4.0", layout="wide", page_icon="🎯")

st.title("🎯 HỆ THỐNG KENO QUANT ENGINE V4.0")
st.caption("Bộ lọc Định lượng & Quản trị vốn Sniper 3 kỳ")

# Sidebar cấu hình
st.sidebar.header("⚙️ Cấu hình Tham số")
nav_capital = st.sidebar.number_input("Tổng Vốn (VNĐ)", value=50000000, step=5000000)
kelly_fraction = st.sidebar.slider("Hệ số Kelly", 0.05, 0.50, 0.10)

# KHO NẠP DỮ LIỆU
st.subheader("📥 Nạp Dữ Liệu Phân Tích")
uploaded_file = st.file_uploader("Tải tệp CSV dữ liệu Keno của bạn lên đây", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
        st.success("✅ Đã tải dữ liệu thành công!")
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
else:
    st.info("💡 Chưa tải file? Dưới đây là dữ liệu mẫu mặc định:")
    # Dữ liệu mặc định nếu không upload file
    df_input = pd.DataFrame([
        {"pair": "[17, 56]", "hits": 15, "c_gap": 14, "a_gap": 12.00, "m_gap": 44, "zone": "CROSS (Bắc cầu)"},
        {"pair": "[30, 67]", "hits": 14, "c_gap": 4,  "a_gap": 15.00, "m_gap": 66, "zone": "CROSS (Bắc cầu)"},
        {"pair": "[12, 13]", "hits": 13, "c_gap": 14, "a_gap": 12.00, "m_gap": 44, "zone": "LOW (01-25)"},
        {"pair": "[54, 68]", "hits": 13, "c_gap": 5,  "a_gap": 10.00, "m_gap": 38, "zone": "HIGH (51-80)"},
        {"pair": "[53, 57]", "hits": 13, "c_gap": 12, "a_gap": 11.11, "m_gap": 45, "zone": "HIGH (51-80)"},
    ])

# BỘ TÍNH TOÁN
results = []
for idx, row in df_input.iterrows():
    p_val = stats.binomtest(int(row["hits"]), n=100, p=0.060126, alternative='greater').pvalue
    gap_ratio = row["c_gap"] / row["a_gap"] if row["a_gap"] > 0 else 0
    breakout = gap_ratio * (1 - row["c_gap"]/row["m_gap"]) * np.log(row["hits"])
    
    p_emp = row["hits"] / 100.0
    b = 8.0
    full_k = (p_emp * (b + 1) - 1) / b
    bet = max(50000, int(np.round((nav_capital * max(0, full_k * kelly_fraction)) / 50000.0)) * 50000)
    
    results.append({
        "Cặp Số": row["pair"],
        "Phân Vùng": row["zone"],
        "Số Lần Trúng": row["hits"],
        "Gap Ratio": round(gap_ratio, 2),
        "Điểm Breakout": round(breakout, 2),
        "Tiền Cược Gợi Ý": f"{bet:,} VNĐ"
    })

df_res = pd.DataFrame(results)

st.subheader("📊 Kết Quả Lắng Lọc Tự Động")
st.dataframe(df_res, use_container_width=True)
