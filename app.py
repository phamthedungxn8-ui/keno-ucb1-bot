import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# Cấu hình trang web Streamlit
st.set_page_config(page_title="Keno Quant V4.0", layout="wide", page_icon="🎯")

st.title("🎯 HỆ THỐNG KENO QUANT ENGINE V4.0")
st.caption("Bộ lọc Định lượng & Quản trị vốn Sniper 3 kỳ")

# Sidebar cấu hình tham số
st.sidebar.header("⚙️ Cấu hình Tham số")
nav_capital = st.sidebar.number_input("Tổng Vốn (VNĐ)", value=50000000, step=5000000)
kelly_fraction = st.sidebar.slider("Hệ số Kelly (Tỉ lệ cược)", 0.05, 0.50, 0.10)

# Dữ liệu gốc
data = [
    {"pair": "[17, 56]", "hits": 15, "c_gap": 14, "a_gap": 12.00, "m_gap": 44, "zone": "CROSS (Bắc cầu)"},
    {"pair": "[30, 67]", "hits": 14, "c_gap": 4,  "a_gap": 15.00, "m_gap": 66, "zone": "CROSS (Bắc cầu)"},
    {"pair": "[12, 13]", "hits": 13, "c_gap": 14, "a_gap": 12.00, "m_gap": 44, "zone": "LOW (01-25)"},
    {"pair": "[54, 68]", "hits": 13, "c_gap": 5,  "a_gap": 10.00, "m_gap": 38, "zone": "HIGH (51-80)"},
    {"pair": "[53, 57]", "hits": 13, "c_gap": 12, "a_gap": 11.11, "m_gap": 45, "zone": "HIGH (51-80)"},
]

# Xử lý tính toán
results = []
for item in data:
    p_val = stats.binomtest(item["hits"], n=100, p=0.060126, alternative='greater').pvalue
    gap_ratio = item["c_gap"] / item["a_gap"]
    breakout = gap_ratio * (1 - item["c_gap"]/item["m_gap"]) * np.log(item["hits"])
    
    # Kelly
    p_emp = item["hits"] / 100.0
    b = 8.0 # Lợi nhuận x9 Bậc 2
    full_k = (p_emp * (b + 1) - 1) / b
    bet = max(50000, int(np.round((nav_capital * max(0, full_k * kelly_fraction)) / 50000.0)) * 50000)
    
    results.append({
        "Cặp Số": item["pair"],
        "Phân Vùng": item["zone"],
        "Số Lần Trúng (100 Kỳ)": item["hits"],
        "Gap Ratio (Điểm Rơi)": round(gap_ratio, 2),
        "Điểm Breakout": round(breakout, 2),
        "Gợi Ý Tiền Cược": f"{bet:,} VNĐ"
    })

df = pd.DataFrame(results)

st.subheader("📊 Danh Sách Cặp Số Tối Ưu Lắng Lọc")
st.dataframe(df, use_container_width=True)

st.success("✅ Hệ thống khuyến nghị: Chỉ đánh Sniper tối đa 3 kỳ tại đúng điểm rơi chín!")
