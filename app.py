import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
import re
from itertools import combinations

# 1. Cấu hình Trang Streamlit
st.set_page_config(page_title="Hệ Thống Phân Tích Cấu Trúc XSMB (OCR 3 Kỳ)", layout="wide")

st.title("🔬 Hệ Thống Phân Tích Động Lực Học Phi Tuyến XSMB (OCR 3 Kỳ)")
st.caption("Trích xuất Bảng giải -> Nhúng Trọng số Không gian -> Tính Toán Lô Xiên Cộng Hưởng")

# 2. Định nghĩa Trọng số Giải (Spatial Weights)
WEIGHTS = {
    'GĐB': 3.5,
    'G1': 2.5,
    'G2': 2.0,
    'G3': 1.5,
    'G4': 1.2,
    'G5': 1.0,
    'G6': 0.8,
    'G7': 2.2
}

# 3. Hàm Tiền Xử Lý Ảnh & OCR Trích Xuất Cấu Trúc Giải
def ocr_extract_prizes(image):
    # Chuyển ảnh sang OpenCV format
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Tăng cường tương phản & Khử nhiễu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR Tesseract (Cấu hình chỉ lấy số và khoảng trắng/dấu gạch)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-\n '
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    # Trích xuất toàn bộ các số có 2 đến 5 chữ số
    raw_numbers = re.findall(r'\b\d{2,5}\b', text)
    
    # Áp dụng Bảng Phân Bổ Số Lượng Giải Chuẩn XSMB (27 số)
    # GĐB(1), G1(1), G2(2), G3(6), G4(4), G5(6), G6(3), G7(4)
    parsed_matrix = {
        'GĐB': raw_numbers[0:1] if len(raw_numbers) >= 1 else ["00000"],
        'G1':  raw_numbers[1:2] if len(raw_numbers) >= 2 else ["00000"],
        'G2':  raw_numbers[2:4] if len(raw_numbers) >= 4 else ["00000", "00000"],
        'G3':  raw_numbers[4:10] if len(raw_numbers) >= 10 else ["00000"]*6,
        'G4':  raw_numbers[10:14] if len(raw_numbers) >= 14 else ["0000"]*4,
        'G5':  raw_numbers[14:20] if len(raw_numbers) >= 20 else ["0000"]*6,
        'G6':  raw_numbers[20:23] if len(raw_numbers) >= 23 else ["000"]*3,
        'G7':  raw_numbers[23:27] if len(raw_numbers) >= 27 else ["00"]*4,
    }
    return parsed_matrix

# 4. Thuật Toán Phân Tích Cấu Trúc Động Lực Học (Xiên 2 / Xiên 3)
def analyze_3_periods_structure(period_data_list):
    # period_data_list: [dict_T2, dict_T1, dict_T0]
    
    # Trích xuất cặp số (2 chữ số cuối) cho từng giải và từng kỳ
    pair_weights = {}
    
    for t_idx, period in enumerate(period_data_list):
        decay_factor = (t_idx + 1) / 3.0  # Trọng số thời gian tăng dần: T-2 (0.33), T-1 (0.66), T (1.0)
        
        for g_name, s_list in period.items():
            w_g = WEIGHTS.get(g_name, 1.0)
            
            for s in s_list:
                if len(s) >= 2:
                    pair = s[-2:]  # Lấy 2 số cuối (đuôi lô)
                    # Điểm tích lũy = Trọng số giải * Trọng số thời gian
                    score = w_g * decay_factor
                    pair_weights[pair] = pair_weights.get(pair, 0.0) + score

    # Tính Toán Chỉ Số Tương Tác Lô Xiên (Xiên 2 & Xiên 3)
    all_pairs = list(pair_weights.keys())
    xien_2_scores = []
    xien_3_scores = []

    # Xiên 2
    for p1, p2 in combinations(all_pairs, 2):
        if p1 != p2:
            s1, s2 = pair_weights[p1], pair_weights[p2]
            # Chỉ số cộng hưởng năng lượng S_ij = (s1 + s2) * (1 + min(s1,s2)/max(s1,s2))
            s_ij = (s1 + s2) * (1.0 + min(s1, s2) / (max(s1, s2) + 1e-5))
            xien_2_scores.append({
                "Cặp Xiên 2": f"{p1} - {p2}",
                "Năng Lượng Tương Tác (S_ij)": round(s_ij, 2),
                "Đồng Bộ Pha (PLV)": round(min(s1, s2) / (max(s1, s2) + 1e-5), 2),
                "Đánh Giá": "Cộng hưởng cao" if s_ij > 8.0 else "Dao động ổn định"
            })

    # Xiên 3
    for p1, p2, p3 in combinations(all_pairs, 3):
        if len({p1, p2, p3}) == 3:
            s1, s2, s3 = pair_weights[p1], pair_weights[p2], pair_weights[p3]
            s_ijk = (s1 + s2 + s3) * 1.5
            xien_3_scores.append({
                "Bộ Xiên 3": f"{p1} - {p2} - {p3}",
                "Năng Lượng Tương Tác (S_ijk)": round(s_ijk, 2),
                "Chỉ Số Bùng Nổ": round(s_ijk * 1.2, 2),
                "Đánh Giá": "Golden Triad Attractor" if s_ijk > 12.0 else "Liên kết trung bình"
            })

    df_x2 = pd.DataFrame(xien_2_scores).sort_values(by="Năng Lượng Tương Tác (S_ij)", ascending=False)
    df_x3 = pd.DataFrame(xien_3_scores).sort_values(by="Năng Lượng Tương Tác (S_ijk)", ascending=False)
    
    return df_x2, df_x3

# 5. Giao Diện Upload 3 Ảnh (3 Kỳ Quay)
st.subheader("📸 Upload Ảnh Bảng Giải 3 Kỳ Liên Tiếp")
col1, col2, col3 = st.columns(3)

with col1:
    img_file_t2 = st.file_uploader("Kỳ T-2 (Xa nhất)", type=["jpg", "png", "jpeg"])
with col2:
    img_file_t1 = st.file_uploader("Kỳ T-1 (Kỳ trước)", type=["jpg", "png", "jpeg"])
with col3:
    img_file_t0 = st.file_uploader("Kỳ T (Mới nhất)", type=["jpg", "png", "jpeg"])

if img_file_t2 and img_file_t1 and img_file_t0:
    st.success("Đã nhận đủ 3 ảnh! Đang kích hoạt OCR & Trích xuất Ma trận Giải...")
    
    img_t2 = Image.open(img_file_t2)
    img_t1 = Image.open(img_file_t1)
    img_t0 = Image.open(img_file_t0)
    
    # Thực hiện OCR
    data_t2 = ocr_extract_prizes(img_t2)
    data_t1 = ocr_extract_prizes(img_t1)
    data_t0 = ocr_extract_prizes(img_t0)
    
    # Hiển thị Ma trận Giải trích xuất
    with st.expander("🔍 Xem Bảng Ma Trận Giải Sau Khi Trích Xuất OCR", expanded=False):
        st.json({"Kỳ T-2": data_t2, "Kỳ T-1": data_t1, "Kỳ T (Mới nhất)": data_t0})
    
    # Phân tích & Đánh giá Động lực học
    df_x2, df_x3 = analyze_3_periods_structure([data_t2, data_t1, data_t0])
    
    st.markdown("---")
    st.subheader("🎯 KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP (ƯU TIÊN LÔ XIÊN)")
    
    tab1, tab2 = st.tabs(["Top Xiên 2 Cộng Hưởng Cực Đại", "Top Xiên 3 Tam Giác Vàng"])
    
    with tab1:
        st.dataframe(df_x2.head(10), use_container_width=True)
        top_1_x2 = df_x2.iloc[0]["Cặp Xiên 2"]
        st.info(f"💡 **Khuyến nghị Xiên 2 tối ưu nhất:** `{top_1_x2}` (Năng lượng tương tác cực đại trên trục GĐB-G7)")
        
    with tab2:
        st.dataframe(df_x3.head(10), use_container_width=True)
        top_1_x3 = df_x3.iloc[0]["Bộ Xiên 3"]
        st.success(f"🔥 **Khuyến nghị Xiên 3 tối ưu nhất:** `{top_1_x3}` (Cụm điểm hút năng lượng - Attractor Cluster)")
else:
    st.warning("Vui lòng tải lên đủ 3 ảnh đại diện cho 3 kỳ quay liên tiếp để kích hoạt thuật toán.")
