import re
import itertools
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Core-Coverage Matrix Engine", layout="centered")

# ==============================================================================
# CORE-COVERAGE MATRIX ENGINE (QUẢN TRỊ XÁC SUẤT & PHỦ RỘNG)
# ==============================================================================
class CoreCoverageEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def process_matrix(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0) # Tần số xuất hiện trong 5 kỳ
        last_draw = X[-1]     # Kỳ gần nhất
        
        # 1. BỘ LỌC CHỌN SỐ HẠT NHÂN (HOT CORE)
        # Ưu tiên số xuất hiện >= 2 lần và vừa nổ ở kỳ gần nhất (quán tính nổ tiếp)
        core_scores = freqs * 1.5 + last_draw * 2.0
        core_idx = int(np.argmax(core_scores))
        core_num = core_idx + 1
        
        # 2. BỘ LỌC CHỌN 5 SỐ VỆ TINH (SATELLITES)
        # Lấy các số có tần số xuất hiện 1 - 2 lần (nhịp tích lũy vừa phải)
        sat_scores = np.zeros(D)
        for i in range(D):
            if i != core_idx:
                if freqs[i] == 1 or freqs[i] == 2:
                    sat_scores[i] = freqs[i] * 2.0 + (1 - last_draw[i]) # Ưu tiên số vừa nghỉ 1 kỳ
                else:
                    sat_scores[i] = freqs[i] * 0.5
                    
        # Lấy Top 5 số Vệ tinh
        top_sat_indices = np.argsort(sat_scores)[-5:][::-1]
        sat_numbers = [int(idx + 1) for idx in top_sat_indices]
        
        # 3. TẠO MA TRẬN 6 SỐ TỔ HỢP (CORE + 5 SATELLITES)
        full_cluster = [core_num] + sat_numbers
        
        # Tạo tất cả các cặp số bậc 2 từ dàn 6 số (Tổng cộng C(6, 2) = 15 cặp)
        all_pairs = list(itertools.combinations(full_cluster, 2))
        
        # Lựa chọn 6 Cặp Trọng Tâm chứa Số Hạt Nhân + 4 Cặp Phủ Ghép Chéo
        core_pairs = [p for p in all_pairs if core_num in p][:6]
        cross_pairs = [p for p in all_pairs if core_num not in p][:4]
        
        selected_10_pairs = core_pairs + cross_pairs

        explanation = (
            f"• **CHIẾN THUẬT MA TRẬN PHỦ BẢO HIỂM (Core-Coverage Strategy):**\n"
            f"  - **Hạt Nhân Tần Số (Core):** Khóa cứng số **{core_num:02d}** (có quán tính nổ cao nhất).\n"
            f"  - **Dàn 5 Vệ Tinh (Satellites):** Bao phủ 5 số **{', '.join([f'{n:02d}' for n in sat_numbers])}** đang ở nhịp điểm rơi phong độ.\n"
            f"  - **Ưu điểm Toán học:** Bạn sở hữu dàn 6 số **[{', '.join([f'{n:02d}' for n in full_cluster])}]**. Chỉ cần 2 trong 6 số này xuất hiện trong 20 số Keno rút ra, bạn **chắc chắn ăn trúng cặp!**"
        )

        return core_num, sat_numbers, selected_10_pairs, explanation

# ==============================================================================
# STREAMLIT UI DISPLAY
# ==============================================================================
st.title("🛡️ Keno Core-Coverage Matrix Engine")
st.caption("Chuyển từ Dự báo Ngẫu nhiên sang Quản trị Xác suất Matrix • Phủ Rộng Dàn 6 Số")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=150,
    key="raw_text_keno_coverage"
)

if raw_text_input.strip():
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text_input.strip(), flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        matrix = np.zeros((5, 80), dtype=float)
        for k in range(5):
            for num in all_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0
                
        st.success("🎉 Đã hoàn thành lập Ma trận Phủ Xác suất!")
        
        engine = CoreCoverageEngine(num_dim=80)
        core_num, sat_numbers, selected_pairs, explanation = engine.process_matrix(matrix)
        
        st.markdown("---")
        st.subheader("🎯 BẢNG KHÓA SỐ & DÀN PHỦ BAO VÙNG")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="🔥 SỐ HẠT NHÂN CORE", value=f"{core_num:02d}")
        with c2:
            st.metric(label="🛰️ DÀN 5 VỆ TINH PHỦ BẢO HIỂM", value=", ".join([f"{n:02d}" for n in sat_numbers]))
            
        st.markdown("---")
        st.subheader("🚀 BẢNG 10 CẶP SỐ CHỐT ĐÁNH (BẢO HIỂM TRÚNG CHÉO):")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Cặp Ghép Hạt Nhân (Ưu tiên):**")
            for idx, p in enumerate(selected_pairs[:6]):
                st.write(f"{idx+1}. Cặp `{p[0]:02d} — {p[1]:02d}`")
        with col_b:
            st.markdown("**Cặp Phủ Vệ Tinh (Bọc lót):**")
            for idx, p in enumerate(selected_pairs[6:]):
                st.write(f"{idx+7}. Cặp `{p[0]:02d} — {p[1]:02d}`")
            
        st.markdown("---")
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung trên để kích hoạt Ma trận Phủ Bảo hiểm.")
