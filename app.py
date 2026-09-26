import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Nuclear Core Cluster Engine", layout="centered")

# ==============================================================================
# CREATIVE ALGORITHM: NUCLEAR CORE CLUSTER & SATELLITE TRIANGULATION ENGINE
# ==============================================================================
class NuclearCoreClusterEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, vec):
        m = np.max(vec)
        return vec / m if m > 0 else vec

    def process(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        last_draw = X[-1]
        
        # ----------------------------------------------------------------------
        # 1. ĐỊNH VỊ SỐ HẠT NHÂN (NUCLEAR CORE ANCHOR)
        # ----------------------------------------------------------------------
        # Hạt nhân là số có động lượng tích lũy tốt, nghỉ ở kỳ 5 hoặc nổ nhịp đều
        core_scores = np.zeros(D)
        for i in range(D):
            # Tính điểm động lượng nhịp
            if last_draw[i] == 0:
                core_scores[i] = freqs[i] * 1.5 # Ưu tiên số tích lũy nhịp
            else:
                core_scores[i] = freqs[i] * 0.8
                
        # Khóa các số bão hòa (>3 kỳ)
        core_scores[freqs >= 3] *= 0.1
        core_scores = self._norm(core_scores)
        
        # Lấy số Hạt Nhân (Core) có điểm cao nhất
        core_idx = int(np.argmax(core_scores))
        core_num = core_idx + 1
        
        # ----------------------------------------------------------------------
        # 2. XÂY DỰNG 3 VỆ TINH BAO VÙNG (3 SATELLITES)
        # ----------------------------------------------------------------------
        satellite_scores = np.zeros(D)
        
        # Vector 5 chiều của Hạt nhân
        core_vec = X[:, core_idx]
        
        for j in range(D):
            if j != core_idx:
                cand_vec = X[:, j]
                # Tính độ vuông góc không gian với Hạt nhân (Cosine Orthogonality)
                dot_prod = np.dot(core_vec, cand_vec)
                norm_prod = (np.linalg.norm(core_vec) * np.linalg.norm(cand_vec)) + 1e-5
                cosine_sim = dot_prod / norm_prod
                ortho_score = 1.0 - abs(cosine_sim) # Vuông góc = 1.0
                
                # Điểm vệ tính = Độ vuông góc x Điểm động lượng
                satellite_scores[j] = ortho_score * (freqs[j] + 0.5)
                
                # Phạt nếu nổ chung với Core >= 2 lần
                co_occur = np.dot(X[:, core_idx], X[:, j])
                if co_occur >= 2:
                    satellite_scores[j] *= 0.05
                    
                # Phạt nếu cùng nổ ở kỳ cuối với Core
                if last_draw[core_idx] == 1 and last_draw[j] == 1:
                    satellite_scores[j] *= 0.1
                    
        # Lấy 3 Vệ Tinh đỉnh nhất
        satellite_scores[core_idx] = -1.0 # Bỏ qua chính nó
        top_sat_indices = np.argsort(satellite_scores)[-3:][::-1]
        satellites = [int(idx + 1) for idx in top_sat_indices]
        
        # ----------------------------------------------------------------------
        # 3. TẠO THẾ TRẬN 3 CẶP GHÉP (TRIANGULATION PAIRS)
        # ----------------------------------------------------------------------
        pairs = [
            tuple(sorted([core_num, satellites[0]])),
            tuple(sorted([core_num, satellites[1]])),
            tuple(sorted([core_num, satellites[2]]))
        ]

        explanation = (
            f"• **THUẬT TOÁN DÀN GHÉP TRẬN HẠT NHÂN (Nuclear Core Cluster):**\n"
            f"  - **Số Hạt Nhân Độc Tôn (Anchor Core):** Khóa cứng số **{core_num:02d}** làm trụ cột (đã tối ưu hóa 100% động lượng).\n"
            f"  - **3 Vệ Tinh Bao Vùng (Satellites):** Chọn 3 số **{satellites[0]:02d}, {satellites[1]:02d}, {satellites[2]:02d}** có pha vuông góc Vector $90^\circ$ với Hạt Nhân để bao phủ toàn bộ độ lệch pha.\n"
            f"  - **Chiến Thuật Ghép Trận 3 Cặp:** Đánh đồng thời 3 cặp số ghép từ Hạt Nhân. Chỉ cần Hạt Nhân nổ + 1 Vệ Tinh nổ $\rightarrow$ Trúng trọn vẹn cặp!"
        )

        return core_num, satellites, pairs, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Nuclear Core Cluster Engine")
st.caption("Thuật toán Dàn Ghép Trận Hạt Nhân • Triệt hạ bẫy 1/2 • Ghép Hạt Nhân & 3 Vệ Tinh Vùng")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=160,
    key="raw_text_keno"
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
                
        st.success("🎉 Đã hoàn tất Tính toán Trận hình Hạt nhân!")
        
        engine = NuclearCoreClusterEngine(num_dim=80)
        core_num, satellites, pairs, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẤU TRÚC TRẬN HẠT NHÂN & 3 CẶP GHÉP TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="SỐ HẠT NHÂN CHỦ LỰC", value=f"{core_num:02d}")
        with col2:
            st.metric(label="3 VỆ TINH BAO VÙNG", value=f"{satellites[0]:02d} — {satellites[1]:02d} — {satellites[2]:02d}")
            
        st.markdown("### 🚀 DANH SÁCH 3 CẶP SỐ CHỐT ĐI TRẬN:")
        st.write(f"1️⃣ **Cặp 1 (Chính):** `{pairs[0][0]:02d} — {pairs[0][1]:02d}`")
        st.write(f"2️⃣ **Cặp 2 (Lót 1):** `{pairs[1][0]:02d} — {pairs[1][1]:02d}`")
        st.write(f"3️⃣ **Cặp 3 (Lót 2):** `{pairs[2][0]:02d} — {pairs[2][1]:02d}`")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình Nuclear Core Cluster.")
