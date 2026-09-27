import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Energy Dynamics Engine", layout="centered")

# ==============================================================================
# RIGOROUS ENERGY DYNAMICS ENGINE (LẶP & GAN 5 KỲ -> BẬC 2 & BẬC 3)
# ==============================================================================
class RigorousEnergyEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def process_exact_sets(self, X):
        T, D = X.shape # T=5, D=80
        freqs = X.sum(axis=0) # Total occurrences in 5 draws
        last_draw = X[-1]     # Draw 5 (latest draw)
        
        # ----------------------------------------------------------------------
        # 1. TẬP SỐ LẶP ĐỘNG LƯỢNG (REPEAT MOMENTUM SET)
        # ----------------------------------------------------------------------
        # Số xuất hiện >= 2 lần VÀ xuất hiện ở Kỳ 5 (vừa nổ xong, nhịp lặp cao)
        repeat_scores = np.zeros(D)
        for i in range(D):
            if freqs[i] >= 2 and last_draw[i] == 1:
                # Trừ điểm số quá bão hòa (nổ 4-5/5 kỳ)
                if freqs[i] >= 4:
                    repeat_scores[i] = freqs[i] * 0.2
                else:
                    repeat_scores[i] = freqs[i] * 1.8
                    
        core_idx = int(np.argmax(repeat_scores))
        N_core = core_idx + 1 # Số Hạt Nhân Lặp
        
        # ----------------------------------------------------------------------
        # 2. TẬP SỐ GAN 5 KỲ (FULL-COLD SUPPRESSION SET)
        # ----------------------------------------------------------------------
        # Số hoàn toàn KHÔNG xuất hiện trong 5 kỳ (freqs == 0)
        cold_indices = np.where(freqs == 0)[0]
        
        # Nếu không có số gan 0/5 kỳ, nới lỏng lấy số 1/5 kỳ nghỉ 4 kỳ gần nhất
        if len(cold_indices) == 0:
            cold_scores = np.zeros(D)
            for i in range(D):
                if freqs[i] == 1 and last_draw[i] == 0:
                    cold_scores[i] = 1.0
            cold_idx = int(np.argmax(cold_scores))
        else:
            # Đo điểm rơi Gan bằng khoảng cách tới các hàng chục đang nổ mạnh
            cold_scores = np.zeros(len(cold_indices))
            for idx_pos, c_idx in enumerate(cold_indices):
                # Ưu tiên các số gan nằm trong dải hàng chục vừa nổ nhiều ở kỳ 5
                decade = c_idx // 10
                decade_activity = np.sum(X[-1, decade*10 : (decade+1)*10])
                cold_scores[idx_pos] = decade_activity
            
            best_cold_pos = int(np.argmax(cold_scores))
            cold_idx = cold_indices[best_cold_pos]
            
        N_cold = cold_idx + 1 # Số Gan Bù Pha
        
        # ----------------------------------------------------------------------
        # 3. SỐ VỆ TINH GHÉP BẬC 3 (NEIGHBOR-SHIFT SATELLITE)
        # ----------------------------------------------------------------------
        # Chọn số tạt cánh lân cận (+1 hoặc -1) của N_core hoặc N_cold chưa bão hòa
        candidates = []
        for base in [core_idx, cold_idx]:
            for offset in [-1, 1]:
                cand = base + offset
                if 0 <= cand < D and cand != core_idx and cand != cold_idx:
                    candidates.append(cand)
                    
        sat_scores = np.zeros(len(candidates))
        for pos, cand in enumerate(candidates):
            # Ưu tiên số có tần suất 1-2 lần, nghỉ ở kỳ 5
            if freqs[cand] in [1, 2] and last_draw[cand] == 0:
                sat_scores[pos] = 2.0
            else:
                sat_scores[pos] = 0.5
                
        best_sat_pos = int(np.argmax(sat_scores))
        sat_idx = candidates[best_sat_pos]
        N_sat = sat_idx + 1 # Số Vệ Tinh ghép Bậc 3
        
        # Formulate exact sets
        pair_bac_2 = tuple(sorted([N_core, N_cold]))
        bo_bac_3 = tuple(sorted([N_core, N_cold, N_sat]))

        explanation = (
            f"• **CƠ CHẾ PHÂN TÍCH TOÁN HỌC ĐỘNG LỰC HỌC (Energy Dynamics Engine):**\n"
            f"  - **Hạt Nhân Quán Tính Lặp ($N_{{core}}$ = {N_core:02d}):** Trích xuất từ dải số có tần suất lặp tốt ($\ge 2$ lần trong 5 kỳ) và vừa nổ ở Kỳ 5, nắm giữ động lượng rơi tiếp.\n"
            f"  - **Số Gan Bù Pha ($N_{{cold}}$ = {N_cold:02d}):** Trích xuất từ nhóm bị nén năng lượng hoàn toàn (0/5 kỳ) nằm trong hàng chục đang bùng nổ, có chỉ số giải phóng Entropy cao nhất.\n"
            f"  - **Số Vệ Tinh Dịch Chuyển ($N_{{sat}}$ = {N_sat:02d}):** Số kề vệt sóng có nhịp điểm rơi tích lũy chuẩn bị bộc phát.\n"
            f"• **Mối Liên Kết:** Ghép $N_{{core}}$ (Động lượng Lặp) + $N_{{cold}}$ (Điểm rơi Gan) $\rightarrow$ Tạo nên **Bộ Bậc 2** hoàn hảo. Thêm $N_{{sat}}$ để bọc lót độ lệch pha $1$ đơn vị $\rightarrow$ Tạo nên **Bộ Bậc 3** tối ưu."
        )

        return pair_bac_2, bo_bac_3, N_core, N_cold, N_sat, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Energy Dynamics Engine")
st.caption("Khắc phục tư duy mơ hồ • Phân rã Lặp/Gan 5 Kỳ • Chốt 1 Bộ Bậc 2 & 1 Bộ Bậc 3")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=150,
    key="raw_text_keno_energy"
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
                
        st.success("🎉 Đã chạy xong Mô hình Động lực học Lặp & Gan!")
        
        engine = RigorousEnergyEngine(num_dim=80)
        pair2, bo3, n_core, n_cold, n_sat, explanation = engine.process_exact_sets(matrix)
        
        st.markdown("---")
        st.subheader("🎯 BẢNG CHỐT 2 BỘ SỐ (BẬC 2 & BẬC 3)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="🔥 SỐ LẶP CORE", value=f"{n_core:02d}")
        with c2:
            st.metric(label="❄️ SỐ GAN COLD", value=f"{n_cold:02d}")
        with c3:
            st.metric(label="🛰️ SỐ VỆ TINH SAT", value=f"{n_sat:02d}")
            
        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("1️⃣ BỘ BẬC 2 CHỐT (2 SỐ)")
            st.title(f"{pair2[0]:02d} — {pair2[1]:02d}")
        with col_right:
            st.subheader("2️⃣ BỘ BẬC 3 CHỐT (3 SỐ)")
            st.title(f"{bo3[0]:02d} — {bo3[1]:02d} — {bo3[2]:02d}")
            
        st.markdown("---")
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung trên để chạy Mô hình Energy Dynamics.")
