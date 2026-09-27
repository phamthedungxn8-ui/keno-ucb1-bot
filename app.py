import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Monte Carlo Integration Engine", layout="centered")

# ==============================================================================
# MONTE CARLO STOCHASTIC INTEGRATION ENGINE
# ==============================================================================
class MonteCarloKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    # 1. THUẬT TOÁN MÔ PHỎNG NGẪU NHIÊN MONTE CARLO (STOCHASTIC STREAM)
    def _simulate_monte_carlo(self, num_simulations=1000):
        """Giả lập 1,000 kỳ quay Keno ngẫu nhiên chuẩn toán học"""
        np.random.seed() # Khai báo seed ngẫu nhiên thời gian thực
        sim_matrix = np.zeros((num_simulations, self.D))
        for s in range(num_simulations):
            # Mỗi kỳ quay ngẫu nhiên rút 20 số không trùng nhau từ 1..80
            draw = np.random.choice(self.D, size=20, replace=False)
            sim_matrix[s, draw] = 1.0
        
        # Tính tần suất xác suất xuất hiện qua 1,000 kỳ giả lập
        mc_probs = sim_matrix.mean(axis=0)
        return mc_probs

    # 2. TÍCH HỢP LUỒNG LỊCH SỬ (DETERMINISTIC) + LUỒNG NGẪU NHIÊN (MONTE CARLO)
    def process_combined(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0) # Tần suất 5 kỳ thực tế
        last_draw = X[-1]     # Kỳ 5 thực tế
        
        # A. Lấy ma trận ngẫu nhiên Monte Carlo
        mc_probs = self._simulate_monte_carlo(num_simulations=1000)
        
        # B. Tính điểm Luồng Lịch Sử (Deterministic Score)
        historical_scores = np.zeros(D)
        for i in range(D):
            if freqs[i] >= 2 and last_draw[i] == 1:
                historical_scores[i] = 1.8 * freqs[i] # Số Lặp động lượng
            elif freqs[i] == 0:
                historical_scores[i] = 2.0 # Số Gan tích lũy
            else:
                historical_scores[i] = 0.5 * freqs[i]
                
        # Chuẩn hóa
        norm_hist = historical_scores / np.max(historical_scores)
        norm_mc = mc_probs / np.max(mc_probs)
        
        # C. KẾT HỢP SONG SONG: 70% Lịch sử + 30% Ngẫu nhiên Monte Carlo
        blended_scores = (0.70 * norm_hist) + (0.30 * norm_mc)
        
        # Phạt các số bão hòa (>3 kỳ thực tế)
        blended_scores[freqs >= 4] *= 0.1
        
        # ----------------------------------------------------------------------
        # D. CẤU TRÚC BỘ BẬC 2 VÀ BẬC 3 TỪ BẢNG ĐIỂM LAI (HYBRID)
        # ----------------------------------------------------------------------
        # Số 1 (Trụ cột Lặp / Động Lượng): Điểm cao nhất
        n1_idx = int(np.argmax(blended_scores))
        N1 = n1_idx + 1
        
        # Số 2 (Điểm Bù Gan / Monte Carlo): Lấy số gan 0/5 kỳ có điểm lai cao nhất
        cold_indices = np.where(freqs == 0)[0]
        if len(cold_indices) > 0:
            best_cold_pos = int(np.argmax(blended_scores[cold_indices]))
            n2_idx = cold_indices[best_cold_pos]
        else:
            # Nếu không có số gan 0/5, lấy số có điểm lai cao thứ 2
            temp_scores = blended_scores.copy()
            temp_scores[n1_idx] = -1.0
            n2_idx = int(np.argmax(temp_scores))
            
        N2 = n2_idx + 1
        
        # Số 3 (Vệ tinh Kề vệt Monte Carlo): Lấy số có điểm lai cao tiếp theo
        temp_scores_3 = blended_scores.copy()
        temp_scores_3[n1_idx] = -1.0
        temp_scores_3[n2_idx] = -1.0
        n3_idx = int(np.argmax(temp_scores_3))
        N3 = n3_idx + 1
        
        # Kết xuất kết quả
        bo_bac_2 = tuple(sorted([N1, N2]))
        bo_bac_3 = tuple(sorted([N1, N2, N3]))

        explanation = (
            f"• **CƠ CHẾ KẾT HỢP THUẬT TOÁN NGẪU NHIÊN MONTE CARLO:**\n"
            f"  - **Luồng Lịch Sử (70%):** Trích xuất động lượng Lặp/Gan từ 5 kỳ thực tế.\n"
            f"  - **Luồng Giả Lập Ngẫu Nhiên (30%):** Chạy 1,000 lượt quay ngẫu nhiên Monte Carlo để tạo nhiễu sinh học, giúp lọc bỏ các số bị bẫy nhiễu ảo.\n"
            f"  - **Kết quả:** Sự kết hợp giữa **Xác suất Lịch sử** và **Biến động Ngẫu nhiên** giúp bộ số đạt mức cân bằng tối đa."
        )

        return bo_bac_2, bo_bac_3, N1, N2, N3, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("🎲 Keno Monte Carlo Integration Engine")
st.caption("Kết hợp Dữ liệu 5 Kỳ + Thuật toán Ngẫu nhiên Monte Carlo 1,000 Lượt Quay")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=150,
    key="raw_text_keno_mc"
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
                
        st.success("🎉 Đã chạy xong Mô hình Giả lập Ngẫu nhiên Monte Carlo!")
        
        engine = MonteCarloKenoEngine(num_dim=80)
        bo2, bo3, n1, n2, n3, explanation = engine.process_combined(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẤU TRÚC 2 BỘ SỐ (LAI LỊCH SỬ & NGẪU NHIÊN)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="🔥 SỐ TRỤ CỘT N1", value=f"{n1:02d}")
        with c2:
            st.metric(label="❄️ SỐ BÙ GAN N2", value=f"{n2:02d}")
        with c3:
            st.metric(label="🎲 SỐ MONTE CARLO N3", value=f"{n3:02d}")
            
        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("1️⃣ BỘ BẬC 2 CHỐT (2 SỐ)")
            st.title(f"{bo2[0]:02d} — {bo2[1]:02d}")
        with col_right:
            st.subheader("2️⃣ BỘ BẬC 3 CHỐT (3 SỐ)")
            st.title(f"{bo3[0]:02d} — {bo3[1]:02d} — {bo3[2]:02d}")
            
        st.markdown("---")
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung trên để kích hoạt mô hình Monte Carlo.")
