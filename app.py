import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Co-Emission Network Engine", layout="centered")

# ==============================================================================
# KENO CO-EMISSION & NEIGHBOR-SHIFT ENGINE
# ==============================================================================
class CoEmissionKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, mat):
        m = np.max(mat)
        return mat / m if m > 0 else mat

    # --------------------------------------------------------------------------
    # 1. CO-OCCURRENCE CONDITIONAL MATRIX (MA TRẬN XÁC SUẤT ĐỒNG PHÁT)
    # --------------------------------------------------------------------------
    def _compute_co_occurrence(self, X):
        """Tính ma trận tần suất và xác suất điều kiện xuất hiện cùng nhau"""
        T, D = X.shape
        # Ma trận đếm số lần cặp (i, j) cùng xuất hiện
        co_matrix = np.dot(X.T, X)
        np.fill_diagonal(co_matrix, 0)
        
        # Quyết định xác suất điều kiện P(j | i)
        freq_i = X.sum(axis=0)
        freq_i[freq_i == 0] = 1.0
        
        cond_prob = co_matrix / freq_i[:, None]
        # Làm đối xứng ma trận cộng hưởng 2 chiều
        co_resonance = (cond_prob + cond_prob.T) / 2.0
        
        return self._norm(co_resonance)

    # --------------------------------------------------------------------------
    # 2. OVER-SATURATION & NEIGHBOR SHIFT GATE (LỌC BÃO HÒA & DỊCH CHUYỂN LÂN CẬN)
    # --------------------------------------------------------------------------
    def _apply_saturation_and_shift(self, X, co_mat):
        """Khử các số bão hòa (>3 kỳ) và dịch chuyển sang số kề bên"""
        T, D = X.shape
        freqs = X.sum(axis=0)
        adjusted_mat = co_mat.copy()
        
        for i in range(D):
            # Nếu số i đã nổ quá 3/5 kỳ -> Bão hòa năng lượng
            if freqs[i] >= 3:
                # Phạt nặng việc chọn lại chính số bão hòa này
                adjusted_mat[i, :] *= 0.15
                adjusted_mat[:, i] *= 0.15
                
                # Chuyển dịch năng lượng sang 2 số lân cận (i-1 và i+1)
                left_neighbor = max(0, i - 1)
                right_neighbor = min(D - 1, i + 1)
                
                if freqs[left_neighbor] < 3:
                    adjusted_mat[left_neighbor, :] *= 1.8
                if freqs[right_neighbor] < 3:
                    adjusted_mat[right_neighbor, :] *= 1.8
                    
        # Phạt các cặp vừa nổ cùng nhau ở kỳ gần nhất
        last_co = np.outer(X[-1], X[-1])
        adjusted_mat[last_co == 1] *= 0.10
        
        return self._norm(adjusted_mat)

    # --------------------------------------------------------------------------
    # PROCESSOR TỔNG HỢP CO-EMISSION
    # --------------------------------------------------------------------------
    def process(self, X):
        # 1. Tính Ma trận Đồng phát Năng lượng
        co_mat = self._compute_co_occurrence(X)
        
        # 2. Áp dụng Bộ lọc Lọc Bão hòa & Dịch chuyển Lân cận
        final_pair_mat = self._apply_saturation_and_shift(X, co_mat)
        
        # 3. Trích xuất Cặp số Chốt có Chỉ số Cùng Nổ cao nhất
        i, j = np.unravel_index(np.argmax(final_pair_mat, axis=None), final_pair_mat.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(final_pair_mat[i, j])

        explanation = (
            f"• **CƠ CHẾ ĐỒNG PHÁT NĂNG LƯỢNG & DỊCH CHUYỂN LÂN CẬN (Co-Emission Engine):**\n"
            f"  - **Ma trận Cùng Nổ P(B|A):** Thay vì chọn 2 số có điểm cá nhân cao, mô hình tối ưu hóa trực tiếp độ đồng xuất hiện của cả cặp.\n"
            f"  - **Lọc Bão Hòa (Over-Saturation Gate):** Triệt tiêu các số đã xuất hiện $\ge 3$ kỳ liên tiếp (như 52) để tránh bẫy hụt số.\n"
            f"  - **Dịch Chuyển Lân Cận (Neighbor Shift):** Chuyển dịch năng lượng từ số bão hòa sang dải số kề bên có xác suất bùng nổ cao hơn.\n"
            f"• **Kết luận Chốt:** Cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Co-Emission Network Engine")
st.caption("Khắc phục bẫy hụt 1 con • Ma trận Xác suất Đồng phát P(B|A) • Lọc Bão hòa & Dịch chuyển Lân cận")

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
                
        st.success("🎉 Đã chạy xong Mô hình Bắt cặp Đồng phát Co-Emission!")
        
        engine = CoEmissionKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ ĐỒNG PHÁT TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT ĐỒNG PHÁT", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Co-Emission Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình Co-Emission.")
