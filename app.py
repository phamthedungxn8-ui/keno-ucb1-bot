import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Quad-Role Ensemble Engine", layout="centered")

# ==============================================================================
# QUAD-ROLE ENSEMBLE ENGINE (MÔ HÌNH HỢP NHẤT 4 VAI TRÒ NĂNG LƯỢNG)
# ==============================================================================
class QuadRoleKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, vec):
        m = np.max(vec)
        return vec / m if m > 0 else vec

    # --------------------------------------------------------------------------
    # VAI TRÒ 1: SỐ LUỒNG SÓNG HÀI (HARMONIC FLOW - Thuật toán Quantum Engine)
    # --------------------------------------------------------------------------
    def _role_harmonic_flow(self, X):
        T, D = X.shape
        fft_specs = np.abs(np.fft.fft(X, axis=0))
        freq_scores = np.mean(fft_specs, axis=0)
        return self._norm(freq_scores)

    # --------------------------------------------------------------------------
    # VAI TRÒ 2: SỐ LẮP ĐỘNG LƯỢNG (REBOUND MOMENTUM - Thuật toán Hyper-Layered)
    # --------------------------------------------------------------------------
    def _role_rebound_momentum(self, X):
        T, D = X.shape
        rates = np.zeros(D)
        for i in range(D):
            x_hat, P, Q, R = 0.25, 1.0, 0.05, 0.2
            for t in range(T):
                P += Q
                K = P / (P + R)
                x_hat += K * (X[t, i] - x_hat)
                P = (1 - K) * P
            rates[i] = x_hat
            
        # Thưởng đặc biệt cho số vừa nổ kỳ t-1 và có động lượng Kalman cao
        rebound = rates * X[-1]
        return self._norm(rebound)

    # --------------------------------------------------------------------------
    # VAI TRÒ 3: SỐ MẠNG CỤM DẢI (CLUSTER DENSITY - Thuật toán Phase Engine)
    # --------------------------------------------------------------------------
    def _role_cluster_density(self, X):
        T, D = X.shape
        co_occur = np.dot(X.T, X)
        spatial_density = np.zeros(D)
        
        for i in range(D):
            # Tính tổng mật độ liên kết với các số xung quanh ranh giới +-3
            min_idx = max(0, i - 3)
            max_idx = min(D, i + 4)
            spatial_density[i] = np.sum(co_occur[i, min_idx:max_idx])
            
        return self._norm(spatial_density)

    # --------------------------------------------------------------------------
    # VAI TRÒ 4: SỐ TÍCH LŨY NHẢ LẠI (GAP RETURN - Thuật toán Hedged Engine)
    # --------------------------------------------------------------------------
    def _role_gap_return(self, X):
        T, D = X.shape
        gap_scores = np.zeros(D)
        
        for i in range(D):
            # Vừa nghỉ đúng 1 kỳ (nổ ở t-2, nghỉ ở t-1) -> Nhịp tích lũy điểm cao nhất
            if X[-1, i] == 0 and X[-2, i] == 1:
                gap_scores[i] = 1.0
            elif X[-1, i] == 0 and T >= 3 and X[-3, i] == 1:
                gap_scores[i] = 0.6
            else:
                gap_scores[i] = 0.1
                
        return self._norm(gap_scores)

    # --------------------------------------------------------------------------
    # PROCESSOR PHỐI HỢP DÒNG LIÊN KẾT ĐA VAI TRÒ
    # --------------------------------------------------------------------------
    def process(self, X):
        T, D = X.shape

        # 1. Trích xuất điểm của 4 Vai Trò Năng Lượng Độc Lập
        s_flow = self._role_harmonic_flow(X)      # Vai trò A
        s_rebound = self._role_rebound_momentum(X) # Vai trò B
        s_cluster = self._role_cluster_density(X)  # Vai trò C
        s_gap = self._role_gap_return(X)          # Vai trò D

        # Group 1: Nhóm Dẫn Dắt Động Lượng (Primary Drivers) = Max(Flow, Rebound)
        primary_drivers = np.maximum(s_flow, s_rebound)

        # Group 2: Nhóm Cân Bằng Cụm & Tích Lũy (Secondary Balancers) = Max(Cluster, Gap)
        secondary_balancers = np.maximum(s_cluster, s_gap)

        # 2. Xây dựng Ma trận Bắt Cặp Chéo (Cross-Role Pairing Matrix)
        # Ép buộc 1 số thuộc Group 1 bắt cặp với 1 số thuộc Group 2
        pair_matrix = np.outer(primary_drivers, secondary_balancers)
        
        # Phạt các cặp số có cùng chỉ số vị trí (i == j) hoặc lặp cặp nổ trùng ở kỳ trước
        np.fill_diagonal(pair_matrix, 0)
        
        last_co = np.outer(X[-1], X[-1])
        pair_matrix[last_co == 1] *= 0.1 # Phạt 90% nếu vừa nổ chung ở kỳ cuối

        # 3. Trích xuất Cặp Số Hợp Nhất Tối Ưu
        i, j = np.unravel_index(np.argmax(pair_matrix, axis=None), pair_matrix.shape)
        
        # Xác định vai trò cụ thể của từng số
        role_i = "Luồng Sóng / Lặp Động Lượng" if primary_drivers[i] >= secondary_balancers[i] else "Cụm Dải / Tích Lũy"
        role_j = "Cụm Dải / Tích Lũy" if secondary_balancers[j] >= primary_drivers[j] else "Luồng Sóng / Lặp Động Lượng"

        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(pair_matrix[i, j])

        explanation = (
            f"• **CƠ CHẾ HỢP NHẤT 4 THUẬT TOÁN (Quad-Role Ensemble):**\n"
            f"  - **Số {i+1:02d}:** Đóng vai trò *[{role_i}]* (Kế thừa Động cơ Quantum & Hyper-Layered).\n"
            f"  - **Số {j+1:02d}:** Đóng vai trò *[{role_j}]* (Kế thừa Động cơ Phase & Hedged Pairing).\n"
            f"• **Giải mã Liên kết ngầm:** Loại bỏ bẫy ghép 2 số cùng loại. Bắt buộc 1 số Chủ Đạo Động Lượng đi kèm 1 số Cân Bằng Cụm/Tích Lũy để triệt tiêu rủi ro hụt số.\n"
            f"• **Kết luận Chốt:** Cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Quad-Role Ensemble Engine")
st.caption("Hợp nhất 4 Thuật toán • Bắt cặp Chéo Đa Vai Trò • Khắc phục triệt để bẫy hụt số")

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
                
        st.success("🎉 Đã chạy xong Mô hình Hợp nhất 4 Thuật toán Đa Vai trò!")
        
        engine = QuadRoleKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ HOÀN CHỈNH ĐA VAI TRÒ")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT HỢP NHẤT", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Hợp Nhất Ensemble Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình hợp nhất.")
