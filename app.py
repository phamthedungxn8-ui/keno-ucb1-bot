import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Master Unified Engine", layout="centered")

# ==============================================================================
# UNIFIED MULTI-AGENT ARCHITECTURE (HỢP NHẤT TOÀN BỘ THUẬT TOÁN)
# ==============================================================================
class UnifiedKenoMasterEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, mat):
        m = np.max(mat)
        return mat / m if m > 0 else mat

    def process(self, X):
        """
        X: Ma trận nhị phân (5 kỳ, 80 số)
        Tích hợp: Micro Bayes + Time Decay + Spatial Distance + SVD Denoise +
                  Surprisal Information + Graph Centrality + Anti-Saturation Penalty
        """
        T, D = X.shape
        freqs = X.sum(axis=0)
        co_occur = np.dot(X.T, X)
        
        # 1. TIMING & DECAY VECTORIZATION (Trạng thái thời gian)
        decay_weights = np.exp(np.linspace(-0.6, 0, T)) # Kỳ gần nhất trọng số cao hơn
        X_decay = X * decay_weights[:, np.newaxis]
        
        # 2. ENGINE A: Micro Bayes & Conditional Link
        p_cond = np.zeros((D, D))
        for i in range(D):
            if freqs[i] > 0:
                p_cond[i, :] = co_occur[i, :] / freqs[i]
        mat_bayes = self._norm(p_cond)

        # 3. ENGINE B: Spatial Distance Embedding (Manhattan Similarity)
        # Đo độ tương đồng vận động không gian 5 chiều giữa các số
        dist_mat = np.array([[np.sum(np.abs(X[:, i] - X[:, j])) for j in range(D)] for i in range(D)])
        mat_spatial = self._norm(1.0 / (1.0 + dist_mat))

        # 4. ENGINE C: SVD Low-Rank Denoising (Lọc nhiễu trắng)
        try:
            U, S, Vt = np.linalg.svd(X_decay, full_matrices=False)
            S_clean = np.zeros_like(S)
            S_clean[0] = S[0]
            if len(S) > 1: S_clean[1] = S[1]
            clean_X = np.dot(U, np.dot(np.diag(S_clean), Vt))
            mat_svd = self._norm(np.maximum(np.dot(clean_X.T, clean_X), 0))
        except:
            mat_svd = np.zeros((D, D))

        # 5. ENGINE D: Surprisal & Graph Centrality (Lượng tin & Cầu nối)
        p_occ = np.clip(freqs / float(T), 1e-4, 1.0 - 1e-4)
        surprisal = -np.log2(p_occ) * (X[-1] + X[-2] * 0.5) # Chỉ lấy số có nhịp bùng nổ
        mat_surprisal = self._norm(np.outer(surprisal, surprisal))
        
        deg = np.sum(co_occur > 0, axis=1)
        mat_graph = self._norm(co_occur * (np.log1p(deg)[:, None] + np.log1p(deg)[None, :]))

        # 6. ENGINE E: Phase Transition Momentum (Động lượng nhịp rebound)
        recency = X[-1] + X[-2] * 0.5
        mat_momentum = self._norm(np.outer(recency == 0.5, recency == 0.5)) # Ưu tiên cặp vừa nghỉ 1 kỳ

        # 7. ANTI-SATURATION PENALTY CHAIN (Bộ lọc chống bẫy lặp)
        penalty = np.ones((D, D))
        penalty[co_occur >= 3] = 0.20 # Phạt 80% nếu lặp >= 3 kỳ
        penalty[co_occur >= 4] = 0.05 # Phạt 95% nếu lặp >= 4 kỳ

        # 8. FUSION TENSOR AGGREGATION (Cộng hưởng tổng hợp)
        fused = (
            1.8 * mat_bayes + 
            1.5 * mat_spatial + 
            1.2 * mat_momentum + 
            1.0 * mat_surprisal + 
            1.0 * mat_graph + 
            0.8 * mat_svd
        ) * penalty
        
        np.fill_diagonal(fused, 0)

        # Trích xuất Cặp Bậc 2 Tối Ưu
        i, j = np.unravel_index(np.argmax(fused, axis=None), fused.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        score = float(fused[i, j])

        hot_str = ", ".join([f"{h+1:02d}" for h in np.where(freqs >= 2)[0]]) or "Không có"
        sat_str = ", ".join([f"{s+1:02d}" for s in np.where(freqs >= 4)[0]]) or "Không có"

        explanation = (
            f"• **Hạt nhân lặp ghi nhận:** [{hot_str}]\n"
            f"• **Số bị phạt bẫy quá tải (>=4 kỳ):** [{sat_str}]\n"
            f"• **Cơ chế:** Đã tích hợp đồng thời 6 Động cơ (Bayes, Manhattan, SVD, Surprisal, Graph & Momentum) "
            f"kết hợp màng lọc triệt bẫy lặp để chốt cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Master Unified Engine")
st.caption("Mô hình Hợp nhất Rút gọn • Multi-Agent Tensor Aggregation")

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
                
        st.success("🎉 Hệ thống Unified Engine đã kích hoạt thành công!")
        
        engine = UnifiedKenoMasterEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP BẬC 2 CHỐT TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ TỐI ƯU", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Điểm Cộng Hưởng Master Score", value=f"{score:.3f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình hợp nhất.")
