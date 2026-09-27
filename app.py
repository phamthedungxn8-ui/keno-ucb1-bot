import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Restructured Hybrid Dual-Pair Engine", layout="centered")

# ==============================================================================
# RESTRUCTURED HYBRID ENGINE (5-IN-1 ALGORITHMS -> 2 SECOND-ORDER PAIRS FOR 1 DRAW)
# ==============================================================================
class RestructuredKenoDualPairEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, vec_or_mat):
        m = np.max(vec_or_mat)
        return vec_or_mat / m if m > 0 else vec_or_mat

    # 1. SPATIAL GNN: Lan truyền năng lượng ma trận kề đồ thị
    def _gnn_layer(self, X):
        A = np.dot(X.T, X)
        np.fill_diagonal(A, 0)
        degree = np.sum(A, axis=1)
        deg_inv_sqrt = np.power(degree, -0.5, where=degree>0)
        deg_inv_sqrt[degree == 0] = 0
        D_mat = np.diag(deg_inv_sqrt)
        L_norm = np.dot(np.dot(D_mat, A), D_mat)
        gnn_signal = np.tanh(np.dot(L_norm, X[-1]))
        gnn_mat = np.outer(gnn_signal, gnn_signal)
        np.fill_diagonal(gnn_mat, 0)
        return self._norm(gnn_mat)

    # 2. TRANSFORMER ATTENTION: Tín hiệu chú ý chuỗi thời gian Q, K, V
    def _transformer_layer(self, X):
        scores = np.dot(X.T, X) / np.sqrt(X.shape[0])
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_mat = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        np.fill_diagonal(attn_mat, 0)
        return self._norm(attn_mat)

    # 3. PHASE-SHIFT: Đo độ lệch pha chu kỳ nghỉ & nổ
    def _phase_shift_layer(self, X):
        T, D = X.shape
        last_seen = np.zeros(D)
        for d in range(D):
            pos = np.where(X[:, d] == 1)[0]
            last_seen[d] = pos[-1] if len(pos) > 0 else -1
        phase_diff = np.abs(last_seen[:, None] - last_seen[None, :])
        np.fill_diagonal(phase_diff, 0)
        return self._norm(phase_diff)

    # 4. CHAOS ORTHOGONALITY & SHANNON ENTROPY: Vuông góc Vector & Cân bằng Hỗn loạn
    def _chaos_entropy_layer(self, X):
        # Cosine Orthogonality
        vectors = X.T
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_vecs = vectors / norms
        cosine_sim = np.dot(norm_vecs, norm_vecs.T)
        ortho_mat = 1.0 - np.abs(cosine_sim)
        
        # Shannon Entropy
        p1 = np.clip(X.sum(axis=0) / X.shape[0], 1e-5, 1.0 - 1e-5)
        p0 = 1.0 - p1
        entropy = - (p1 * np.log2(p1) + p0 * np.log2(p0))
        
        entropy_mat = np.outer(entropy, entropy)
        res_mat = ortho_mat * entropy_mat
        np.fill_diagonal(res_mat, 0)
        return self._norm(res_mat)

    # --------------------------------------------------------------------------
    # PROCESSOR TÁI CẤU TRÚC: CHỐT 2 CẶP SỐ BẬC 2 CHO 1 KỲ DỰ BÁO
    # --------------------------------------------------------------------------
    def process_dual_pairs(self, X):
        T, D = X.shape
        
        # Tích hợp 4 Tầng Thuật toán
        m_gnn = self._gnn_layer(X)
        m_attn = self._transformer_layer(X)
        m_phase = self._phase_shift_layer(X)
        m_chaos = self._chaos_entropy_layer(X)
        
        # Tổng hợp Ma trận Năng lượng Đa tầng (Multi-Layer Ensemble Matrix)
        ensemble_mat = (0.30 * m_chaos) + (0.25 * m_attn) + (0.25 * m_gnn) + (0.20 * m_phase)
        
        # LỌC CỨNG: Phạt nặng các cặp số đã cùng xuất hiện >= 2 lần trong 5 kỳ
        co_occur = np.dot(X.T, X)
        ensemble_mat[co_occur >= 2] *= 0.05
        
        # Phạt các số bão hòa (nổ >= 3 kỳ)
        freqs = X.sum(axis=0)
        for d in range(D):
            if freqs[d] >= 3:
                ensemble_mat[d, :] *= 0.1
                ensemble_mat[:, d] *= 0.1
                
        ensemble_mat = self._norm(ensemble_mat)
        
        # ----------------------------------------------------------------------
        # BỘ LỌC PHÂN RÃ BẬC 2 (SECOND-ORDER DECOUPLING SELECTION)
        # ----------------------------------------------------------------------
        # Cặp 1: Cặp số có điểm số Ensemble cao nhất toàn hệ thống (Primary Pair)
        i1, j1 = np.unravel_index(np.argmax(ensemble_mat, axis=None), ensemble_mat.shape)
        pair1 = sorted([int(i1 + 1), int(j1 + 1)])
        score1 = float(ensemble_mat[i1, j1])
        
        # Triệt tiêu năng lượng của Cặp 1 và các vùng lân cận để ép hệ thống tìm Cặp 2 độc lập
        decoupled_mat = ensemble_mat.copy()
        decoupled_mat[i1, :] = 0.0
        decoupled_mat[:, i1] = 0.0
        decoupled_mat[j1, :] = 0.0
        decoupled_mat[:, j1] = 0.0
        
        # Cặp 2: Cặp số Bậc 2 bổ trợ pha (Secondary Pair)
        i2, j2 = np.unravel_index(np.argmax(decoupled_mat, axis=None), decoupled_mat.shape)
        pair2 = sorted([int(i2 + 1), int(j2 + 1)])
        score2 = float(decoupled_mat[i2, j2])

        explanation = (
            f"• **TÁI CẤU TRÚC MÔ HÌNH PHÂN TÍCH 1 KỲ (Dual-Pair Hybrid Dynamics):**\n"
            f"  - **Tích Hợp Trọn Bộ 5 Thuật Toán:** Kết hợp GNN Đồ thị (25%), Transformer Attention (25%), Chaos Orthogonality Vector (30%), Shannon Entropy và Phase-Shift (20%).\n"
            f"  - **Phân Rã Cặp Bậc 2 (Decoupling Gate):** Thay vì chốt 1 cặp rủi ro, hệ thống trích xuất 2 Cặp Số Bậc 2 độc lập không gian. Cặp 1 giữ vai trò Chủ lực, Cặp 2 đóng vai trò Lót pha triệt tiêu lệch pha vi mô."
        )

        return pair1, score1, pair2, score2, explanation

# ==============================================================================
# STREAMLIT UI DISPLAY
# ==============================================================================
st.title("⚡ Keno Restructured Dual-Pair Engine")
st.caption("Tập trung 1 Kỳ Dự Báo • Tích hợp trọn bộ 5 thuật toán lõi • Chốt 2 Cặp Số Bậc 2 Độc Lập")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=150,
    key="raw_text_keno_dual"
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
                
        st.success("🎉 Đã chạy xong Mô hình Tái Cấu Trúc Dual-Pair!")
        
        engine = RestructuredKenoDualPairEngine(num_dim=80)
        pair1, score1, pair2, score2, explanation = engine.process_dual_pairs(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ CHỐT 1 KỲ (2 CẶP BẬC 2 TỐI ƯU)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🔥 CẶP 1 (CHỦ LỰC)", value=f"{pair1[0]:02d} — {pair1[1]:02d}")
            st.caption(f"Ensemble Score: {score1:.4f}")
        with col2:
            st.metric(label="🛡️ CẶP 2 (LÓT PHA BẬC 2)", value=f"{pair2[0]:02d} — {pair2[1]:02d}")
            st.caption(f"Decoupled Score: {score2:.4f}")
            
        st.markdown("---")
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung trên để chạy Mô hình Dual-Pair 1 Kỳ.")
