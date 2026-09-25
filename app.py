import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Advanced Algorithmic Engine", layout="centered")

# ==============================================================================
# OPTIMIZED MULTI-ALGORITHMIC PIPELINE
# ==============================================================================
class KenoAlgorithmicOptimizer:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _normalize(self, mat):
        """Standardize scoring matrix to [0, 1] range"""
        max_val = np.max(mat)
        if max_val > 0:
            return mat / max_val
        return mat

    def _build_decay_matrix(self, raw_matrix):
        """
        Tối ưu Dữ liệu Đầu vào: Gán trọng số giảm dần theo thời gian (Time Decay)
        Các kỳ gần nhất sẽ có ảnh hưởng mạnh hơn các kỳ xa.
        """
        T, D = raw_matrix.shape
        decay_weights = np.exp(np.linspace(-0.5, 0, T))  # Trọng số tăng dần về kỳ gần nhất
        decay_matrix = raw_matrix * decay_weights[:, np.newaxis]
        return decay_matrix

    # --------------------------------------------------------------------------
    # THUẬT TOÁN 1: Entanglement với Time-Decay
    # --------------------------------------------------------------------------
    def _engine_weighted_entanglement(self, decay_matrix, hot_indices):
        T, D = decay_matrix.shape
        mat = np.zeros((D, D))
        for t in range(T):
            active = np.where(decay_matrix[t] > 0)[0]
            weight_factor = decay_matrix[t, active[0]] if len(active) > 0 else 1.0
            for i in active:
                for j in active:
                    if i != j:
                        w = weight_factor
                        if i in hot_indices: w += 1.0
                        if j in hot_indices: w += 1.0
                        mat[i, j] += w
        np.fill_diagonal(mat, 0)
        return self._normalize(mat)

    # --------------------------------------------------------------------------
    # THUẬT TOÁN 2: Jaccard Similarity (Đo độ tương đồng tập hợp)
    # --------------------------------------------------------------------------
    def _engine_jaccard_similarity(self, raw_matrix):
        T, D = raw_matrix.shape
        mat = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                col_i = raw_matrix[:, i]
                col_j = raw_matrix[:, j]
                intersection = np.sum(np.logical_and(col_i, col_j))
                union = np.sum(np.logical_or(col_i, col_j))
                if union > 0:
                    sim = intersection / union
                    mat[i, j] = sim
                    mat[j, i] = sim
        return self._normalize(mat)

    # --------------------------------------------------------------------------
    # THUẬT TOÁN 3: Graph Laplacian Spectral Analysis (Lọc cụm liên kết)
    # --------------------------------------------------------------------------
    def _engine_spectral_laplacian(self, raw_matrix):
        # Tạo ma trận kề (Adjacency Matrix)
        adj = np.dot(raw_matrix.T, raw_matrix)
        np.fill_diagonal(adj, 0)
        
        # Tính Ma trận Bậc (Degree Matrix)
        degrees = np.sum(adj, axis=1)
        deg_mat = np.diag(degrees)
        
        # Ma trận Laplacian L = D - A
        laplacian = deg_mat - adj
        
        # Chiếu tín hiệu qua Laplacian để tìm vùng năng lượng liên kết tối ưu
        spectral_mat = np.dot(adj, np.linalg.pinv(laplacian + np.eye(self.D) * 1e-5))
        spectral_mat = np.maximum(spectral_mat, 0)
        np.fill_diagonal(spectral_mat, 0)
        return self._normalize(spectral_mat)

    # --------------------------------------------------------------------------
    # THUẬT TOÁN 4: SVD Low-Rank Approximation (Khử nhiễu không gian)
    # --------------------------------------------------------------------------
    def _engine_svd_denoise(self, decay_matrix):
        D = decay_matrix.shape[1]
        mat = np.zeros((D, D))
        try:
            U, S, Vt = np.linalg.svd(decay_matrix, full_matrices=False)
            S_clean = np.zeros_like(S)
            S_clean[0] = S[0]
            if len(S) > 1: S_clean[1] = S[1]
            clean_matrix = np.dot(U, np.dot(np.diag(S_clean), Vt))
            raw_svd = np.dot(clean_matrix.T, clean_matrix)
            mat = np.maximum(raw_svd, 0)
            np.fill_diagonal(mat, 0)
        except (np.linalg.LinAlgError, ValueError):
            mat = np.zeros((D, D))
        return self._normalize(mat)

    # --------------------------------------------------------------------------
    # TỔNG HỢP VÀ LỌC ĐA TẦNG
    # --------------------------------------------------------------------------
    def process(self, raw_matrix):
        T, D = raw_matrix.shape
        freqs = raw_matrix.sum(axis=0)
        hot_indices = np.where(freqs >= 2)[0]

        # 1. Tiền xử lý dữ liệu đầu vào với Time Decay
        decay_matrix = self._build_decay_matrix(raw_matrix)

        # 2. Chạy 4 Engine thuật toán song song
        e1 = self._engine_weighted_entanglement(decay_matrix, hot_indices)
        e2 = self._engine_jaccard_similarity(raw_matrix)
        e3 = self._engine_spectral_laplacian(raw_matrix)
        e4 = self._engine_svd_denoise(decay_matrix)

        # 3. Kết hợp có trọng số (Fusion Operator)
        fused_matrix = 1.5 * e1 + 1.2 * e2 + 1.0 * e3 + 0.8 * e4

        # 4. Tầng lọc loại bỏ nhiễu (Filtering Chain)
        co_occur = np.dot(raw_matrix.T, raw_matrix)
        fused_matrix[co_occur == 0] *= 0.1  # Phạt nặng các cặp số chưa từng đi cùng nhau trong quá khứ

        # Triệt tiêu 80% nhiễu điểm thấp
        threshold = np.percentile(fused_matrix, 80)
        fused_matrix[fused_matrix < threshold] = 0

        if np.max(fused_matrix) == 0:
            fused_matrix = e1 + e2

        # Chốt kết quả cặp Bậc 2
        i, j = np.unravel_index(np.argmax(fused_matrix, axis=None), fused_matrix.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(fused_matrix[i, j])

        hot_str = ", ".join([f"{h+1:02d}" for h in hot_indices]) if len(hot_indices) > 0 else "Không có"
        explanation = (
            f"**Hạt nhân lặp:** [{hot_str}]\n\n"
            f"**Quy trình phân tích thuật toán:**\n"
            f"1. *Xử lý đầu vào:* Áp dụng Ma trận Suy giảm Thời gian (Time Decay) cho 5 kỳ gần nhất.\n"
            f"2. *Kết hợp Thuật toán:* Tích hợp Lực kéo Lặp, Tương đồng Jaccard, Phổ đồ thị Laplacian & SVD Lọc nhiễu.\n"
            f"3. *Kết quả:* Cặp số **({num1:02d}, {num2:02d})** đạt điểm cộng hưởng thống kê cao nhất."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Optimized Algorithmic Engine")
st.caption("Tối ưu hóa dữ liệu đầu vào Time-Decay • Spectral Laplacian • Jaccard • SVD")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 04 06 10 18 ...\nKì 2: ...",
    height=160,
    key="raw_text_keno"
)

if raw_text_input.strip():
    raw_data = raw_text_input.strip()
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_data, flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        valid_numbers = all_numbers[:100]
        
        matrix = np.zeros((5, 80), dtype=float)
        for k in range(5):
            ky_nums = valid_numbers[k * 20 : (k + 1) * 20]
            for num in ky_nums:
                matrix[k, num - 1] = 1.0
                
        st.success("🎉 Đã tối ưu hóa vector đầu vào và tính toán hệ thống thuật toán!")
        
        optimizer = KenoAlgorithmicOptimizer(num_dim=80)
        best_pair, score, explanation = optimizer.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP BẬC 2 TỐI ƯU NHẤT")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="CẶP SỐ CHỐT",
                value=f"{best_pair[0]:02d} — {best_pair[1]:02d}"
            )
        with col2:
            st.metric(
                label="Điểm Cộng Hưởng Thuật Toán",
                value=f"{score:.3f}"
            )
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Hãy dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình tối ưu.")
