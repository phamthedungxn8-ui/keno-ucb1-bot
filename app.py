import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Hybrid Multi-Agent Engine", layout="centered")

# ==============================================================================
# HYBRID ENGINE: MÔ HÌNH CŨ (CORE ANCHOR) + MÔ HÌNH MỚI (PARALLEL FILTER)
# ==============================================================================
class KenoHybridEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _normalize(self, mat):
        """Chuẩn hóa ma trận điểm về thang [0, 1]"""
        max_val = np.max(mat)
        if max_val > 0:
            return mat / max_val
        return mat

    # --------------------------------------------------------------------------
    # 1. CORE MÔ HÌNH CŨ: LỰC KÉO HẠT NHÂN VÀ TẦN SUẤT LẶP TRỰC TIẾP
    # --------------------------------------------------------------------------
    def _legacy_core_engine(self, matrix, hot_indices):
        T, D = matrix.shape
        legacy_mat = np.zeros((D, D))
        
        # Quét lực kéo hạt nhân trực tiếp theo logic mô hình cũ
        for t in range(T):
            active = np.where(matrix[t] == 1)[0]
            for i in active:
                for j in active:
                    if i != j:
                        weight = 1.0
                        # Ưu tiên cộng điểm cao cho hạt nhân lặp xuất hiện >= 2 lần
                        if i in hot_indices: weight += 2.0
                        if j in hot_indices: weight += 2.0
                        # Nếu cả 2 đều là hạt nhân lặp cùng xuất hiện
                        if i in hot_indices and j in hot_indices: weight += 1.5
                        legacy_mat[i, j] += weight
                        
        np.fill_diagonal(legacy_mat, 0)
        return self._normalize(legacy_mat)

    # --------------------------------------------------------------------------
    # 2. CÁC ENGINE MÔ HÌNH MỚI: PHÂN TÍCH VÀ KHỬ NHIỄU SONG SONG
    # --------------------------------------------------------------------------
    def _engine_entropy(self, matrix):
        """Đo sự tụ năng lượng Entropy"""
        T, D = matrix.shape
        p = matrix.sum(axis=0) / float(T)
        p = np.clip(p, 1e-5, 1.0 - 1e-5)
        entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
        energy = 1.0 - (entropy / np.max(entropy))
        mat = np.outer(energy, energy)
        np.fill_diagonal(mat, 0)
        return self._normalize(mat)

    def _engine_holographic(self, matrix):
        """Mạng lưới khoảng cách đối xứng"""
        T, D = matrix.shape
        mat = np.zeros((D, D))
        for t in range(T):
            active = np.where(matrix[t] == 1)[0]
            if len(active) > 1:
                diffs = [abs(active[a] - active[b]) for a in range(len(active)) for b in range(a+1, len(active))]
                if diffs:
                    common_diff = max(set(diffs), key=diffs.count)
                    for i in active:
                        for j in active:
                            if i != j and abs(i - j) == common_diff:
                                mat[i, j] += 1.0
        np.fill_diagonal(mat, 0)
        return self._normalize(mat)

    def _engine_svd(self, matrix):
        """SVD Phân rã lọc nhiễu trắng"""
        D = matrix.shape[1]
        mat = np.zeros((D, D))
        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
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
    # 3. HYBRID PIPELINE (KẾT HỢP VÀ LỌC NÂNG CAO)
    # --------------------------------------------------------------------------
    def process(self, matrix):
        T, D = matrix.shape
        freqs = matrix.sum(axis=0)
        hot_indices = np.where(freqs >= 2)[0]

        # Bước 1: Tính ma trận gốc từ Mô Hình Cũ (Anchor Weight)
        legacy_score = self._legacy_core_engine(matrix, hot_indices)

        # Bước 2: Chạy 3 Engine của Mô Hình Mới
        e_entropy = self._engine_entropy(matrix)
        e_holo = self._engine_holographic(matrix)
        e_svd = self._engine_svd(matrix)

        # Tổng hợp điểm cộng hưởng từ mô hình mới
        new_refinement = 0.4 * e_entropy + 0.3 * e_holo + 0.3 * e_svd
        new_refinement = self._normalize(new_refinement)

        # Bước 3: Tích hợp Hybrid (Gộp 60% Mô hình cũ + 40% Mô hình mới)
        hybrid_matrix = 1.8 * legacy_score + 1.2 * new_refinement

        # Bước 4: Áp dụng Tầng lọc Đa tầng (Filter Chain)
        co_occur = np.dot(matrix.T, matrix)
        hybrid_matrix[co_occur == 0] *= 0.15 # Giảm 85% điểm nếu chưa từng đi chung kỳ nào

        # Lọc ngưỡng Top 15% điểm cao nhất để triệt tiêu nhiễu
        threshold = np.percentile(hybrid_matrix, 85)
        hybrid_matrix[hybrid_matrix < threshold] = 0

        if np.max(hybrid_matrix) == 0:
            hybrid_matrix = legacy_score + new_refinement

        # Chốt vị trí cặp số tối ưu
        i, j = np.unravel_index(np.argmax(hybrid_matrix, axis=None), hybrid_matrix.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(hybrid_matrix[i, j])

        hot_str = ", ".join([f"{h+1:02d}" for h in hot_indices]) if len(hot_indices) > 0 else "Không có"
        explanation = (
            f"**Hạt nhân lặp ghi nhận:** [{hot_str}]\n\n"
            f"**Cơ chế tích hợp Hybrid:**\n"
            f"• **Core Lực kéo (Mô hình cũ):** Đã cố định khung điểm ưu tiên cho các hạt nhân xuất hiện liên tục.\n"
            f"• **Multi-Engine Filter (Mô hình mới):** Quét bổ sung Entropy, Mạng khoảng cách & SVD Khử nhiễu.\n"
            f"• **Chốt kết quả:** Cặp số **({num1:02d}, {num2:02d})** đạt điểm cân bằng cao nhất giữa lực kéo thực tế và cộng hưởng toán học."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Hybrid Meta-Engine")
st.caption("Kết hợp Core Lực Kéo (Mô hình cũ) + Lọc Đa Tầng SVD/Entropy (Mô hình mới)")

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
                
        st.success("🎉 Hệ thống đã nạp thành công 5 kỳ và kích hoạt Mô Hình Hybrid!")
        
        engine = KenoHybridEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 ĐỒNG THUẬN HYBRID (BẬC 2)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="CẶP SỐ CHỐT TỐI ƯU",
                value=f"{best_pair[0]:02d} — {best_pair[1]:02d}"
            )
        with col2:
            st.metric(
                label="Chỉ Số Cộng Hưởng Hybrid",
                value=f"{score:.3f}"
            )
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Hãy dán chuỗi số 5 kỳ vào khung phía trên để mô hình Hybrid tự động vận hành.")
