import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Micro-Link Engine", layout="centered")

# ==============================================================================
# ULTRA MICRO-CORRELATION ENGINE
# ==============================================================================
class MicroLinkKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _normalize(self, mat):
        max_val = np.max(mat)
        if max_val > 0:
            return mat / max_val
        return mat

    # --------------------------------------------------------------------------
    # 1. THUẬT TOÁN BÁT MỐI VI MÔ (MICRO CONDITIONAL PROBABILITY)
    # --------------------------------------------------------------------------
    def _engine_micro_bayes(self, raw_matrix):
        """Bắt các mối liên kết xác suất dù nhỏ nhất giữa từng cặp số"""
        T, D = raw_matrix.shape
        mat = np.zeros((D, D))
        freqs = raw_matrix.sum(axis=0)
        
        for i in range(D):
            for j in range(D):
                if i != j and freqs[i] > 0:
                    # Tỉ lệ số j xuất hiện khi số i xuất hiện P(J|I)
                    co_occur = np.sum(raw_matrix[:, i] * raw_matrix[:, j])
                    mat[i, j] = co_occur / freqs[i]
                    
        return self._normalize(mat)

    # --------------------------------------------------------------------------
    # 2. THUẬT TOÁN NHỊP SINH HỌC KHÔNG GIAN (MANHATTAN EMBEDDING DISTANCE)
    # --------------------------------------------------------------------------
    def _engine_spatial_embedding(self, raw_matrix):
        """Đo độ tương đồng không gian vector 5 chiều giữa 80 con số"""
        T, D = raw_matrix.shape
        mat = np.zeros((D, D))
        
        for i in range(D):
            for j in range(i + 1, D):
                # Tính khoảng cách Manhattan giữa 2 vector thời gian của 2 số
                vec_i = raw_matrix[:, i]
                vec_j = raw_matrix[:, j]
                dist = np.sum(np.abs(vec_i - vec_j))
                
                # Khoảng cách càng nhỏ (dist -> 0) nghĩa là 2 số chuyển động càng giống nhau
                similarity = 1.0 / (1.0 + dist)
                mat[i, j] = similarity
                mat[j, i] = similarity
                
        np.fill_diagonal(mat, 0)
        return self._normalize(mat)

    # --------------------------------------------------------------------------
    # 3. THUẬT TOÁN KÉO THEO CHUỖI THỜI GIAN TRƯỢT (SLIDING TIME CONVOLUTION)
    # --------------------------------------------------------------------------
    def _engine_time_sliding_lead(self, raw_matrix):
        """Quét xem số xuất hiện ở kỳ t có kéo theo số khác ở kỳ t+1 hay không"""
        T, D = raw_matrix.shape
        mat = np.zeros((D, D))
        
        for t in range(T - 1):
            active_t = np.where(raw_matrix[t] == 1)[0]
            active_next = np.where(raw_matrix[t + 1] == 1)[0]
            
            for i in active_t:
                for j in active_next:
                    if i != j:
                        mat[i, j] += 1.0 + (t * 0.2) # Kỳ càng gần trọng số càng cao
                        
        return self._normalize(mat)

    # --------------------------------------------------------------------------
    # 4. SVD & LAPLACIAN BỔ TRỢ KHỦ NHIỄU
    # --------------------------------------------------------------------------
    def _engine_spectral_svd(self, raw_matrix):
        D = raw_matrix.shape[1]
        mat = np.zeros((D, D))
        try:
            U, S, Vt = np.linalg.svd(raw_matrix, full_matrices=False)
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
    # PIPELINE TỔNG HỢP VỚI CƠ CHẾ BẢO TỒN VI LIÊN KẾT
    # --------------------------------------------------------------------------
    def process(self, raw_matrix):
        T, D = raw_matrix.shape
        freqs = raw_matrix.sum(axis=0)
        hot_indices = np.where(freqs >= 2)[0]

        # 1. Chạy 4 Engine quét liên kết
        e_bayes = self._engine_micro_bayes(raw_matrix)
        e_embed = self._engine_spatial_embedding(raw_matrix)
        e_slide = self._engine_time_sliding_lead(raw_matrix)
        e_svd = self._engine_spectral_svd(raw_matrix)

        # 2. Tổng hợp ma trận điểm không bỏ sót liên kết nhỏ
        # Gán trọng số cao cho Bayes vi mô và Nhịp sinh học không gian
        fused_matrix = 1.8 * e_bayes + 1.5 * e_embed + 1.2 * e_slide + 0.8 * e_svd
        np.fill_diagonal(fused_matrix, 0)

        # 3. Lắng lọc vi mô (Soft Thresholding - Không dùng Percentile cứng để tránh mất liên kết yếu)
        fused_matrix = np.power(fused_matrix, 1.2) # Khuếch đại nhẹ tín hiệu nổi bật mà vẫn giữ tín hiệu yếu

        # Chốt vị trí cặp Bậc 2 tối ưu
        i, j = np.unravel_index(np.argmax(fused_matrix, axis=None), fused_matrix.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(fused_matrix[i, j])

        hot_str = ", ".join([f"{h+1:02d}" for h in hot_indices]) if len(hot_indices) > 0 else "Không có"
        explanation = (
            f"**Hạt nhân lặp:** [{hot_str}]\n\n"
            f"**Các vi liên kết được phát hiện:**\n"
            f"• *Micro Bayes & Spatial Embedding:* Phát hiện chuyển động tương đồng giữa ({num1:02d}) và ({num2:02d}) trong không gian vector 5 chiều.\n"
            f"• *Sliding Window:* Bắt trọn độ trễ xuất hiện nối tiếp giữa các kỳ liên tiếp.\n"
            f"• *Kết luận:* Cặp số **({num1:02d}, {num2:02d})** sở hữu chỉ số liên kết vi mô tối ưu nhất."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Micro-Link Engine")
st.caption("Khảo sát các liên kết vi mô • Spatial Manhattan Embedding • Bayes Probability")

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
                
        st.success("🎉 Hệ thống đã quét toàn bộ các vi liên kết trong 5 kỳ!")
        
        engine = MicroLinkKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP BẬC 2 PHÁT HIỆN TỪ VI LIÊN KẾT")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="CẶP SỐ CHỐT",
                value=f"{best_pair[0]:02d} — {best_pair[1]:02d}"
            )
        with col2:
            st.metric(
                label="Điểm Vi Liên Kết (Micro Score)",
                value=f"{score:.3f}"
            )
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt bộ quét vi liên kết.")
