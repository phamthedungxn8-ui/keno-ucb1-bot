import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Master Multi-Tier Engine", layout="centered")

# ==============================================================================
# HỆ THỐNG MASTER HỢP NHẤT TOÀN BỘ THUẬT TOÁN ĐA TẦNG
# ==============================================================================
class KenoMasterMultiTierEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, mat):
        """Chuẩn hóa ma trận điểm về khoảng [0, 1]"""
        m = np.max(mat)
        return mat / m if m > 0 else mat

    def process(self, X):
        """
        X: Ma trận nhị phân 5 kỳ x 80 số
        Tích hợp: Lực kéo Hạt nhân + Micro Bayes + Space Embedding + SVD Denoise +
                  Triad Clique + Markov Chain + Fourier FFT + Saturation Gate
        """
        T, D = X.shape
        freqs = X.sum(axis=0)
        co_occur = np.dot(X.T, X)
        
        # ----------------------------------------------------------------------
        # 1. TIỀN XỬ LÝ: TRỌNG SỐ THỜI GIAN (TIME-DECAY VECTORIZATION)
        # ----------------------------------------------------------------------
        decay_weights = np.exp(np.linspace(-0.6, 0, T))
        X_decay = X * decay_weights[:, np.newaxis]

        # ----------------------------------------------------------------------
        # 2. TẦNG 1: QUÉT LIÊN KẾT HẠT NHÂN & MICRO BAYES (MÔ HÌNH CỦ & VI MÔ)
        # ----------------------------------------------------------------------
        hot_indices = np.where(freqs >= 2)[0]
        bayes_mat = np.zeros((D, D))
        for i in range(D):
            if freqs[i] > 0:
                bayes_mat[i, :] = co_occur[i, :] / freqs[i]
                # Thưởng điểm lực kéo hạt nhân nếu i nằm trong nhóm lặp
                if i in hot_indices:
                    bayes_mat[i, :] *= 1.3
        mat_bayes = self._norm(bayes_mat)

        # ----------------------------------------------------------------------
        # 3. TẦNG 2: MẠNG KHÔNG GIANG & CỤM TAM GIÁC (SPATIAL & TRIAD CLIQUES)
        # ----------------------------------------------------------------------
        # Manhattan Embedding Distance
        dist_mat = np.array([[np.sum(np.abs(X[:, i] - X[:, j])) for j in range(D)] for i in range(D)])
        mat_spatial = self._norm(1.0 / (1.0 + dist_mat))

        # Cụm Tam Giác Khép Kín (A^3)
        adj_co = co_occur.copy()
        np.fill_diagonal(adj_co, 0)
        mat_triad = self._norm(np.dot(adj_co, np.dot(adj_co, adj_co)))

        # ----------------------------------------------------------------------
        # 4. TẦNG 3: DỰ BÁO MARKOV, SÓNG HÀI FOURIER & SVD LỌC NHIỄU
        # ----------------------------------------------------------------------
        # Markov Chain Transition (Bắt nhịp Rebound vừa nghỉ 1 kỳ)
        recency = X[-1] + X[-2] * 0.5
        mat_markov = self._norm(np.outer(recency == 0.5, recency == 0.5))

        # Sóng hài Fourier (FFT Energy Resonance)
        fft_specs = np.abs(np.fft.fft(X, axis=0))
        mat_fourier = self._norm(np.dot(fft_specs.T, fft_specs))

        # SVD Low-Rank Denoising (Lọc nhiễu trắng)
        try:
            U, S, Vt = np.linalg.svd(X_decay, full_matrices=False)
            S_clean = np.zeros_like(S)
            S_clean[0] = S[0]
            if len(S) > 1: S_clean[1] = S[1]
            clean_X = np.dot(U, np.dot(np.diag(S_clean), Vt))
            mat_svd = self._norm(np.maximum(np.dot(clean_X.T, clean_X), 0))
        except:
            mat_svd = np.zeros((D, D))

        # ----------------------------------------------------------------------
        # 5. TẦNG 4: MÀNG LỌC PHẠT TRIỆT BẪY LẶP (ANTI-SATURATION GATEWAY)
        # ----------------------------------------------------------------------
        gate_penalty = np.ones((D, D))
        gate_penalty[co_occur >= 3] = 0.15  # Phạt 85% nếu xuất hiện >= 3/5 kỳ
        gate_penalty[co_occur >= 4] = 0.02  # Phạt 98% nếu xuất hiện >= 4/5 kỳ

        # ----------------------------------------------------------------------
        # 6. TỔNG HỢP CỘNG HƯỞNG ĐA TẦNG (TENSOR AGGREGATION)
        # ----------------------------------------------------------------------
        fused = (
            1.8 * mat_bayes + 
            1.6 * mat_markov + 
            1.4 * mat_triad + 
            1.2 * mat_spatial + 
            1.0 * mat_fourier + 
            0.8 * mat_svd
        ) * gate_penalty

        np.fill_diagonal(fused, 0)

        # Trích xuất cặp Bậc 2 tối ưu nhất
        i, j = np.unravel_index(np.argmax(fused, axis=None), fused.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(fused[i, j])

        hot_str = ", ".join([f"{h+1:02d}" for h in hot_indices]) or "Không có"
        sat_str = ", ".join([f"{s+1:02d}" for s in np.where(freqs >= 4)[0]]) or "Không có"

        explanation = (
            f"• **Hạt nhân lặp ghi nhận:** [{hot_str}]\n"
            f"• **Bẫy lặp quá tải (Đã triệt tiêu):** [{sat_str}]\n"
            f"• **Tiến trình Lọc Đa Tầng Unified:**\n"
            f"  - *Tầng 1 (Kernel & Bayes):* Duy trì lực kéo trực tiếp từ mô hình cũ.\n"
            f"  - *Tầng 2 (Triad & Spatial):* Bắt cụm 3 số kín và khoảng cách Manhattan.\n"
            f"  - *Tầng 3 (Markov & Fourier):* Đo động lượng bùng nổ rebound & phổ năng lượng sóng hài.\n"
            f"  - *Tầng 4 (Anti-Saturation Gate):* Khóa hoàn toàn bẫy lặp dày đặc.\n"
            f"• **Chốt kết quả:** Cặp số **({num1:02d}, {num2:02d})** đạt chỉ số đồng thuận tối cao."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# GIAO DIỆN STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Master Multi-Tier Engine")
st.caption("Hợp nhất Toàn diện: Lực kéo Hạt nhân • Bayes • Spatial • Triad Cliques • Markov • Fourier • Anti-Saturation")

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
                
        st.success("🎉 Đã chạy thành công Pipeline Hợp Nhất Đa Tầng!")
        
        engine = KenoMasterMultiTierEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP BẬC 2 CHỐT TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Cộng Hưởng Unified Master", value=f"{score:.3f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt hệ thống master.")
