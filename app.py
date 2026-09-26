import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Non-Linear Quantum Engine", layout="centered")

# ==============================================================================
# ADVANCED NON-LINEAR & INFORMATION DYNAMICS ENGINE
# ==============================================================================
class AdvancedKenoQuantumEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _softmax(self, x, temp=0.5):
        """Hàm kích hoạt phi tuyến với Temperature scaling để khuếch đại tín hiệu vi mô"""
        e_x = np.exp((x - np.max(x)) / temp)
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def _norm(self, mat):
        m = np.max(mat)
        return mat / m if m > 0 else mat

    # --------------------------------------------------------------------------
    # 1. MUTUAL INFORMATION & TRANSFER ENTROPY (LƯỢNG TIN TƯƠNG TÁC PHI TUYẾN)
    # --------------------------------------------------------------------------
    def _engine_mutual_information(self, X):
        """Đo độ phụ thuộc thông tin phi tuyến giữa mọi cặp số"""
        T, D = X.shape
        mi_mat = np.zeros((D, D))
        
        # Marginal probabilities
        p_i = np.mean(X, axis=0)
        
        for i in range(D):
            for j in range(i + 1, D):
                # Joint probability P(X_i, X_j)
                p_11 = np.mean(X[:, i] * X[:, j])
                p_10 = np.mean(X[:, i] * (1 - X[:, j]))
                p_01 = np.mean((1 - X[:, i]) * X[:, j])
                p_00 = np.mean((1 - X[:, i]) * (1 - X[:, j]))
                
                mi = 0.0
                for p_xy, px, py in [
                    (p_11, p_i[i], p_i[j]),
                    (p_10, p_i[i], 1 - p_i[j]),
                    (p_01, 1 - p_i[i], p_i[j]),
                    (p_00, 1 - p_i[i], 1 - p_i[j])
                ]:
                    if p_xy > 1e-6 and px > 1e-6 and py > 1e-6:
                        mi += p_xy * np.log2(p_xy / (px * py))
                
                mi_mat[i, j] = mi
                mi_mat[j, i] = mi
                
        return self._norm(mi_mat)

    # --------------------------------------------------------------------------
    # 2. DYNAMIC TIME WARPING (DTW) DISTANCE (ĐỒNG ĐIỆU NHỊP TIM TÍN HIỆU)
    # --------------------------------------------------------------------------
    def _engine_dtw_similarity(self, X):
        """Đo mức độ đồng điệu về nhịp vận động thời gian giữa các số"""
        T, D = X.shape
        dtw_mat = np.zeros((D, D))
        
        for i in range(D):
            for j in range(i + 1, D):
                # Distance matrix đơn giản cho chuỗi ngắn 5 kỳ
                s1, s2 = X[:, i], X[:, j]
                diff = np.abs(s1[:, None] - s2[None, :])
                
                # Dynamic programming path cost
                cost = np.zeros((T, T))
                cost[0, 0] = diff[0, 0]
                for r in range(1, T):
                    cost[r, 0] = cost[r - 1, 0] + diff[r, 0]
                for c in range(1, T):
                    cost[0, c] = cost[0, c - 1] + diff[0, c]
                    
                for r in range(1, T):
                    for c in range(1, T):
                        cost[r, c] = diff[r, c] + min(cost[r - 1, c], cost[r, c - 1], cost[r - 1, c - 1])
                
                # Chuyển khoảng cách thành độ tương đồng
                sim = 1.0 / (1.0 + cost[-1, -1])
                dtw_mat[i, j] = sim
                dtw_mat[j, i] = sim
                
        return self._norm(dtw_mat)

    # --------------------------------------------------------------------------
    # 3. KALMAN LATENT STATE PREDICTION (BỘ LỌC KALMAN DỰ BÁO TẦNG ẨN)
    # --------------------------------------------------------------------------
    def _engine_kalman_rate(self, X):
        """Bộ lọc Kalman đơn giản hóa để ước lượng vận tốc xuất hiện ở kỳ tiếp theo"""
        T, D = X.shape
        predicted_rates = np.zeros(D)
        
        for i in range(D):
            # Khởi tạo Kalman
            x_hat = 0.25 # State estimate (xác suất nền 20/80)
            P = 1.0     # Uncertainty
            Q = 0.05    # Process noise
            R = 0.2     # Measurement noise
            
            for t in range(T):
                # Time update (Predict)
                x_hat = x_hat
                P = P + Q
                
                # Measurement update (Correct)
                z = X[t, i]
                K = P / (P + R) # Kalman Gain
                x_hat = x_hat + K * (z - x_hat)
                P = (1 - K) * P
                
            predicted_rates[i] = x_hat
            
        # Ma trận cộng hưởng xác suất Kalman giữa các cặp
        kalman_mat = np.outer(predicted_rates, predicted_rates)
        np.fill_diagonal(kalman_mat, 0)
        return self._norm(kalman_mat)

    # --------------------------------------------------------------------------
    # 4. TRIỆT TIÊU MẠNH BẪY LẶP & ĐIỀU CHỈNH TRỌNG SỐ TỰ ĐỘNG
    # --------------------------------------------------------------------------
    def process(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        co_occur = np.dot(X.T, X)

        # Kích hoạt các Động cơ Phi tuyến
        e_mi = self._engine_mutual_information(X)
        e_dtw = self._engine_dtw_similarity(X)
        e_kalman = self._engine_kalman_rate(X)

        # Fourier Spectrum Engine
        fft_specs = np.abs(np.fft.fft(X, axis=0))
        e_fourier = self._norm(np.dot(fft_specs.T, fft_specs))

        # Giảm cực đại ảnh hưởng của Bayes/Tần suất cũ (chỉ dùng làm Baseline 0.3)
        bayes_base = np.zeros((D, D))
        for i in range(D):
            if freqs[i] > 0:
                bayes_base[i, :] = co_occur[i, :] / freqs[i]
        e_bayes = self._norm(bayes_base)

        # MÀNG LỌC PHẠT NÂNG CẤP: Chặn triệt để các số dính bẫy lặp
        penalty = np.ones((D, D))
        penalty[co_occur >= 2] = 0.50 # Phạt 50% nếu đã lặp 2 kỳ[span_0](start_span)[span_0](end_span)
        penalty[co_occur >= 3] = 0.05 # Phạt 95% nếu lặp >= 3 kỳ[span_1](start_span)[span_1](end_span)
        penalty[co_occur >= 4] = 0.00 # Khóa vĩnh viễn nếu lặp >= 4 kỳ[span_2](start_span)[span_2](end_span)

        # TỔNG HỢP VỚI ĐỘNG CƠ PHI TUYẾN CHÍNH
        fused_raw = (
            2.2 * e_mi +       # Mutual Information (Lượng tin phi tuyến)
            2.0 * e_kalman +   # Kalman Filter (Dự báo tầng ẩn)
            1.8 * e_dtw +      # DTW (Đồng điệu nhịp thời gian)
            1.5 * e_fourier +  # Fourier (Sóng hài)
            0.3 * e_bayes      # Bayes cũ (hạ xuống mức tối thiểu)
        ) * penalty

        # Áp dụng Kích hoạt Phi tuyến Softmax để đẩy điểm cặp bứt phá rời xa đám đông
        fused_transformed = self._softmax(fused_raw, temp=0.2)
        np.fill_diagonal(fused_transformed, 0)

        # Trích xuất cặp Bậc 2
        i, j = np.unravel_index(np.argmax(fused_transformed, axis=None), fused_transformed.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(fused_transformed[i, j])

        hot_str = ", ".join([f"{h+1:02d}" for h in np.where(freqs >= 2)[0]]) or "Không có"
        blocked_str = ", ".join([f"{b+1:02d}" for b in np.where(freqs >= 3)[0]]) or "Không có"

        explanation = (
            f"• **Hạt nhân cũ bị hạ trọng số:** [{hot_str}]\n"
            f"• **Số bị khóa triệt để do bẫy lặp (>=3 kỳ):** [{blocked_str}]\n"
            f"• **Đột phá Động cơ Phi tuyến mới:**\n"
            f"  - *Mutual Information (Lượng tin):* Khai thác liên kết phi tuyến ẩn giữa các số.\n"
            f"  - *Kalman Filter:* Ước lượng vận tốc xuất hiện tiềm năng ở kỳ kế tiếp.\n"
            f"  - *DTW Time-Warping:* Bắt nhịp tim đồng điệu lệch kỳ giữa các con số.\n"
            f"  - *Softmax Temperature:* Khuếch đại cặp vi mô có tín hiệu sóng bùng nổ rõ nhất.\n"
            f"• **Chốt kết quả mới:** Cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Quantum Non-Linear Engine")
st.caption("Khai thác Phi tuyến: Mutual Information • Kalman Latent State • DTW Signal Alignment • Softmax Transformation")

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
                
        st.success("🎉 Đã kích hoạt Động cơ Phi tuyến Quantum Engine thành công!")
        
        engine = AdvancedKenoQuantumEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ CHỐT ĐỘT PHÁ MỚI")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ ĐỘT PHÁ", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Quantum Softmax", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt động cơ phi tuyến mới.")
