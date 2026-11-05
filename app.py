import numpy as np
import pandas as pd
import streamlit as st
import re

st.set_page_config(page_title="MDM-IDS v115.1: Spatial Topological Core", layout="wide")

class RobustSpatialEngine:
    def __init__(self, n_positions=80, n_drawn=20):
        self.D = n_positions
        self.K = n_drawn

    def parse_data(self, raw_text: str, min_kies=3, max_kies=20):
        """
        Trình phân tích dữ liệu đa năng: 
        Tự động quét tất cả các con số từ 1-80 có trong văn bản,
        cứ đủ 20 số liên tiếp sẽ được gộp thành 1 kỳ quay (bất chấp có chữ Kỳ hay không).
        """
        # Tìm tất cả các con số hợp lệ từ 1 đến 80 trong đoạn text dán vào
        all_tokens = re.findall(r'\b\d{1,2}\b', raw_text)
        all_numbers = [int(n) for n in all_tokens if 1 <= int(n) <= self.D]
        
        total_kies = len(all_numbers) // self.K

        if total_kies < min_kies:
            return None, total_kies

        kies_to_use = min(total_kies, max_kies)
        # Lấy các kỳ gần nhất
        used_numbers = all_numbers[-(kies_to_use * self.K):]

        matrix = np.zeros((kies_to_use, self.D), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * self.K : (k + 1) * self.K]:
                matrix[k, num - 1] = 1.0

        return matrix, kies_to_use

    def custom_wasserstein_1d(self, u_vec: np.ndarray, v_vec: np.ndarray) -> float:
        u_cdf = np.cumsum(u_vec) / (u_vec.sum() + 1e-9)
        v_cdf = np.cumsum(v_vec) / (v_vec.sum() + 1e-9)
        return float(np.sum(np.abs(u_cdf - v_cdf)))

    def analyze_spatial_signals(self, matrix: np.ndarray):
        T, D = matrix.shape
        
        # 1. Wasserstein W1
        w1_series = []
        for t in range(1, T):
            w1_val = self.custom_wasserstein_1d(matrix[t-1], matrix[t])
            w1_series.append(w1_val)
        
        latest_w1 = w1_series[-1] if len(w1_series) > 0 else 0.0
        w1_trend = "COMPRESSING (Co cụm dải)" if latest_w1 < np.mean(w1_series if len(w1_series) > 0 else [0]) else "EXPANDING (Bùng nổ dải)"

        # 2. Von Neumann Entropy
        half = D // 2
        entropies = []
        for t in range(T):
            vec = matrix[t]
            pA = vec[:half].sum() / self.K
            pB = vec[half:].sum() / self.K
            probs = np.array([pA, pB])
            probs = probs[probs > 0]
            S = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
            entropies.append(S)

        latest_entropy = entropies[-1]
        entropy_status = "BALANCED (Entropy cực đại)" if latest_entropy > 0.95 else "LOCALIZED (Nén lệch pha)"

        # 3. Spatial Fourier Spectrum
        fft_amps = np.abs(np.fft.fft(matrix, axis=1))
        power_spectrum = np.mean(fft_amps**2, axis=0)[:D // 2]
        dominant_k = int(np.argmax(power_spectrum[1:]) + 1)
        harmonic_period = round(D / dominant_k, 2)

        # 4. Top 5 & Phân rã phân thớ
        spatial_scores = np.mean(matrix[-min(3, T):], axis=0) * (1.0 + np.std(matrix, axis=0))
        top_5_indices = np.argsort(spatial_scores)[::-1][:5]
        final_top_5 = [int(idx + 1) for idx in top_5_indices]

        structural_breakdown = []
        for i in range(8):
            indices = list(range(i * 10, (i + 1) * 10))
            group_scores = {idx + 1: spatial_scores[idx] for idx in indices}
            sorted_g = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            structural_breakdown.append({
                "Phân thớ cấu trúc": f"Đầu {i} ({(i*10)+1:02d}-{(i+1)*10:02d})",
                "Top 2 số Không gian": f"{sorted_g[0][0]:02d}, {sorted_g[1][0]:02d}",
                "Xung lượng Tô-pô": round(float(np.mean([s for _, s in sorted_g[:2]])), 4)
            })

        return {
            "latest_w1": round(latest_w1, 4),
            "w1_trend": w1_trend,
            "latest_entropy": round(latest_entropy, 4),
            "entropy_status": entropy_status,
            "dominant_k": dominant_k,
            "harmonic_period": harmonic_period,
            "final_top_5": final_top_5,
            "structural_breakdown": structural_breakdown
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌐 MDM-IDS v115.1: ROBUST SPATIAL ENGINE")
st.caption("Tự động nhận diện chuỗi số • Wasserstein W1 • Von Neumann Entropy • Spatial Fourier Spectrum")

with st.sidebar:
    st.header("⚙️ Cấu Hình Không Gian")
    target_order = st.slider("Cửa sổ kỳ phân tích:", 3, 10, 4)

raw_input = st.text_area(
    f"Dán chuỗi kết quả Keno (Hệ thống tự động nhận diện mọi định dạng):",
    placeholder="Dán các dãy 20 số liên tiếp của từng kỳ vào đây...",
    height=200
)

if raw_input.strip():
    engine = RobustSpatialEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        res = engine.analyze_spatial_signals(matrix)

        st.success(f"⚛️ Đã phân tích thành công Tô-pô Không gian trên {total_kies} kỳ phát hiện!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📊 Wasserstein W1 (Độ biến dạng)", value=res['latest_w1'], delta=res['w1_trend'])
        with col2:
            st.metric(label="🌀 Von Neumann Entropy", value=res['latest_entropy'], delta=res['entropy_status'])
        with col3:
            st.metric(label="📻 Tần số Sóng hài k", value=f"Bậc k={res['dominant_k']}", delta=f"Chu kỳ ~{res['harmonic_period']} số")

        st.markdown("---")
        st.subheader("🎯 TOP 5 CON SỐ TÔ-PÔ TỐI ƯU CHO KỲ TIẾP THEO")
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['final_top_5']])
        st.markdown(f"<div style='text-align:center; padding:25px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 PHÂN RÃ XUNG LƯỢNG TÔ-PÔ THEO PHÂN THỚ")
        df_struct = pd.DataFrame(res['structural_breakdown'])
        st.table(df_struct)
    else:
        st.warning(f"Dữ liệu chưa đủ! Hệ thống mới quét được {total_kies} kỳ (cần tối thiểu {target_order} kỳ, mỗi kỳ đúng 20 số).")
else:
    st.info("Hãy dán dữ liệu các kỳ quay vào khung trên để khởi chạy hệ thống phân tích không gian!")
