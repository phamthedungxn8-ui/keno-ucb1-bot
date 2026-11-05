import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v115.0: Spatial Topological Core", layout="wide")

class PureNumPySpatialEngine:
    def __init__(self, n_positions=80, n_drawn=20):
        self.D = n_positions
        self.K = n_drawn

    def parse_data(self, raw_text: str, min_kies=3, max_kies=20):
        cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text.strip(), flags=re.IGNORECASE)
        all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= self.D]
        total_kies = len(all_numbers) // 20

        if total_kies < min_kies:
            return None, total_kies

        kies_to_use = min(total_kies, max_kies)
        used_numbers = all_numbers[-kies_to_use * 20:]

        matrix = np.zeros((kies_to_use, self.D), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0

        return matrix, kies_to_use

    def custom_wasserstein_1d(self, u_vec: np.ndarray, v_vec: np.ndarray) -> float:
        """Hàm tính khoảng cách Wasserstein W1 thuần NumPy không phụ thuộc scipy."""
        u_cdf = np.cumsum(u_vec) / (u_vec.sum() + 1e-9)
        v_cdf = np.cumsum(v_vec) / (v_vec.sum() + 1e-9)
        return float(np.sum(np.abs(u_cdf - v_cdf)))

    def analyze_spatial_signals(self, matrix: np.ndarray):
        T, D = matrix.shape
        
        # 1. Tính chuỗi khoảng cách Wasserstein W1 giữa các kỳ
        w1_series = []
        for t in range(1, T):
            w1_val = self.custom_wasserstein_1d(matrix[t-1], matrix[t])
            w1_series.append(w1_val)
        
        latest_w1 = w1_series[-1] if len(w1_series) > 0 else 0.0
        w1_trend = "COMPRESSING (Co cụm dải)" if latest_w1 < np.mean(w1_series if len(w1_series) > 0 else [0]) else "EXPANDING (Bùng nổ dải)"

        # 2. Von Neumann Entropy (Phân vùng Nửa A / Nửa B)
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

        # 3. Phân tích Phổ Fourier Không gian (Spatial Fourier Spectrum via NumPy)
        fft_amps = np.abs(np.fft.fft(matrix, axis=1))
        power_spectrum = np.mean(fft_amps**2, axis=0)[:D // 2]
        dominant_k = int(np.argmax(power_spectrum[1:]) + 1)
        harmonic_period = round(D / dominant_k, 2)

        # 4. Phân rã cấu trúc Top 5 số có điểm cộng hưởng không gian cao nhất
        spatial_scores = np.mean(matrix[-min(3, T):], axis=0) * (1.0 + np.std(matrix, axis=0))
        top_5_indices = np.argsort(spatial_scores)[::-1][:5]
        final_top_5 = [int(idx + 1) for idx in top_5_indices]

        # Phân rã 8 phân thớ
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
st.title("🌐 MDM-IDS v115.0: SPATIAL TOPOLOGICAL ENGINE")
st.caption("Khoảng cách Wasserstein W1 • Von Neumann Entropy • Spatial Fourier Spectrum")

with st.sidebar:
    st.header("⚙️ Cấu Hình Không Gian")
    target_order = st.slider("Cửa sổ kỳ phân tích:", 3, 10, 4)

raw_input = st.text_area(
    f"Dán thủ công dữ liệu Keno (Chỉ cần từ {target_order} kỳ):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = PureNumPySpatialEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        res = engine.analyze_spatial_signals(matrix)

        st.success(f"⚛️ Đã phân tích thành công Tô-pô Không gian trên {total_kies} kỳ!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📊 Wasserstein W1 (Độ biến dạng)", value=res['latest_w1'], delta=res['w1_trend'])
        with col2:
            st.metric(label="🌀 Von Neumann Entropy", value=res['latest_entropy'], delta=res['entropy_status'])
        with col3:
            st.metric(label="📻 Tần số Sóng hài k", value=f"Bậc k={res['dominant_k']}", delta=f"Chu kỳ ~{res['harmonic_period']} số")

        st.markdown("---")
        st.subheader("🎯 TOP 5 CON SỐ TÔ-PÔ TỐI ƯU")
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['final_top_5']])
        st.markdown(f"<div style='text-align:center; padding:25px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 PHÂN RÃ XUNG LƯỢNG TÔ-PÔ THEO PHÂN THỚ")
        df_struct = pd.DataFrame(res['structural_breakdown'])
        st.table(df_struct)
    else:
        st.warning(f"Cần tối thiểu {target_order} kỳ dữ liệu.")
else:
    st.info("Dán dữ liệu từ 3-4 kỳ Keno để quét trường không gian Tô-pô!")
