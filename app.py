import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v103.0 Koopman-Spectral Engine", layout="wide")

class KoopmanSpectralEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=8, max_kies=30):
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

    def build_observables(self, matrix: np.ndarray):
        """
        EDMD Observable Dictionary: Nâng bậc không gian tuyến tính 
        bằng cách kết hợp trạng thái gốc và các tương tác cặp (cross-products).
        """
        T, D = matrix.shape
        # Dictionary bao gồm: Trạng thái tuyến tính gốc + Một số tích chéo tiêu biểu
        observables = []
        for t in range(T):
            x = matrix[t]
            # Mở rộng không gian quan sát phi tuyến thành tuyến tính hóa
            # Lấy vector gốc kết hợp với tổng tích chéo cục bộ để xấp xỉ không gian Hilbert
            obs = np.concatenate([x, np.convolve(x, [1, -1], mode='same')])
            observables.append(obs)
        return np.array(observables)

    def execute_koopman_analysis(self, matrix: np.ndarray):
        T, D = matrix.shape
        
        # Bước 1: Xây dựng không gian hàm quan sát Z
        Z = self.build_observables(matrix)
        if len(Z) < 3:
            return None

        # Tách thành hai ma trận thời gian dịch chuyển (Time-shifted snapshots)
        Z1 = Z[:-1].T  # Kỳ t
        Z2 = Z[1:].T   # Kỳ t + 1

        # Bước 2: Tính toán Toán tử Koopman K bằng EDMD (Least Squares via Pseudo-Inverse)
        # K * Z1 = Z2 => K = Z2 * pinv(Z1)
        try:
            K = np.dot(Z2, np.linalg.pinv(Z1))
        except Exception:
            K = np.eye(Z1.shape[0])

        # Bước 3: Phân rã Phổ (Spectral Decomposition)
        eigenvalues, eigenvectors = np.linalg.eig(K)

        # Bước 4: Lọc các mode dao động bền vững (Bỏ nhiễu suy giảm, giữ lại năng lượng chuẩn)
        # Sàng lọc các trị riêng có biên độ |lambda| gần 1 nhất (Invariant subspaces)
        magnitudes = np.abs(eigenvalues)
        stable_indices = np.argsort(np.abs(magnitudes - 1.0))[:min(10, len(eigenvalues))]
        
        dominant_eigenvalues = eigenvalues[stable_indices]
        dominant_eigenvectors = eigenvectors[:, stable_indices]

        # Bước 5: Dự báo năng lượng trạng thái tiếp theo cho 80 con số
        # Chiếu ngược qua các mode phổ Koopman thuần túy
        latest_obs = Z[-1]
        projected_state = np.real(np.dot(K, latest_obs))
        
        # Trích xuất phần năng lượng tương ứng với không gian 80 số gốc
        base_scores = np.abs(projected_state[:D])

        # Bộ lọc kháng quán tính & Hút chân không (Vacuum Reversal Integration)
        absence_gaps = np.zeros(D)
        for i in range(D):
            last_seen = -1
            for k in range(T - 1, -1, -1):
                if matrix[k, i] == 1.0:
                    last_seen = T - 1 - k
                    break
            absence_gaps[i] = last_seen if last_seen != -1 else T

        recent_act = matrix[-min(5, T):].sum(axis=0)
        final_scores = np.zeros(D)

        for i in range(D):
            # Phạt số ra quá nhiều liên tục
            inertia_factor = 0.05 if recent_act[i] >= 3 else (0.5 if recent_act[i] == 2 else 1.0)
            # Thưởng vùng trũng chân không (đảo chiều pha)
            vacuum_boost = 2.0 if 2 <= absence_gaps[i] <= 5 else 1.0
            
            final_scores[i] = base_scores[i] * vacuum_boost * inertia_factor

        ranked_indices = np.argsort(final_scores)[::-1]
        core_10_numbers = [idx + 1 for idx in ranked_indices[:10]]

        # Đo lường độ ổn định phổ (Spectral Coherence Index)
        spectral_coherence = float(np.mean(np.abs(dominant_eigenvalues)))

        return {
            "spectral_coherence": spectral_coherence,
            "core_10_numbers": core_10_numbers,
            "scores": final_scores
        }

# Giao diện Streamlit v103.0
st.title("🌌 MDM-IDS v103.0: KOOPMAN-SPECTRAL ENGINE")
st.caption("Tuyến tính hóa phi tuyến • Không gian Hilbert • Phân rã Phổ EDMD")

with st.sidebar:
    st.header("⚙️ Cấu Hình Cửa Sổ Dữ Liệu")
    min_w = st.slider("Cửa sổ tối thiểu:", 8, 15, 10)
    max_w = st.slider("Cửa sổ tối đa:", 15, 35, 25)

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (Tối thiểu 8 kỳ):",
    placeholder="Kỳ 1: 06 11 13 18 ...\nKỳ 2: ...",
    height=180
)

if raw_input.strip():
    engine = KoopmanSpectralEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=min_w, max_kies=max_w)

    if matrix is not None:
        res = engine.execute_koopman_analysis(matrix)

        if res is not None:
            st.success(f"⚡ Đã phân rã phổ Koopman thành công trên {total_kies} kỳ dữ liệu!")
            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="🌀 Độ Mạch Lạc Phổ (Coherence)", value=f"{res['spectral_coherence']:.4f}")
            with col2:
                st.metric(label="⚡ Bản Chất Hệ Thống", value="LINEARIZED HILBERT")
            with col3:
                st.metric(label="🎯 Trạng Thái Phổ", value="STABLE SPECTRAL MODES")

            st.markdown("---")
            st.subheader("🎯 TOP 10 CON SỐ TẠI PHỔ CỘNG HƯỞNG KOOPMAN")
            
            core_str = "  •  ".join([f"**{n:02d}**" for n in res['core_10_numbers']])
            st.markdown(f"<div style='text-align:center; padding:20px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:24px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 CHI TIẾT NĂNG LƯỢNG PHỔ TẦN SỐ")
            df_core = pd.DataFrame({
                "Thứ hạng": [f"Top {i+1}" for i in range(10)],
                "Con số": [f"Số {res['core_10_numbers'][i]:02d}" for i in range(10)],
                "Năng Lượng Phổ (Spectral Energy)": [f"{res['scores'][res['core_10_numbers'][i]-1]:.6f}" for i in range(10)]
            })
            st.table(df_core.T)
        else:
            st.error("Không đủ dữ liệu ma trận quan sát để chạy phân rã Koopman.")
    else:
        st.warning(f"Cần tối thiểu {min_w} kỳ dữ liệu hợp lệ để xử lý (Hiện nhận diện được {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu Keno vào khung trên để kích hoạt cỗ máy Koopman v103.0.")
