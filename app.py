import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v16.5 Multi-Timeframe Architecture", layout="wide")

# ==============================================================================
# LỚP 0: MULTI-TIMEFRAME DATA PIPELINE (10 - 30 KỲ)
# ==============================================================================
class MultiTimeframePipeline:
    """Xử lý dữ liệu đầu vào 10-30 kỳ với cấu trúc ma trận đa tầng và suy giảm theo thời gian."""
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_and_build_tensor(self, raw_text: str, min_kies=10, max_kies=30):
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


# ==============================================================================
# LỚP 1: MULTI-WINDOW SIGNAL & ENTROPY LAYER
# ==============================================================================
class MultiWindowSignalLayer:
    """Tính toán tín hiệu trên 3 cửa sổ: Ngắn (6), Trung (15), Dài (30)."""
    def process(self, X):
        T, D = X.shape
        
        # Thiết lập các cửa sổ thời gian
        w_short = min(6, T)
        w_med = min(15, T)
        w_long = T

        # Trọng số thời gian (Exponential Time-Decay)
        weights = np.exp(np.linspace(-1.5, 0, T))
        weights /= weights.sum()

        # Tần suất có trọng số thời gian
        weighted_freqs = np.dot(weights, X)

        # Ma trận tương quan co-occurrence trong cửa sổ trung & dài
        X_med = X[-w_med:]
        cov_med = np.corrcoef(X_med.T)
        cov_med = np.nan_to_num(cov_med)

        # Duality Bias trên cửa sổ ngắn
        recent_k = X[-1]
        even_mask = np.array([1 if (i + 1) % 2 == 0 else 0 for i in range(D)])
        large_mask = np.array([1 if (i + 1) > 40 else 0 for i in range(D)])
        
        even_ratio = np.sum(recent_k * even_mask) / 20.0
        large_ratio = np.sum(recent_k * large_mask) / 20.0

        duality_bias = np.zeros(D)
        for i in range(D):
            b_even = (1.0 - even_ratio) if even_mask[i] else even_ratio
            b_large = (1.0 - large_ratio) if large_mask[i] else large_ratio
            duality_bias[i] = 0.5 * (b_even + b_large)

        return weighted_freqs, cov_med, duality_bias


# ==============================================================================
# LỚP 2: CONSENSUS & PAIR/TRIPLET DENSITY LAYER (ĐẶC THỤ BẬC 2 & BẬC 3)
# ==============================================================================
class PairDensityConsensusLayer:
    """Tối ưu mật độ liên kết cặp cho Bậc 2 và bộ ba cho Bậc 3."""
    def process(self, X, weighted_freqs, cov_matrix, duality_bias):
        T, D = X.shape
        freqs_short = X[-6:].sum(axis=0) if T >= 6 else X.sum(axis=0)

        # Mật độ tương quan từ ma trận Covariance
        pair_density = np.sum(np.maximum(0, cov_matrix), axis=1) / float(D)

        # Khối lọc pha bộc phát trên cửa sổ ngắn
        phase_scores = np.ones(D)
        for i in range(D):
            if freqs_short[i] in [1, 2]:
                phase_scores[i] = 1.95  # Vùng ranh giới pha bộc phát cực cao
            elif freqs_short[i] == 0:
                phase_scores[i] = 0.35  # Phạt số đóng băng
            elif freqs_short[i] >= 4:
                phase_scores[i] = 0.15  # Phạt số bão hòa

        # Chuẩn hóa
        norm_wf = (weighted_freqs - weighted_freqs.min()) / (weighted_freqs.max() - weighted_freqs.min() + 1e-9)
        norm_pair = (pair_density - pair_density.min()) / (pair_density.max() - pair_density.min() + 1e-9)
        norm_dual = (duality_bias - duality_bias.min()) / (duality_bias.max() - duality_bias.min() + 1e-9)

        # Tổng hợp xung lực v16.5
        scores = (norm_wf * 0.30 + norm_pair * 0.40 + norm_dual * 0.30) * phase_scores
        return scores


# ==============================================================================
# LỚP 3: ANTI-REPETITION & ROUTER FOR BAC 2 & BAC 3
# ==============================================================================
class Bac2Bac3RouterLayer:
    """Lọc triệt tiêu lặp kỳ T-1 và xuất dàn Dual-Core cho Bậc 2 và Bậc 3."""
    def route(self, X, scores):
        D = X.shape[1]
        final_scores = scores.copy()

        # Triệt tiêu lặp số ở kỳ T-1
        for i in range(D):
            if X[-1, i] == 1:
                final_scores[i] *= 0.20

        ranked_indices = np.argsort(final_scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]

        # Dàn Bậc 2
        bac2_alpha = sorted([ranked_numbers[0], ranked_numbers[2]])
        bac2_beta = sorted([ranked_numbers[1], ranked_numbers[3]])
        bac2_backup = sorted([ranked_numbers[4], ranked_numbers[5]])

        # Dàn Bậc 3
        bac3_alpha = sorted([ranked_numbers[0], ranked_numbers[2], ranked_numbers[4]])
        bac3_beta = sorted([ranked_numbers[1], ranked_numbers[3], ranked_numbers[5]])
        bac3_backup = sorted([ranked_numbers[6], ranked_numbers[7], ranked_numbers[8]])

        return {
            "bac2_alpha": bac2_alpha,
            "bac2_beta": bac2_beta,
            "bac2_backup": bac2_backup,
            "bac3_alpha": bac3_alpha,
            "bac3_beta": bac3_beta,
            "bac3_backup": bac3_backup,
            "scores": final_scores,
            "ranked_all": ranked_numbers
        }


# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ MDM-IDS v16.5: MULTI-TIMEFRAME PIPELINE (10–30 KỲ)")
st.caption("Kiến Trúc Dữ Liệu Đa Tần Số • Exponential Time-Decay • Chuyên Biệt Bậc 2 & Bậc 3")

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu (Nên nhập từ 10 đến 30 kỳ):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...\n...\nKỳ 20: ...",
    height=200
)

if raw_input.strip():
    pipeline = MultiTimeframePipeline()
    matrix, total_kies = pipeline.parse_and_build_tensor(raw_input, min_kies=10, max_kies=30)

    if matrix is not None:
        # Lớp 1
        l1 = MultiWindowSignalLayer()
        w_freqs, cov_matrix, duality = l1.process(matrix)

        # Lớp 2
        l2 = PairDensityConsensusLayer()
        raw_scores = l2.process(matrix, w_freqs, cov_matrix, duality)

        # Lớp 3
        l3 = Bac2Bac3RouterLayer()
        res = l3.route(matrix, raw_scores)

        st.success(f"⚡ Đã phân tích thành công ma trận đa thời gian gồm {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        # HIỂN THỊ BẬC 2
        st.subheader("🔥 MỤC TIÊU BẬC 2 (CHỌN 2 - ĂN 90.000 VNĐ)")
        c2a, c2b, c2c = st.columns(3)
        with c2a:
            st.markdown(f"<div style='text-align:center; padding:12px; background:#0A192F; border-radius:8px; border:2px solid #00F0FF;'>"
                        f"<span style='color:#00F0FF; font-weight:bold;'>BẬC 2 - ALPHA</span>"
                        f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_alpha'][0]:02d} — {res['bac2_alpha'][1]:02d}</h2></div>", unsafe_allow_html=True)
        with c2b:
            st.markdown(f"<div style='text-align:center; padding:12px; background:#1A0903; border-radius:8px; border:2px solid #FF5500;'>"
                        f"<span style='color:#FF5500; font-weight:bold;'>BẬC 2 - BETA (BÙNG NỔ)</span>"
                        f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_beta'][0]:02d} — {res['bac2_beta'][1]:02d}</h2></div>", unsafe_allow_html=True)
        with c2c:
            st.markdown(f"<div style='text-align:center; padding:12px; background:#111; border-radius:8px; border:1px solid #555;'>"
                        f"<span style='color:#AAA; font-weight:bold;'>BẬC 2 - DỰ PHÒNG</span>"
                        f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_backup'][0]:02d} — {res['bac2_backup'][1]:02d}</h2></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # HIỂN THỊ BẬC 3
        st.subheader("⚡ MỤC TIÊU BẬC 3 (CHỌN 3 - THƯỞNG 200.000 VNĐ / HOÀN VỐN 20.000 VNĐ)")
        c3a, c3b, c3c = st.columns(3)
        with c3a:
            st.info(f"Dàn Alpha:\n### **{' - '.join([f'{n:02d}' for n in res['bac3_alpha']])}**")
        with c3b:
            st.warning(f"Dàn Beta (Bộc phát):\n### **{' - '.join([f'{n:02d}' for n in res['bac3_beta']])}**")
        with c3c:
            st.write(f"Dàn Dự phòng:\n### **{' - '.join([f'{n:02d}' for n in res['bac3_backup']])}**")

        st.markdown("---")
        st.subheader("📊 TOP 12 CON SỐ XUNG LỰC ĐA TẦN SỐ")
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(12)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(12)],
            "Cấu trúc Router": ["Alpha" if i % 2 == 0 else "Beta (Bộc phát)" for i in range(12)],
            "Điểm Xung Lực v16.5": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(12)]
        })
        st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu 10 kỳ để phân tích đa tần số (Hiện có {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu 10-30 kỳ Keno vào khung trên để khởi chạy mô hình v16.5.")
