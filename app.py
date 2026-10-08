import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v16.0 Modular Pipeline Architecture", layout="wide")

# ==============================================================================
# LỚP 0: UNIFIED DATA MATRIX PIPELINE (MÔ HÌNH DỮ LIỆU ĐẦU VÀO)
# ==============================================================================
class KenoDataPipeline:
    """Chịu trách nhiệm tiếp nhận, chuẩn hóa và trích xuất cấu trúc ma trận 2D/3D."""
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_and_build_matrix(self, raw_text: str, window_min=6, window_max=8):
        cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text.strip(), flags=re.IGNORECASE)
        all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= self.D]
        total_kies = len(all_numbers) // 20

        if total_kies < window_min:
            return None, total_kies

        kies_to_use = min(total_kies, window_max)
        used_numbers = all_numbers[-kies_to_use * 20:]

        matrix = np.zeros((kies_to_use, self.D), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0

        return matrix, kies_to_use


# ==============================================================================
# LỚP 1: SIGNAL PROCESSING & ENTROPY ENGINE
# ==============================================================================
class SignalEntropyLayer:
    """Xử lý tín hiệu thô, độ lệch pha nhị nguyên và entropy Shannon thời gian thực."""
    def process(self, X):
        T, D = X.shape
        p1 = np.mean(X == 1, axis=0) + 1e-9
        p0 = np.mean(X == 0, axis=0) + 1e-9

        # Shannon Entropy
        entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))

        # Duality Alignment
        even_mask = np.array([1 if (i + 1) % 2 == 0 else 0 for i in range(D)])
        large_mask = np.array([1 if (i + 1) > 40 else 0 for i in range(D)])
        recent_k = X[-1]

        even_ratio = np.sum(recent_k * even_mask) / 20.0
        large_ratio = np.sum(recent_k * large_mask) / 20.0

        duality_bias = np.zeros(D)
        for i in range(D):
            b_even = (1.0 - even_ratio) if even_mask[i] else even_ratio
            b_large = (1.0 - large_ratio) if large_mask[i] else large_ratio
            duality_bias[i] = 0.5 * (b_even + b_large)

        return entropy, duality_bias


# ==============================================================================
# LỚP 2: MULTI-AGENT CONSENSUS & TENSOR SVD LAYER
# ==============================================================================
class MultiAgentConsensusLayer:
    """Hợp nhất đa Agent: Tensor SVD Entanglement, Phase Shift, Wavefunction."""
    def process(self, X, entropy, duality_bias):
        T, D = X.shape
        freqs = X.sum(axis=0)

        # Agent Tensor SVD
        if T >= 4:
            cov = np.corrcoef(X.T)
            cov = np.nan_to_num(cov)
            entanglement = np.sum(np.abs(cov), axis=1) / float(D)
        else:
            entanglement = np.ones(D)

        # Agent Phase Shift
        phase_scores = np.ones(D)
        for i in range(D):
            if freqs[i] in [1, 2, 3]:
                phase_scores[i] = 1.90  # Vùng ranh giới pha bộc phát
            elif freqs[i] == 0:
                phase_scores[i] = 0.30  # Số đóng băng
            elif freqs[i] >= 4:
                phase_scores[i] = 0.15  # Bão hòa

        # Chuẩn hóa
        norm_ent = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-9)
        norm_dual = (duality_bias - duality_bias.min()) / (duality_bias.max() - duality_bias.min() + 1e-9)
        norm_svd = (entanglement - entanglement.min()) / (entanglement.max() - entanglement.min() + 1e-9)

        # Tích chéo hội tụ 3 dòng tín hiệu
        raw_consensus = (norm_ent ** 1.1) * (norm_dual ** 1.3) * (norm_svd ** 1.5) * phase_scores
        return raw_consensus


# ==============================================================================
# LỚP 3: ANTI-REPETITION & COMBINATORIAL ROUTER
# ==============================================================================
class CombinatorialRouterLayer:
    """Lọc triệt tiêu lặp số T-1 và tổ hợp xuất kết quả chuyên biệt cho Bậc 2 & Bậc 3."""
    def route_bac2_bac3(self, X, consensus_scores):
        D = X.shape[1]
        final_scores = consensus_scores.copy()

        # Triệt tiêu lặp số ở kỳ gần nhất T-1
        for i in range(D):
            if X[-1, i] == 1:
                final_scores[i] *= 0.20

        ranked_indices = np.argsort(final_scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]

        # Phân bổ Dual-Core Alpha / Beta
        bac2_alpha = sorted([ranked_numbers[0], ranked_numbers[2]])
        bac2_beta = sorted([ranked_numbers[1], ranked_numbers[3]])
        bac2_backup = sorted([ranked_numbers[4], ranked_numbers[5]])

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
# STREAMLIT UI - PHIÊN BẢN v16.0 ARCHITECTURE
# ==============================================================================
st.title("⚡ MDM-IDS v16.0: MODULAR PIPELINE ARCHITECTURE")
st.caption("Kiến Trúc Phân Lớp Thuật Toán Tối Ưu • Chuẩn Hóa Matrix Pipeline • Chuyên Biệt Bậc 2 & Bậc 3")

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu (6-8 kỳ gần nhất):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=160
)

if raw_input.strip():
    pipeline = KenoDataPipeline()
    matrix, total_kies = pipeline.parse_and_build_matrix(raw_input)

    if matrix is not None:
        # Lớp 1
        l1 = SignalEntropyLayer()
        entropy, duality = l1.process(matrix)

        # Lớp 2
        l2 = MultiAgentConsensusLayer()
        raw_consensus = l2.process(matrix, entropy, duality)

        # Lớp 3
        l3 = CombinatorialRouterLayer()
        res = l3.route_bac2_bac3(matrix, raw_consensus)

        st.success(f"⚡ Xử lý thành công qua 4 lớp pipeline độc lập với {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        # KHỐI BẬC 2
        st.subheader("🔥 BẬC 2 CHUYÊN BIỆT (ĂN 90.000 VNĐ)")
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

        # KHỐI BẬC 3
        st.subheader("⚡ BẬC 3 CHUYÊN BIỆT (THƯỞNG 200.000 VNĐ / HOÀN VỐN 20.000 VNĐ)")
        c3a, c3b, c3c = st.columns(3)
        with c3a:
            st.info(f"Dàn Alpha:\n### **{' - '.join([f'{n:02d}' for n in res['bac3_alpha']])}**")
        with c3b:
            st.warning(f"Dàn Beta (Bộc phát):\n### **{' - '.join([f'{n:02d}' for n in res['bac3_beta']])}**")
        with c3c:
            st.write(f"Dàn Dự phòng:\n### **{' - '.join([f'{n:02d}' for n in res['bac3_backup']])}**")

        st.markdown("---")
        st.subheader("📊 BẢNG TOP 12 CON SỐ TỔNG HỢP QUA PIPELINE")
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(12)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(12)],
            "Cấu trúc Router": ["Alpha" if i % 2 == 0 else "Beta (Bộc phát)" for i in range(12)],
            "Điểm Tín Hiệu v16": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(12)]
        })
        st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu 6 kỳ cuốn chiếu để khởi chạy Pipeline (Hiện có {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu 6-8 kỳ Keno vào khung trên để thực thi pipeline v16.0.")
