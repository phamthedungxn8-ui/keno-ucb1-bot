import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v36.0 Advanced Fano-Algebraic Engine", layout="wide")

# ==============================================================================
# LỚP 0: MULTI-LAYER FILTER (TRÍCH XUẤT TOP 14 SỐ TIỀM NĂNG)
# ==============================================================================
class AdvancedMultiLayerFilter:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def extract_top_pool(self, raw_text: str, min_kies=5, max_kies=15):
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

        freqs = matrix.sum(axis=0)
        recent_k = matrix[-1]
        
        scores = np.zeros(self.D)
        for i in range(self.D):
            penalty = 0.15 if recent_k[i] == 1.0 else 1.0
            if freqs[i] in [1, 2]:
                scores[i] = 2.2 * penalty / (freqs[i] + 0.5)
            else:
                scores[i] = 0.4 * penalty / (freqs[i] + 1.0)

        ranked_indices = np.argsort(scores)[::-1]
        top_14 = [idx + 1 for idx in ranked_indices[:14]]

        return matrix, top_14, scores


# ==============================================================================
# LỚP 1: ADVANCED FANO ALGEBRAIC GEOMETRY ENGINE
# ==============================================================================
class AdvancedFanoAlgebraicEngine:
    """Áp dụng cấu trúc đồng phẳng Fano toàn cục trên 14 số để tìm 2 cặp Bậc 2 Alpha & Beta tối ưu."""
    def __init__(self):
        # 7 đường thẳng chuẩn của Mặt phẳng Fano PG(2,2) biểu diễn qua chỉ số 0-6
        self.fano_lines = [
            (0, 1, 2), (3, 4, 5), (0, 3, 6), 
            (1, 4, 6), (2, 5, 6), (0, 4, 5), (1, 3, 5)
        ]

    def compute_fano_pairs(self, top_14_numbers, scores):
        # Chia 14 số thành 2 không gian Fano 7 điểm tương đương
        space_1 = top_14_numbers[:7]
        space_2 = top_14_numbers[7:]

        def extract_best_collinear_pair(space):
            node_map = {i: space[i] for i in range(7)}
            best_pair = None
            max_val = -1.0
            
            for line in self.fano_lines:
                n1, n2, n3 = node_map[line[0]], node_map[line[1]], node_map[line[2]]
                pairs = [(n1, n2), (n2, n3), (n1, n3)]
                for p in pairs:
                    # Điểm hình học Fano kết hợp điểm xung lực đa tầng
                    geom_weight = 1.2 # Hệ số cộng hưởng hình học hữu hạn
                    val = (scores[p[0]-1] + scores[p[1]-1]) * geom_weight
                    if val > max_val:
                        max_val = val
                        best_pair = sorted(list(p))
            return best_pair, max_val

        pair_alpha, score_a = extract_best_collinear_pair(space_1)
        pair_beta, score_b = extract_best_collinear_pair(space_2)

        return pair_alpha, pair_beta, space_1, space_2


# ==============================================================================
# STREAMLIT UI (v36.0)
# ==============================================================================
st.title("📐 MDM-IDS v36.0: ADVANCED FANO-ALGEBRAIC ENGINE")
st.caption("Cải Tiến Hình Học Hữu Hạn • Ma Trận Tương Thích Đồng Phẳng Fano • Tối Ưu Bậc 2 Alpha & Beta")

with st.sidebar:
    st.header("⚙️ Cấu Hình Không Gian")
    min_kies = st.slider("Số kỳ đầu vào tối thiểu:", 5, 10, 5)

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (Từ 5 kỳ trở lên):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=160
)

if raw_input.strip():
    pipeline = AdvancedMultiLayerFilter()
    res_tuple = pipeline.extract_top_pool(raw_input, min_kies=min_kies)

    if res_tuple[0] is not None:
        matrix, top_14, scores = res_tuple
        
        fano_engine = AdvancedFanoAlgebraicEngine()
        pair_alpha, pair_beta, space_1, space_2 = fano_engine.compute_fano_pairs(top_14, scores)

        st.success(f"⚡ Đã hoàn tất phân rã 14 số qua không gian hình học đại số Fano v36.0!")
        st.markdown("---")

        st.info(f"🧩 **Tập 14 số từ bộ lọc đa tầng chuẩn hóa:** `{top_14}`")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔥 BẬC 2 - ALPHA (KHÔNG GIAN FANO 1)")
            st.markdown(f"<div style='text-align:center; padding:15px; background:#0A192F; border-radius:10px; border:2px solid #00F0FF;'>"
                        f"<span style='color:#00F0FF; font-weight:bold;'>CẶP ĐỒNG PHẲNG TỐI ƯU #1</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{pair_alpha[0]:02d} — {pair_alpha[1]:02d}</h1>"
                        f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Thuộc không gian: {space_1}</p></div>", unsafe_allow_html=True)

        with col_b:
            st.subheader("⚡ BẬC 2 - BETA (KHÔNG GIAN FANO 2)")
            st.markdown(f"<div style='text-align:center; padding:15px; background:#1A0903; border-radius:10px; border:2px solid #FF5500;'>"
                        f"<span style='color:#FF5500; font-weight:bold;'>CẶP ĐỒNG PHẲNG TỐI ƯU #2</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{pair_beta[0]:02d} — {pair_beta[1]:02d}</h1>"
                        f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Thuộc không gian: {space_2}</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 BẢNG XẾP HẠNG XUNG LỰC 14 SỐ TRONG HÌNH HỌC HỮU HẠN")
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(14)],
            "Con số": [f"Số {top_14[i]:02d}" for i in range(14)],
            "Ánh xạ Không gian": ["Không gian Fano 1" if i < 7 else "Không gian Fano 2" for i in range(14)],
            "Điểm Đại Số Hữu Hạn": [f"{scores[top_14[i]-1]:.6f}" for i in range(14)]
        })
        st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu {min_kies} kỳ dữ liệu để khởi chạy pipeline (Hiện có {res_tuple[1]} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu Keno vào khung trên để khởi chạy mô hình v36.0.")
