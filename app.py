import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MDM-IDS v38.0 Enterprise Launcher", 
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MODULE 1: MULTI-LAYER DATA TENSOR PIPELINE
# ==============================================================================
class EnterpriseDataPipeline:
    """Xử lý, chuẩn hóa và trích xuất ma trận tensor đa khung thời gian."""
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def process_input(self, raw_text: str, min_kies=5, max_kies=25):
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

        # Trích xuất xung lực đa tầng (Kalman-like weights + Frequency decay)
        freqs = matrix.sum(axis=0)
        recent_k = matrix[-1]
        
        scores = np.zeros(self.D)
        for i in range(self.D):
            penalty = 0.2 if recent_k[i] == 1.0 else 1.0 # Chống kẹt lặp T-1
            if freqs[i] in [1, 2]:
                scores[i] = 2.5 * penalty / (freqs[i] + 0.5) # Vùng bộc phát
            else:
                scores[i] = 0.5 * penalty / (freqs[i] + 1.0)

        ranked_indices = np.argsort(scores)[::-1]
        top_14 = [idx + 1 for idx in ranked_indices[:14]]

        return matrix, top_14, scores


# ==============================================================================
# MODULE 2: FANO-ALGEBRAIC GEOMETRY MAPPING
# ==============================================================================
class FanoAlgebraicEngine:
    """Áp dụng cấu trúc đồng phẳng Mặt phẳng Fano PG(2,2) trên tập 14 số ưu tú."""
    def __init__(self):
        self.fano_lines = [
            (0, 1, 2), (3, 4, 5), (0, 3, 6), 
            (1, 4, 6), (2, 5, 6), (0, 4, 5), (1, 3, 5)
        ]

    def compute_fano_pairs(self, top_14_numbers, scores):
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
                    val = scores[p[0]-1] + scores[p[1]-1]
                    if val > max_val:
                        max_val = val
                        best_pair = sorted(list(p))
            return best_pair, max_val

        pair_alpha, _ = extract_best_collinear_pair(space_1)
        pair_beta, _ = extract_best_collinear_pair(space_2)

        return pair_alpha, pair_beta, space_1, space_2


# ==============================================================================
# MODULE 3: DUAL-CORE COMBINATORIAL ROUTER (BẬC 2 & BẬC 3)
# ==============================================================================
class DualCoreRouter:
    """Định tuyến kết quả thành các dàn Alpha (Tích lũy) và Beta (Bùng nổ)."""
    def route(self, top_14, scores):
        # Sắp xếp lại toàn bộ 14 số theo điểm số đại số
        ranked_14 = sorted(top_14, key=lambda x: scores[x-1], reverse=True)
        
        alpha_core = [ranked_14[i] for i in range(0, len(ranked_14), 2)]
        beta_core = [ranked_14[i] for i in range(1, len(ranked_14), 2)]

        bac3_alpha = sorted(alpha_core[:3])
        bac3_beta = sorted(beta_core[:3])

        return bac3_alpha, bac3_beta, ranked_14


# ==============================================================================
# STREAMLIT UI LAUNCHER
# ==============================================================================
st.title("⚡ MDM-IDS v38.0: ENTERPRISE LAUNCHER")
st.caption("Kiến Trúc Tích Hợp Đa Mô Hỏa • Bộ Lọc Đa Tầng • Hình Học Hữu Hạn Fano • Dual-Core Alpha/Beta")

with st.sidebar:
    st.header("⚙️ Cấu Hình Hệ Thống")
    min_kies = st.slider("Cửa sổ kỳ tối thiểu:", 5, 15, 5)
    max_kies = st.slider("Cửa sổ kỳ tối đa:", 15, 30, 20)
    st.info("Launcher tự động làm mới và đồng bộ các luồng thuật toán hình học Fano và Dual-Core.")

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (Từ 5 đến 30 kỳ):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=180
)

if raw_input.strip():
    pipeline = EnterpriseDataPipeline()
    res_tuple = pipeline.process_input(raw_input, min_kies=min_kies, max_kies=max_kies)

    if res_tuple[0] is not None:
        matrix, top_14, scores = res_tuple
        
        # Chạy Hình học Fano
        fano_engine = FanoAlgebraicEngine()
        pair_alpha, pair_beta, space_1, space_2 = fano_engine.compute_fano_pairs(top_14, scores)

        # Chạy Dual-Core Router cho Bậc 3
        router = DualCoreRouter()
        bac3_alpha, bac3_beta, ranked_14 = router.route(top_14, scores)

        st.success(f"⚡ Khởi chạy thành công toàn bộ pipeline trên {len(matrix)} kỳ dữ liệu thực tế!")
        st.markdown("---")

        # Giao diện Tabs chuyên nghiệp
        tab_b2, tab_b3, tab_geom = st.tabs([
            "🔥 BẬC 2 (CẶP ĐỒNG PHẲNG FANO)", 
            "⚡ BẬC 3 (DUAL-CORE ALPHA/BETA)", 
            "📐 KHÔNG GIAN HÌNH HỌC 14 SỐ"
        ])

        with tab_b2:
            st.subheader("🎯 CẶP BẬC 2 TỐI ƯU QUA MẶT PHẲNG FANO PG(2,2)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='text-align:center; padding:18px; background:#0A192F; border-radius:10px; border:2px solid #00F0FF;'>"
                            f"<span style='color:#00F0FF; font-weight:bold;'>BẬC 2 - ALPHA (FANO 1)</span>"
                            f"<h1 style='color:#FFF; margin:10px 0;'>{pair_alpha[0]:02d} — {pair_alpha[1]:02d}</h1>"
                            f"<p style='color:#AAA; margin:0;'>Không gian điểm: {space_1}</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='text-align:center; padding:18px; background:#1A0903; border-radius:10px; border:2px solid #FF5500;'>"
                            f"<span style='color:#FF5500; font-weight:bold;'>BẬC 2 - BETA (FANO 2)</span>"
                            f"<h1 style='color:#FFF; margin:10px 0;'>{pair_beta[0]:02d} — {pair_beta[1]:02d}</h1>"
                            f"<p style='color:#AAA; margin:0;'>Không gian điểm: {space_2}</p></div>", unsafe_allow_html=True)

        with tab_b3:
            st.subheader("⚡ BẬC 3 CHUYÊN BIỆT (HOÀN VỐN / THƯỞNG LỚN)")
            c3_1, c3_2 = st.columns(2)
            with c3_1:
                st.info(f"**BẬC 3 - ALPHA:**\n### **{' - '.join([f'{n:02d}' for n in bac3_alpha])}**\n*Động cơ tích lũy pha ổn định.*")
            with c3_2:
                st.warning(f"**BẬC 3 - BETA:**\n### **{' - '.join([f'{n:02d}' for n in bac3_beta])}**\n*Động cơ bộc phát vùng biên.*")

        with tab_geom:
            st.subheader("📊 BẢNG XẾP HẠNG TẬP 14 SỐ ƯU TÚ QUA CÁC LỚP")
            df_top = pd.DataFrame({
                "Thứ hạng": [f"Top {i+1}" for i in range(14)],
                "Con số": [f"Số {top_14[i]:02d}" for i in range(14)],
                "Phân loại Fano": ["Không gian 1" if i < 7 else "Không gian 2" for i in range(14)],
                "Điểm Xung Lực": [f"{scores[top_14[i]-1]:.6f}" for i in range(14)]
            })
            st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu {min_kies} kỳ dữ liệu để khởi chạy launcher (Hiện nhận diện được {res_tuple[1]} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu Keno vào khung trên để khởi chạy MDM-IDS v38.0 Enterprise Launcher.")
