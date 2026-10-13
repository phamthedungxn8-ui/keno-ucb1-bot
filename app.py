import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v35.0 Fano-Geometry Engine", layout="wide")

# ==============================================================================
# LỚP 0: MULTI-LAYER FILTER (RÚT GỌN 80 SỐ XUỐNG 14 SỐ TIỀM NĂNG)
# ==============================================================================
class MultiLayerFilterPipeline:
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

        # Tính điểm xung lực đa tầng (Tần suất + Đồ thị cộng sinh ngắn hạn)
        freqs = matrix.sum(axis=0)
        recent_k = matrix[-1]
        
        scores = np.zeros(self.D)
        for i in range(self.D):
            # Phạt số vừa ra ở T-1 để chống lặp
            penalty = 0.2 if recent_k[i] == 1.0 else 1.0
            # Ưu tiên số xuất hiện 1-2 lần trong 5 kỳ
            if freqs[i] in [1, 2]:
                scores[i] = 2.0 * penalty / (freqs[i] + 1)
            else:
                scores[i] = 0.5 * penalty / (freqs[i] + 1.0)

        ranked_indices = np.argsort(scores)[::-1]
        # Lấy đúng 14 con số đứng đầu qua bộ lọc đa tầng
        top_14 = [idx + 1 for idx in ranked_indices[:14]]

        return matrix, top_14, scores


# ==============================================================================
# LỚP 1: FANO PLANE & FINITE GEOMETRY MAPPING ENGINE
# ==============================================================================
class FanoGeometryEngine:
    """Ánh xạ 14 số thành 2 cấu trúc Mặt phẳng Fano PG(2,2) và tìm cặp đồng phẳng."""
    def __init__(self):
        # Định nghĩa 7 đường thẳng của Mặt phẳng Fano (dưới dạng các bộ 3 điểm)
        # Các điểm được đánh số từ 0 đến 6 (tương ứng giá trị số % 7 hoặc chỉ số mảng)
        self.fano_lines = [
            (0, 1, 2), (3, 4, 5), (0, 3, 6), 
            (1, 4, 6), (2, 5, 6), (0, 4, 5), (1, 3, 5)
        ]

    def process_fano_routing(self, top_14_numbers, scores):
        # Chia 14 số thành 2 khối Fano (Mỗi khối 7 số)
        block_1 = top_14_numbers[:7]
        block_2 = top_14_numbers[7:]

        def evaluate_fano_block(block):
            # Ánh xạ 7 số của block vào 7 điểm Fano (0 -> 6 qua modulo 7 hoặc vị trí)
            # Tìm cặp số nằm trên cùng một đường thẳng Fano và có tổng điểm xung lực cao nhất
            best_pair = None
            max_line_score = -1.0
            
            # Tạo từ điển ánh xạ giá trị số theo chỉ số Fano 0-6
            node_map = {i: block[i] for i in range(7)}
            
            for line in self.fano_lines:
                # line chứa 3 chỉ số điểm, ví dụ (0, 1, 2)
                n1, n2, n3 = node_map[line[0]], node_map[line[1]], node_map[line[2]]
                
                # Xét 3 cặp Bậc 2 có thể có trên đường thẳng Fano này
                pairs_on_line = [(n1, n2), (n2, n3), (n1, n3)]
                for p in pairs_on_line:
                    p_score = scores[p[0]-1] + scores[p[1]-1]
                    if p_score > max_line_score:
                        max_line_score = p_score
                        best_pair = sorted(list(p))
                        
            return best_pair, max_line_score

        # Trích xuất cặp Alpha từ Block 1 (Mặt phẳng Fano thứ nhất)
        pair_alpha, score_alpha = evaluate_fano_block(block_1)
        
        # Trích xuất cặp Beta từ Block 2 (Mặt phẳng Fano thứ hai)
        pair_beta, score_beta = evaluate_fano_block(block_2)

        return pair_alpha, pair_beta, block_1, block_2


# ==============================================================================
# UI STREAMLIT (v35.0)
# ==============================================================================
st.title("📐 MDM-IDS v35.0: FANO-GEOMETRY BẬC 2 ENGINE")
st.caption("Bộ Lọc Đa Tầng 5 Kỳ • Ánh Xạ Hình Học Hữu Hạn PG(2,2) • Mặt Phẳng Fano & Cặp Đồng Phẳng")

with st.sidebar:
    st.header("⚙️ Cấu Hình Không Gian")
    min_kies = st.slider("Số kỳ đầu vào tối thiểu:", 5, 10, 5)

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (Từ 5 kỳ trở lên):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=160
)

if raw_input.strip():
    pipeline = MultiLayerFilterPipeline()
    res_tuple = pipeline.extract_top_pool(raw_input, min_kies=min_kies)

    if res_tuple[0] is not None:
        matrix, top_14, scores = res_tuple
        
        # Ánh xạ qua Hình học Fano
        fano_engine = FanoGeometryEngine()
        pair_alpha, pair_beta, block_1, block_2 = fano_engine.process_fano_routing(top_14, scores)

        st.success(f"⚡ Đã lọc thô còn 14 số qua đa tầng và chiếu xạ thành công lên 2 Mặt phẳng Fano!")
        st.markdown("---")

        st.info(f"🧩 **Tập 14 số ưu tú từ bộ lọc đa tầng:** `{top_14}`")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔥 BẬC 2 - ALPHA (MẶT PHẲNG FANO 1)")
            st.markdown(f"<div style='text-align:center; padding:15px; background:#0A192F; border-radius:10px; border:2px solid #00F0FF;'>"
                        f"<span style='color:#00F0FF; font-weight:bold;'>CẶP ĐỒNG PHẲNG FANO #1</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{pair_alpha[0]:02d} — {pair_alpha[1]:02d}</h1>"
                        f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Thuộc khối điểm: {block_1}</p></div>", unsafe_allow_html=True)

        with col_b:
            st.subheader("⚡ BẬC 2 - BETA (MẶT PHẲNG FANO 2)")
            st.markdown(f"<div style='text-align:center; padding:15px; background:#1A0903; border-radius:10px; border:2px solid #FF5500;'>"
                        f"<span style='color:#FF5500; font-weight:bold;'>CẶP ĐỒNG PHẲNG FANO #2</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{pair_beta[0]:02d} — {pair_beta[1]:02d}</h1>"
                        f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Thuộc khối điểm: {block_2}</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 CHI TIẾT ĐIỂM XUNG LỰC 14 SỐ QUA BỘ LỌC ĐA TẦNG")
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(14)],
            "Con số": [f"Số {top_14[i]:02d}" for i in range(14)],
            "Phân bổ Fano": ["Mặt phẳng Fano 1" if i < 7 else "Mặt phẳng Fano 2" for i in range(14)],
            "Điểm Lọc Đa Tầng": [f"{scores[top_14[i]-1]:.6f}" for i in range(14)]
        })
        st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu {min_kies} kỳ dữ liệu để khởi chạy pipeline (Hiện có {res_tuple[1]} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu Keno vào khung trên để khởi chạy mô hình Fano-Geometry v35.0.")
