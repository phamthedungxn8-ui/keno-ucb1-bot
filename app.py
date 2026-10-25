import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v40.0 Phase Space Engine", layout="wide")

# ==============================================================================
# LỚP 0: PHASE SPACE EMBEDDING PIPELINE
# ==============================================================================
class PhaseSpacePipeline:
    """Xây dựng không gian pha đa chiều từ chuỗi thời gian nhị phân của 80 số."""
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def build_phase_space(self, raw_text: str, min_kies=8, max_kies=30, embedding_dim=4):
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

        return matrix, kies_to_use, embedding_dim


# ==============================================================================
# LỚP 1: PHASE TRAJECTORY & VELOCITY ENGINE (TÁI CẤU TRÚC KHÔNG GIAN PHA)
# ==============================================================================
class PhaseSpaceVelocityEngine:
    """Tính toán vận tốc quỹ đạo và gia tốc chuyển pha trong không gian nhúng."""
    def compute_phase_scores(self, matrix, embedding_dim):
        T, D = matrix.shape
        if T < embedding_dim:
            return np.zeros(D)

        phase_scores = np.zeros(D)
        
        for i in range(D):
            # Trích xuất chuỗi thời gian của con số i (0 hoặc 1 qua T kỳ)
            ts = matrix[:, i]
            
            # Xây dựng các vector trạng thái nhúng (Time-Delay Embedding vectors)
            vectors = []
            for t in range(T - embedding_dim + 1):
                vec = ts[t : t + embedding_dim]
                vectors.append(vec)
            vectors = np.array(vectors) # Kịch bản không gian m-chiều
            
            if len(vectors) < 2:
                phase_scores[i] = ts.sum()
                continue

            # Tính khoảng cách dịch chuyển quỹ đạo (Phase Velocity / Euclidean Norm of changes)
            velocity = np.linalg.norm(vectors[-1] - vectors[-2])
            
            # Trọng số gia tốc: Nếu số đó vừa chuyển từ 0 sang 1 ở kỳ cuối (T-1), tạo xung lực lớn
            recent_momentum = ts[-1] * 2.0 - ts[-2] * 0.5
            
            # Điểm không gian pha tổng hợp = Vận tốc quỹ đạo * Mật độ xuất hiện gần - Hệ số phạt lặp
            frequency_weight = np.mean(ts[-5:]) # Tần suất trong 5 kỳ gần
            
            phase_scores[i] = (velocity + recent_momentum) * (1.0 - frequency_weight * 0.5)

        return phase_scores


# ==============================================================================
# LỚP 2: DUAL-CORE PHASE ROUTER (ĐỊNH TUYẾN BẬC 2 & BẬC 3)
# ==============================================================================
class PhaseSpaceRouter:
    """Ánh xạ điểm không gian pha thành các dàn Alpha và Beta."""
    def route(self, matrix, scores):
        D = len(scores)
        final_scores = scores.copy()

        # Khử quán tính kỳ T-1 (Chống kẹt lặp số cứng nhắc)
        for i in range(D):
            if matrix[-1, i] == 1.0:
                final_scores[i] *= 0.25

        ranked_indices = np.argsort(final_scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]

        # Phân rã Dual-Core Alpha / Beta
        alpha_core = [ranked_numbers[i] for i in range(0, 14, 2)]
        beta_core = [ranked_numbers[i] for i in range(1, 14, 2)]

        bac2_alpha = sorted(alpha_core[:2])
        bac2_beta = sorted(beta_core[:2])

        bac3_alpha = sorted(alpha_core[:3])
        bac3_beta = sorted(beta_core[:3])

        return {
            "bac2_alpha": bac2_alpha,
            "bac2_beta": bac2_beta,
            "bac3_alpha": bac3_alpha,
            "bac3_beta": bac3_beta,
            "ranked_all": ranked_numbers,
            "scores": final_scores
        }


# ==============================================================================
# STREAMLIT UI (v40.0)
# ==============================================================================
st.title("🌌 MDM-IDS v40.0: PHASE SPACE EMBEDDING ENGINE")
st.caption("Tái Cấu Trúc Không Gian Pha (Takens' Embedding) • Vận Tốc Quỹ Đạo Động Lực Học • Tối Ưu Bậc 2 & Bậc 3")

with st.sidebar:
    st.header("⚙️ Cấu Hình Không Gian Pha")
    min_window = st.slider("Cửa sổ kỳ tối thiểu:", 8, 15, 10)
    max_window = st.slider("Cửa sổ kỳ tối đa:", 15, 35, 25)
    embedding_dim = st.slider("Số chiều nhúng (Embedding Dim $m$):", 2, 6, 4)

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (Từ 8 đến 35 kỳ):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=180
)

if raw_input.strip():
    pipeline = PhaseSpacePipeline()
    res_tuple = pipeline.build_phase_space(raw_input, min_kies=min_window, max_kies=max_window, embedding_dim=embedding_dim)

    if res_tuple[0] is not None:
        matrix, total_kies, dim = res_tuple
        
        # Chạy Động cơ Vận tốc Không gian pha
        velocity_engine = PhaseSpaceVelocityEngine()
        phase_scores = velocity_engine.compute_phase_scores(matrix, dim)

        # Chạy Định tuyến Dual-Core
        router = PhaseSpaceRouter()
        res = router.route(matrix, phase_scores)

        st.success(f"⚡ Đã nhúng thành công không gian pha $m={dim}$ chiều trên {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔥 MỤC TIÊU BẬC 2 (CHỌN 2 - ĂN 90.000 VNĐ)")
            st.markdown(f"<div style='text-align:center; padding:15px; background:#0A192F; border-radius:10px; border:2px solid #00F0FF;'>"
                        f"<span style='color:#00F0FF; font-weight:bold;'>BẬC 2 - ALPHA (QUỸ ĐẠO TÍCH LŨY)</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{res['bac2_alpha'][0]:02d} — {res['bac2_alpha'][1]:02d}</h1></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; padding:15px; background:#1A0903; border-radius:10px; border:2px solid #FF5500;'>"
                        f"<span style='color:#FF5500; font-weight:bold;'>BẬC 2 - BETA (QUỸ ĐẠO BỘC PHÁT)</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{res['bac2_beta'][0]:02d} — {res['bac2_beta'][1]:02d}</h1></div>", unsafe_allow_html=True)

        with col_b:
            st.subheader("⚡ MỤC TIÊU BẬC 3 (CHỌN 3 - HOÀN VỐN / THƯỞNG)")
            b3_a_str = " - ".join([f"{n:02d}" for n in res["bac3_alpha"]])
            b3_b_str = " - ".join([f"{n:02d}" for n in res["bac3_beta"]])
            st.info(f"**BẬC 3 - ALPHA:**\n### **{b3_a_str}**")
            st.warning(f"**BẬC 3 - BETA:**\n### **{b3_b_str}**")

        st.markdown("---")
        st.subheader("📊 BẢNG XẾP HẠNG VẬN TỐC QUỸ ĐẠO KHÔNG GIAN PHA (TOP 12)")
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(12)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(12)],
            "Cấu trúc Core": ["Alpha Core" if i % 2 == 0 else "Beta Core" for i in range(12)],
            "Điểm Vận Tốc Pha": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(12)]
        })
        st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu {min_window} kỳ dữ liệu để khởi chạy không gian pha (Hiện nhận diện được {res_tuple[1]} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu Keno vào khung trên để khởi chạy mô hình Không gian pha v40.0.")
