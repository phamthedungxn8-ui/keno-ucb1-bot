import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v50.0 Topological Spin-Glass Engine", layout="wide")

# ==============================================================================
# LỚP 0: SPIN-GLASS DATA TENSOR PIPELINE
# ==============================================================================
class SpinGlassPipeline:
    """Chuyển đổi chuỗi quay số thành ma trận tương tác Spin (Ising-like Matrix)."""
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def build_spin_matrix(self, raw_text: str, min_kies=8, max_kies=30):
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
# LỚP 1: SPIN GLASS & TOPOLOGICAL FRUSTRATION ENGINE
# ==============================================================================
class TopologicalSpinEngine:
    """Tính toán ma trận năng lượng tương tác Spin và độ bất mãn năng lượng (Frustration)."""
    def compute_spin_energy(self, matrix):
        T, D = matrix.shape
        
        # Chuyển đổi trạng thái 0/1 thành Spin -1/+1
        spin_matrix = np.where(matrix == 1.0, 1.0, -1.0)
        
        # Tính ma trận tương tác J (Coupling Matrix giữa các con số qua các kỳ)
        # J_ij đo lường mức độ đồng bộ hoặc nghịch pha giữa số i và số j
        coupling_j = np.dot(spin_matrix.T, spin_matrix) / float(T)
        
        # Tính năng lượng cục bộ (Local Field / Energy) của từng số
        # Số nào có năng lượng tương tác bất ổn định (Frustration cao) sẽ có tiềm năng chuyển pha lớn
        local_field = np.sum(coupling_j, axis=1)
        
        # Chuẩn hóa năng lượng kết hợp với quán tính ngắn hạn (5 kỳ gần nhất)
        recent_activity = matrix[-min(5, T):].sum(axis=0)
        
        spin_scores = np.zeros(D)
        for i in range(D):
            # Cân bằng giữa trường năng lượng tổng thể và nhịp bộc phát cục bộ (xuất hiện 1-2 lần gần đây)
            activity_boost = 1.8 if recent_activity[i] in [1, 2] else 0.4
            # Khử quán tính kỳ T-1 trực tiếp
            penalty = 0.15 if matrix[-1, i] == 1.0 else 1.0
            
            spin_scores[i] = abs(local_field[i]) * activity_boost * penalty

        return spin_scores, coupling_j


# ==============================================================================
# LỚP 2: TOPOLOGICAL DUAL-CORE ROUTER
# ==============================================================================
class TopologicalRouter:
    """Định tuyến năng lượng Spin thành các dàn Alpha và Beta tối ưu."""
    def route(self, matrix, scores):
        D = len(scores)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]

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
            "scores": scores
        }


# ==============================================================================
# STREAMLIT UI (v50.0)
# ==============================================================================
st.title("⚛️ MDM-IDS v50.0: TOPOLOGICAL SPIN-GLASS ENGINE")
st.caption("Vật Lý Thống Kê Thủy Tinh Spin (Ising Coupling) • Năng Lượng Bất Mãn (Frustration) • Tối Ưu Bậc 2 & Bậc 3")

with st.sidebar:
    st.header("⚙️ Cấu Hình Thủy Tinh Spin")
    min_window = st.slider("Cửa sổ kỳ tối thiểu:", 8, 15, 10)
    max_window = st.slider("Cửa sổ kỳ tối đa:", 15, 35, 25)

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (Từ 8 đến 35 kỳ):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=180
)

if raw_input.strip():
    pipeline = SpinGlassPipeline()
    matrix, total_kies = pipeline.build_spin_matrix(raw_input, min_kies=min_window, max_kies=max_window)

    if matrix is not None:
        spin_engine = TopologicalSpinEngine()
        spin_scores, _ = spin_engine.compute_spin_energy(matrix)

        router = TopologicalRouter()
        res = router.route(matrix, spin_scores)

        st.success(f"⚡ Đã ánh xạ thành công mạng lưới Thủy tinh Spin và tương tác Ising trên {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔥 MỤC TIÊU BẬC 2 (CHỌN 2 - ĂN 90.000 VNĐ)")
            st.markdown(f"<div style='text-align:center; padding:15px; background:#0A192F; border-radius:10px; border:2px solid #00F0FF;'>"
                        f"<span style='color:#00F0FF; font-weight:bold;'>BẬC 2 - ALPHA (NĂNG LƯỢNG TÍCH LŨY)</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{res['bac2_alpha'][0]:02d} — {res['bac2_alpha'][1]:02d}</h1></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; padding:15px; background:#1A0903; border-radius:10px; border:2px solid #FF5500;'>"
                        f"<span style='color:#FF5500; font-weight:bold;'>BẬC 2 - BETA (NĂNG LƯỢNG BỘC PHÁT)</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{res['bac2_beta'][0]:02d} — {res['bac2_beta'][1]:02d}</h1></div>", unsafe_allow_html=True)

        with col_b:
            st.subheader("⚡ MỤC TIÊU BẬC 3 (CHỌN 3 - HOÀN VỐN / THƯỞNG)")
            b3_a_str = " - ".join([f"{n:02d}" for n in res["bac3_alpha"]])
            b3_b_str = " - ".join([f"{n:02d}" for n in res["bac3_beta"]])
            st.info(f"**BẬC 3 - ALPHA:**\n### **{b3_a_str}**")
            st.warning(f"**BẬC 3 - BETA:**\n### **{b3_b_str}**")

        st.markdown("---")
        st.subheader("📊 BẢNG XẾP HẠNG NĂNG LƯỢNG SPIN & FRUSTRATION (TOP 12)")
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(12)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(12)],
            "Cấu trúc Core": ["Alpha Core" if i % 2 == 0 else "Beta Core" for i in range(12)],
            "Điểm Năng Lượng Spin": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(12)]
        })
        st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu {min_window} kỳ dữ liệu để khởi chạy mô hình Thủy tinh Spin (Hiện nhận diện được {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu Keno vào khung trên để khởi chạy mô hình v50.0.")
