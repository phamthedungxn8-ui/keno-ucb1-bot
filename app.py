import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v45.0 Quantum-Entropy Engine", layout="wide")

# ==============================================================================
# LỚP 0: QUANTUM-ENTROPY DATA TENSOR PIPELINE
# ==============================================================================
class QuantumEntropyPipeline:
    """Chuyển đổi dữ liệu chuỗi kỳ quay thành ma trận mật độ thông tin và entropy."""
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def build_entropy_tensor(self, raw_text: str, min_kies=8, max_kies=30):
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
# LỚP 1: DYNAMIC SHANNON-VON NEUMANN ENTROPY ENGINE
# ==============================================================================
class InformationEntropyEngine:
    """Tính toán dòng chảy thông tin (Information Gain) và độ bất định động học."""
    def compute_entropy_flux(self, matrix):
        T, D = matrix.shape
        entropy_scores = np.zeros(D)
        
        # Tính xác suất xuất hiện của từng số trong cửa sổ thời gian
        p1 = np.mean(matrix, axis=0)
        p0 = 1.0 - p1
        
        # Tránh lỗi log(0)
        p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)
        p0 = np.clip(p0, 1e-9, 1.0 - 1e-9)
        
        # Shannon Information Entropy cơ sở cho từng số
        base_entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))

        # Đánh giá dòng chảy thông tin động (Dynamic Information Flow trong 5 kỳ gần nhất)
        recent_window = matrix[-min(5, T):]
        recent_activity = recent_window.sum(axis=0)

        for i in range(D):
            # Số nào có entropy cao nhưng hoạt động ở vùng biên (xuất hiện 1-2 lần gần đây) nhận xung lực mạnh
            activity_factor = 1.5 if recent_activity[i] in [1, 2] else 0.5
            # Phạt nhẹ số vừa ra ở kỳ T-1 để chống kẹt lặp
            penalty = 0.2 if matrix[-1, i] == 1.0 else 1.0
            
            entropy_scores[i] = base_entropy[i] * activity_factor * penalty

        return entropy_scores


# ==============================================================================
# LỚP 2: DUAL-CORE INFORMATION ROUTER (BẬC 2 & BẬC 3)
# ==============================================================================
class InformationRouter:
    """Định tuyến điểm năng lượng thông tin thành các dàn Alpha và Beta tối ưu."""
    def route(self, matrix, scores):
        D = len(scores)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]

        # Phân rã Dual-Core Alpha / Beta từ dòng chảy thông tin
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
# STREAMLIT UI (v45.0)
# ==============================================================================
st.title("🧬 MDM-IDS v45.0: QUANTUM-ENTROPY INFORMATION ENGINE")
st.caption("Lý Thuyết Thông Tin Động • Dòng Chảy Shannon-Von Neumann Entropy • Tối Ưu Bậc 2 & Bậc 3")

with st.sidebar:
    st.header("⚙️ Cấu Hình Thông Tin")
    min_window = st.slider("Cửa sổ kỳ tối thiểu:", 8, 15, 10)
    max_window = st.slider("Cửa sổ kỳ tối đa:", 15, 35, 25)

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (Từ 8 đến 35 kỳ):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=180
)

if raw_input.strip():
    pipeline = QuantumEntropyPipeline()
    matrix, total_kies = pipeline.build_entropy_tensor(raw_input, min_kies=min_window, max_kies=max_window)

    if matrix is not None:
        # Chạy Động cơ Entropy Thông tin
        entropy_engine = InformationEntropyEngine()
        entropy_scores = entropy_engine.compute_entropy_flux(matrix)

        # Chạy Định tuyến Thông tin Dual-Core
        router = InformationRouter()
        res = router.route(matrix, entropy_scores)

        st.success(f"⚡ Đã phân tích thành công dòng chảy Shannon Entropy trên {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔥 MỤC TIÊU BẬC 2 (CHỌN 2 - ĂN 90.000 VNĐ)")
            st.markdown(f"<div style='text-align:center; padding:15px; background:#0A192F; border-radius:10px; border:2px solid #00F0FF;'>"
                        f"<span style='color:#00F0FF; font-weight:bold;'>BẬC 2 - ALPHA (THÔNG TIN TÍCH LŨY)</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{res['bac2_alpha'][0]:02d} — {res['bac2_alpha'][1]:02d}</h1></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; padding:15px; background:#1A0903; border-radius:10px; border:2px solid #FF5500;'>"
                        f"<span style='color:#FF5500; font-weight:bold;'>BẬC 2 - BETA (THÔNG TIN BỘC PHÁT)</span>"
                        f"<h1 style='color:#FFF; margin:10px 0;'>{res['bac2_beta'][0]:02d} — {res['bac2_beta'][1]:02d}</h1></div>", unsafe_allow_html=True)

        with col_b:
            st.subheader("⚡ MỤC TIÊU BẬC 3 (CHỌN 3 - HOÀN VỐN / THƯỞNG)")
            b3_a_str = " - ".join([f"{n:02d}" for n in res["bac3_alpha"]])
            b3_b_str = " - ".join([f"{n:02d}" for n in res["bac3_beta"]])
            st.info(f"**BẬC 3 - ALPHA:**\n### **{b3_a_str}**")
            st.warning(f"**BẬC 3 - BETA:**\n### **{b3_b_str}**")

        st.markdown("---")
        st.subheader("📊 BẢNG XẾP HẠNG ĐỘ BẤT ĐỊNH & THÔNG TIN (TOP 12)")
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(12)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(12)],
            "Cấu trúc Core": ["Alpha Core" if i % 2 == 0 else "Beta Core" for i in range(12)],
            "Điểm Shannon Entropy": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(12)]
        })
        st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu {min_window} kỳ dữ liệu để khởi chạy mô hình Entropy (Hiện nhận diện được {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu Keno vào khung trên để khởi chạy mô hình Thông tin Lượng tử v45.0.")
