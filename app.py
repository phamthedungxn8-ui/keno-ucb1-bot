import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS NCG: Non-Commutative Core", layout="wide")

class NonCommutativeEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=5, max_kies=20):
        """
        Tối ưu hóa cho dữ liệu ít kỳ: Cho phép xử lý từ 5 kỳ nhập tay trở lên.
        Lọc sạch cú pháp, bóc tách chính xác ma trận nhị phân 80 số/kỳ.
        """
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

    def compute_non_commutative_algebra(self, matrix: np.ndarray):
        """
        Tính toán giao hoạn tử [A, B] = A*B - B*A giữa các kỳ quay liên tiếp
        để trích xuất năng lượng tương tác không giao hoạn trong không gian toán tử.
        """
        T, D = matrix.shape
        
        A = matrix[:-1] # Trạng thái kỳ t
        B = matrix[1:]  # Trạng thái kỳ t+1

        commutator_energy = np.zeros(D)
        for i in range(D):
            col_a = A[:, i]
            col_b = B[:, i]
            # Đo lường độ bất đối xứng thứ tự qua tensor tương tác
            non_comm_tensor = np.outer(col_a, col_b) - np.outer(col_b, col_a)
            commutator_energy[i] = np.linalg.norm(non_comm_tensor)

        # Bộ lọc kháng quán tính: Phạt các số xuất hiện liên tục ở cuối chuỗi
        recent_act = matrix[-min(3, T):].sum(axis=0)
        final_scores = np.zeros(D)

        for i in range(D):
            penalty = 0.1 if recent_act[i] >= 2 else 1.0
            final_scores[i] = commutator_energy[i] * penalty

        ranked_indices = np.argsort(final_scores)[::-1]
        core_10_numbers = [idx + 1 for idx in ranked_indices[:10]]

        # Đo lường biên độ bất đối xứng toàn cục (Non-Commutative Curvature Index)
        ncg_curvature = float(np.mean(commutator_energy))

        return {
            "ncg_curvature": ncg_curvature,
            "core_10_numbers": core_10_numbers,
            "scores": final_scores
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌌 MDM-IDS: NON-COMMUTATIVE GEOMETRY ENGINE")
st.caption("Phân rã đại số toán tử không giao hoạn [A, B] • Tối ưu hóa cho chuỗi dữ liệu ngắn")

with st.sidebar:
    st.header("⚙️ Cấu Hình Kỳ Nhập Tay")
    min_w = st.slider("Số kỳ tối thiểu:", 4, 10, 5)
    max_w = st.slider("Số kỳ tối đa phân tích:", 10, 25, 15)

raw_input = st.text_area(
    "Dán thủ công dữ liệu Keno (Hỗ trợ từ 5 kỳ trở lên):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = NonCommutativeEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=min_w, max_kies=max_w)

    if matrix is not None:
        res = engine.compute_non_commutative_algebra(matrix)

        st.success(f"⚡ Đã ánh xạ thành công đại số toán tử trên {total_kies} kỳ dữ liệu nhập tay!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🌀 Độ Cong Không Giao Hoạn", value=f"{res['ncg_curvature']:.4f}")
        with col2:
            st.metric(label="⚡ Cấu Trúc Toán Tử", value="ACTIVE [A, B] != 0")
        with col3:
            st.metric(label="🎯 Trạng Thái Biên", value="NON-COMMUTATIVE CORE")

        st.markdown("---")
        st.subheader("🎯 TOP 10 CON SỐ TẠI LÕI ĐẠI SỐ KHÔNG GIAO HOÀN")
        
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['core_10_numbers']])
        st.markdown(f"<div style='text-align:center; padding:20px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:24px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 CHI TIẾT NĂNG LƯỢNG TƯƠNG TÁC TOÁN TỬ")
        df_core = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Con số": [f"Số {res['core_10_numbers'][i]:02d}" for i in range(10)],
            "Năng Lượng Giao Hoạn (NCG Score)": [f"{res['scores'][res['core_10_numbers'][i]-1]:.6f}" for i in range(10)]
        })
        st.table(df_core.T)

    else:
        st.warning(f"Cần tối thiểu {min_w} kỳ dữ liệu hợp lệ để xử lý (Hệ thống ghi nhận được {total_kies} kỳ). Hãy kiểm tra lại định dạng dữ liệu dán vào.")
else:
    st.info("Nhập tay hoặc dán dữ liệu Keno ngắn vào khung trên để khởi chạy lõi NCG.")
