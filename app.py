import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS: NCG & Sheaf Consensus Engine", layout="wide")

class SheafNonCommutativeEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=6, max_kies=25):
        """Tối ưu hóa dữ liệu nhập tay: Hỗ trợ từ 6 kỳ trở lên để chia các patch cục bộ."""
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

    def compute_patch_ncg(self, patch_matrix: np.ndarray):
        """Tính toán năng lượng giao hoạn tử NCG cho từng patch cục bộ."""
        T, D = patch_matrix.shape
        if T < 2:
            return np.zeros(D)
        
        A = patch_matrix[:-1]
        B = patch_matrix[1:]
        
        commutator_energy = np.zeros(D)
        for i in range(D):
            col_a = A[:, i]
            col_b = B[:, i]
            tensor = np.outer(col_a, col_b) - np.outer(col_b, col_a)
            commutator_energy[i] = np.linalg.norm(tensor)
            
        return commutator_energy

    def execute_sheaf_consensus(self, matrix: np.ndarray, patch_size=5):
        """
        Lý thuyết Bó (Sheaf Theory): Cắt dữ liệu thành các patch cục bộ, 
        tính NCG trên từng patch và 'dán' (glue) lại thành chân lý toàn cục.
        """
        T, D = matrix.shape
        
        # Tạo các patch chồng lấp (overlapping local patches)
        patches = []
        for start in range(T - patch_size + 1):
            patch = matrix[start : start + patch_size]
            patches.append(patch)
            
        if not patches:
            patches = [matrix]

        # Thu thập điểm số từ các không gian cục bộ
        global_scores = np.zeros(D)
        consistency_counts = np.zeros(D)
        patch_energies = []

        for patch in patches:
            p_energy = self.compute_patch_ncg(patch)
            patch_energies.append(np.mean(p_energy))
            
            # Xếp hạng cục bộ trong patch
            top_local_indices = np.argsort(p_energy)[-15:] # Top 15 số mạnh nhất cục bộ
            for idx in top_local_indices:
                consistency_counts[idx] += 1.0 # Tăng điểm đồng thuận khi xuất hiện ở nhiều patch
            global_scores += p_energy

        # Áp dụng điều kiện dán Sheaf (Gluing Condition): 
        # Số nào xuất hiện đồng thuận ở nhiều patch cục bộ sẽ được cộng hưởng mạnh
        sheaf_glued_scores = global_scores * (1.0 + consistency_counts / len(patches))

        # Bộ lọc kháng quán tính: Phạt các số xuất hiện liên tục ở cuối chuỗi
        recent_act = matrix[-min(3, T):].sum(axis=0)
        final_scores = np.zeros(D)

        for i in range(D):
            penalty = 0.1 if recent_act[i] >= 2 else 1.0
            final_scores[i] = sheaf_glued_scores[i] * penalty

        ranked_indices = np.argsort(final_scores)[::-1]
        core_10_numbers = [idx + 1 for idx in ranked_indices[:10]]

        # Chỉ số đồng thuận toàn cục (Global Sheaf Consistency Index)
        sheaf_consistency = float(np.mean(consistency_counts))

        return {
            "sheaf_consistency": sheaf_consistency,
            "core_10_numbers": core_10_numbers,
            "scores": final_scores
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌌 MDM-IDS: NCG & SHEAF CONSENSUS ENGINE")
st.caption("Hình học Không Giao Hoạn [A, B] kết hợp Lý Thuyết Bó (Sheaf Theory & Gluing Patches)")

with st.sidebar:
    st.header("⚙️ Cấu Hình Không Gian Bó")
    min_w = st.slider("Số kỳ tối thiểu:", 6, 12, 6)
    patch_w = st.slider("Kích thước Local Patch:", 3, 6, 5)

raw_input = st.text_area(
    "Dán thủ công dữ liệu Keno (Hỗ trợ từ 6 kỳ trở lên):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = SheafNonCommutativeEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=min_w)

    if matrix is not None:
        res = engine.execute_sheaf_consensus(matrix, patch_size=patch_w)

        st.success(f"⚡ Đã dán khớp các tầng cục bộ (Sheaf Gluing) thành công trên {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🌀 Chỉ Số Đồng Thuận Bó", value=f"{res['sheaf_consistency']:.4f}")
        with col2:
            st.metric(label="⚡ Cấu Trúc Toàn Cục", value="SHEAF GLUED")
        with col3:
            st.metric(label="🎯 Trạng Thái Biên", value="CONSENSUS CORE")

        st.markdown("---")
        st.subheader("🎯 TOP 10 CON SỐ TẠI LÕI ĐỒNG THUẬN BÓ")
        
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['core_10_numbers']])
        st.markdown(f"<div style='text-align:center; padding:20px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:24px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 CHI TIẾT NĂNG LƯỢNG ĐỒNG THUẬN TOÀN CỤC")
        df_core = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Con số": [f"Số {res['core_10_numbers'][i]:02d}" for i in range(10)],
            "Năng Lượng Đồng Thuận (Sheaf Score)": [f"{res['scores'][res['core_10_numbers'][i]-1]:.6f}" for i in range(10)]
        })
        st.table(df_core.T)

    else:
        st.warning(f"Cần tối thiểu {min_w} kỳ dữ liệu hợp lệ để chia tách các patch cục bộ (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Nhập tay dữ liệu Keno vào khung trên để khởi chạy cỗ máy NCG kết hợp Sheaf Consensus.")
