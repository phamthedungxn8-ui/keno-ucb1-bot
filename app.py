import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v106.0: Multi-Structural Cohomology", layout="wide")

class MultiStructuralCohomologyEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=6, max_kies=25):
        """Lọc sạch dữ liệu nhập tay và lập ma trận nhị phân theo thời gian."""
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
        """Tính toán năng lượng giao hoạn tử NCG [A, B] trên từng patch."""
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

    def execute_multi_structural_analysis(self, matrix: np.ndarray, patch_size=5):
        """
        Phân rã đa cấu trúc: Chia không gian thành các phân thớ đầu số (Decade Fibers),
        tính toán Sheaf Cohomology và trích xuất điểm sáng từ mỗi cấu trúc.
        """
        T, D = matrix.shape
        patches = []
        for start in range(T - patch_size + 1):
            patch = matrix[start : start + patch_size]
            patches.append(patch)
            
        if not patches:
            patches = [matrix]

        patch_energies = []
        for patch in patches:
            patch_energies.append(self.compute_patch_ncg(patch))
        patch_energies = np.array(patch_energies)

        global_mean = np.mean(patch_energies, axis=0)
        cohomology_error = np.std(patch_energies, axis=0)
        phase_scores = global_mean / (1.0 + cohomology_error)

        # Định nghĩa các phân thớ cấu trúc (Đầu số: 01-10, 11-20, ..., 71-80)
        structure_groups = {
            f"Đầu {i} ({(i*10)+1:02d}-{(i+1)*10:02d})": list(range(i * 10, (i + 1) * 10))
            for i in range(8)
        }

        selected_cores = []
        structural_breakdown = []

        for group_name, indices in structure_groups.items():
            # Trích xuất điểm số của các số trong phân thớ hiện tại
            group_scores = {idx + 1: phase_scores[idx] for idx in indices}
            # Sắp xếp lấy Top 2 số ưu tú nhất từ mỗi cấu trúc đầu số
            sorted_group = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            top_2_in_group = sorted_group[:2]
            
            for num, score in top_2_in_group:
                selected_cores.append((num, score, group_name))
            
            structural_breakdown.append({
                "Cấu trúc": group_name,
                "Top số đề xuất": f"{sorted_group[0][0]:02d}, {sorted_group[1][0]:02d}",
                "Năng lượng trung bình": float(np.mean([s for _, s in sorted_group[:2]]))
            })

        # Sắp xếp lại toàn bộ tập hợp đã trích xuất từ các cấu trúc theo năng lượng liên thông pha
        selected_cores = sorted(selected_cores, key=lambda x: x[1], reverse=True)
        
        # Lọc lấy tập siêu hẹp Top 5 đại diện cho toàn bộ các cấu trúc
        final_top_5 = [item[0] for item in selected_cores[:5]]
        entanglement_index = float(np.mean(phase_scores))

        return {
            "entanglement_index": entanglement_index,
            "final_top_5": final_top_5,
            "structural_breakdown": structural_breakdown,
            "scores": phase_scores
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌌 MDM-IDS v106.0: MULTI-STRUCTURAL COHOMOLOGY")
st.caption("Phân rã đa cấu trúc đầu số • Đối ngẫu Sheaf Cohomology • Tối ưu hóa bậc cao")

with st.sidebar:
    st.header("⚙️ Cấu Hình Không Gian Bậc Cao")
    min_w = st.slider("Số kỳ tối thiểu (Bậc):", 6, 15, 7)
    patch_w = st.slider("Kích thước Local Patch:", 3, 8, 5)

raw_input = st.text_area(
    "Dán thủ công dữ liệu Keno (Hỗ trợ từ 6-15 kỳ):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = MultiStructuralCohomologyEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=min_w)

    if matrix is not None:
        res = engine.execute_multi_structural_analysis(matrix, patch_size=patch_w)

        st.success(f"⚡ Đã bóc tách cấu trúc và liên thông pha thành công trên {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🌀 Chỉ Số Liên Thông Pha", value=f"{res['entanglement_index']:.4f}")
        with col2:
            st.metric(label="⚡ Phân Thớ Cấu Trúc", value="8 GROUPS RESOLVED")
        with col3:
            st.metric(label="🎯 Tập Giá Trị Lõi", value="TOP 5 MULTI-CORE")

        st.markdown("---")
        st.subheader("🎯 TOP 5 CON SỐ TỐI ƯU (TỔNG HỢP ĐA CẤU TRÚC)")
        
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['final_top_5']])
        st.markdown(f"<div style='text-align:center; padding:25px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 CHI TIẾT PHÂN RÃ THEO PHÂN THỚ ĐẦU SỐ")
        df_struct = pd.DataFrame(res['structural_breakdown'])
        st.table(df_struct)

    else:
        st.warning(f"Cần tối thiểu {min_w} kỳ dữ liệu hợp lệ để chạy phân rã đa cấu trúc (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Nhập tay dữ liệu Keno vào khung trên để khởi chạy cỗ máy v106.0.")
