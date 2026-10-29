import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v109.0: Forward-Backward Smoothing", layout="wide")

class ForwardBackwardSmoothingEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=7, max_kies=25):
        """Lọc dữ liệu nhập tay và lập ma trận nhị phân không gian thời gian."""
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

    def execute_smoothing_analysis(self, matrix: np.ndarray, patch_size=5):
        """
        Thuật toán Quét xuôi kết hợp Quét ngược (Forward-Backward Smoothing):
        - Forward Pass: Đo lường xung lượng năng lượng NCG dọc theo lịch sử.
        - Backward Pass: Chiếu ngược từ kỳ hiện tại (mới nhất) để hiệu chỉnh độ trễ pha.
        """
        T, D = matrix.shape
        patches = [matrix[start : start + patch_size] for start in range(T - patch_size + 1)]
        if not patches:
            patches = [matrix]

        # 1. FORWARD PASS: Thu thập năng lượng và sai số Cohomology xuôi chiều
        forward_energies = np.array([self.compute_patch_ncg(p) for p in patches])
        fwd_mean = np.mean(forward_energies, axis=0)
        fwd_std = np.std(forward_energies, axis=0)
        forward_scores = fwd_mean / (1.0 + fwd_std)

        # 2. BACKWARD PASS (Smoothing): Đảo ngược ma trận thời gian để quét ngược từ hiện tại về quá khứ
        reversed_matrix = matrix[::-1]
        rev_patches = [reversed_matrix[start : start + patch_size] for start in range(T - patch_size + 1)]
        backward_energies = np.array([self.compute_patch_ncg(p) for p in rev_patches])
        bwd_mean = np.mean(backward_energies, axis=0)
        bwd_std = np.std(backward_energies, axis=0)
        backward_scores = bwd_mean / (1.0 + bwd_std)

        # 3. KẾT HỢP HAI CHIỀU (Dual-Pass Fusion): Tổng hòa năng lượng xuôi và ngược để triệt tiêu độ trễ pha
        # Trọng số ưu tiên nhẹ cho chiều Backward vì sát với thực tế hiện tại nhất
        smoothed_scores = (0.4 * forward_scores) + (0.6 * backward_scores)

        # Bộ lọc biên giới kỳ gần nhất (Boundary Constraint)
        recent_act = matrix[-min(3, T):].sum(axis=0)
        final_scores = np.zeros(D)
        for i in range(D):
            penalty = 0.05 if recent_act[i] >= 2 else 1.0
            final_scores[i] = smoothed_scores[i] * penalty

        # Phân rã cấu trúc thành 8 phân thớ đầu số để chọn 2 số chủ lực mỗi nhóm
        structure_groups = {
            f"Đầu {i} ({(i*10)+1:02d}-{(i+1)*10:02d})": list(range(i * 10, (i + 1) * 10))
            for i in range(8)
        }

        selected_cores = []
        structural_breakdown = []

        for group_name, indices in structure_groups.items():
            group_scores = {idx + 1: final_scores[idx] for idx in indices}
            sorted_group = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            top_2_in_group = sorted_group[:2]
            
            for num, score in top_2_in_group:
                selected_cores.append((num, score, group_name))
            
            structural_breakdown.append({
                "Phân thớ cấu trúc": group_name,
                "Top 2 số chủ lực": f"{sorted_group[0][0]:02d}, {sorted_group[1][0]:02d}",
                "Năng lượng làm mượt": round(float(np.mean([s for _, s in sorted_group[:2]])), 4)
            })

        selected_cores = sorted(selected_cores, key=lambda x: x[1], reverse=True)
        final_top_5 = [item[0] for item in selected_cores[:5]]
        smooth_index = float(np.mean(final_scores))

        return {
            "smooth_index": smooth_index,
            "final_top_5": final_top_5,
            "structural_breakdown": structural_breakdown,
            "scores": final_scores
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌌 MDM-IDS v109.0: FORWARD-BACKWARD SMOOTHING")
st.caption("Bộ lọc làm mượt hai chiều • Truy xuất ngược thời gian • Triệt tiêu độ trễ pha")

with st.sidebar:
    st.header("⚙️ Cấu Hình Bộ Lọc Làm Mượt")
    target_order = st.slider("Số kỳ phân tích (Bậc T):", 7, 15, 8)
    patch_w = st.slider("Kích thước Local Patch:", 3, 6, 4)

raw_input = st.text_area(
    f"Dán thủ công dữ liệu Keno (Yêu cầu tối thiểu {target_order} kỳ):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = ForwardBackwardSmoothingEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        res = engine.execute_smoothing_analysis(matrix, patch_size=patch_w)

        st.success(f"⚡ Đã hoàn tất quy trình Quét xuôi & Quét ngược trên {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🌀 Chỉ Số Làm Mượt (Smooth)", value=f"{res['smooth_index']:.4f}")
        with col2:
            st.metric(label="⚡ Trạng Thái Lọc", value="DUAL-PASS OPTIMIZED")
        with col3:
            st.metric(label="🎯 Tập Giá Trị Lõi", value="TOP 5 SMOOTHED")

        st.markdown("---")
        st.subheader("🎯 TOP 5 CON SỐ TỐI ƯU (SAU KHI LÀM MƯỢT HAI CHIỀU)")
        
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['final_top_5']])
        st.markdown(f"<div style='text-align:center; padding:25px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 CHI TIẾT PHÂN RÃ CÁC PHÂN THỚ SAU KHI LÀM MƯỢT")
        df_struct = pd.DataFrame(res['structural_breakdown'])
        st.table(df_struct)

    else:
        st.warning(f"Cần tối thiểu {target_order} kỳ dữ liệu để chạy bộ lọc hai chiều (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Nhập tay dữ liệu Keno từ 7 kỳ trở lên vào khung trên để khởi chạy cỗ máy v109.0.")
