import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v108.0: Evolutionary Auto-Optimizer", layout="wide")

class EvolutionaryStructuralEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=7, max_kies=25):
        """Lọc dữ liệu nhập tay và chuyển đổi thành ma trận nhị phân không gian thời gian."""
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

    def run_evolutionary_optimization(self, matrix: np.ndarray):
        """
        Thuật toán tiến hóa thử-sai tự động: Quét qua các kích thước patch (từ 3 đến 6) 
        và trọng số phạt quán tính để tìm ra không gian cộng hưởng bền vững nhất.
        """
        T, D = matrix.shape
        best_score = -float('inf')
        best_results = None
        best_params = {"patch": 4, "penalty_factor": 0.05}

        # Vòng lặp tiến hóa thử nghiệm các tham số cấu trúc
        for p_size in range(3, min(7, T)):
            patches = [matrix[start : start + p_size] for start in range(T - p_size + 1)]
            if not patches:
                continue

            patch_energies = np.array([self.compute_patch_ncg(p) for p in patches])
            global_mean = np.mean(patch_energies, axis=0)
            cohomology_error = np.std(patch_energies, axis=0)
            
            # Thử nghiệm các hệ số phạt quán tính khác nhau trong quá trình tiến hóa
            for pen in [0.01, 0.05, 0.1, 0.2]:
                recent_act = matrix[-min(3, T):].sum(axis=0)
                final_scores = np.zeros(D)
                
                phase_scores = global_mean / (1.0 + cohomology_error)
                for i in range(D):
                    penalty = pen if recent_act[i] >= 2 else 1.0
                    final_scores[i] = phase_scores[i] * penalty

                # Đánh giá độ thích nghi (Fitness Score) của cấu trúc
                fitness = float(np.mean(np.sort(final_scores)[-5:]))
                if fitness > best_score:
                    best_score = fitness
                    best_params = {"patch": p_size, "penalty_factor": pen}
                    best_results = final_scores

        # Phân rã 8 phân thớ cấu trúc đầu số dựa trên thông số tối ưu nhất
        structure_groups = {
            f"Đầu {i} ({(i*10)+1:02d}-{(i+1)*10:02d})": list(range(i * 10, (i + 1) * 10))
            for i in range(8)
        }

        selected_cores = []
        structural_breakdown = []

        for group_name, indices in structure_groups.items():
            group_scores = {idx + 1: best_results[idx] for idx in indices}
            sorted_group = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            top_2_in_group = sorted_group[:2]
            
            for num, score in top_2_in_group:
                selected_cores.append((num, score, group_name))
            
            structural_breakdown.append({
                "Phân thớ cấu trúc": group_name,
                "Top 2 số chủ lực": f"{sorted_group[0][0]:02d}, {sorted_group[1][0]:02d}",
                "Năng lượng liên thông": round(float(np.mean([s for _, s in sorted_group[:2]])), 4)
            })

        selected_cores = sorted(selected_cores, key=lambda x: x[1], reverse=True)
        final_top_5 = [item[0] for item in selected_cores[:5]]

        return {
            "best_params": best_params,
            "fitness_score": best_score,
            "final_top_5": final_top_5,
            "structural_breakdown": structural_breakdown,
            "scores": best_results
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌌 MDM-IDS v108.0: EVOLUTIONARY AUTO-OPTIMIZER")
st.caption("Thuật toán tiến hóa tự động thử-sai • Tối ưu hóa đa cấu trúc bậc cao")

with st.sidebar:
    st.header("⚙️ Cấu Hình Tiến Hoá AI")
    target_order = st.slider("Số kỳ phân tích (Bậc T):", 7, 15, 8)
    st.info("Hệ thống sẽ tự động quét và tối ưu hóa trọng số pha mà không cần điều chỉnh thủ công.")

raw_input = st.text_area(
    f"Dán thủ công dữ liệu Keno (Yêu cầu tối thiểu {target_order} kỳ):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = EvolutionaryStructuralEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        # Kích hoạt quá trình tiến hóa tự động
        res = engine.run_evolutionary_optimization(matrix)

        st.success(f"🧬 Quá trình tiến hóa AI đã hoàn tất trên {total_kies} kỳ! (Patch tối ưu: {res['best_params']['patch']}, Phạt quán tính: {res['best_params']['penalty_factor']})")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🌀 Chỉ Số Thích Nghi (Fitness)", value=f"{res['fitness_score']:.4f}")
        with col2:
            st.metric(label="⚡ Trạng Thái Tiến Hóa", value="OPTIMIZED CORES")
        with col3:
            st.metric(label="🎯 Tập Giá Trị Lõi", value="TOP 5 EVOLVED")

        st.markdown("---")
        st.subheader("🎯 TOP 5 CON SỐ TỐI ƯU SAU KHI TIẾN HÓA")
        
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['final_top_5']])
        st.markdown(f"<div style='text-align:center; padding:25px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 CHI TIẾT PHÂN RÃ CÁC PHÂN THỚ SAU KHI TỰ ĐIỀU CHỈNH")
        df_struct = pd.DataFrame(res['structural_breakdown'])
        st.table(df_struct)

    else:
        st.warning(f"Cần tối thiểu {target_order} kỳ dữ liệu để chạy thuật toán tiến hóa (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Nhập tay dữ liệu Keno từ 7 kỳ trở lên vào khung trên để cỗ máy tiến hóa tự động vận hành.")
