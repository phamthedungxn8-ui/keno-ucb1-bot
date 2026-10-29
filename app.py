import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v111.0: Adaptive Short-Window Engine", layout="wide")

class AdaptiveShortWindowEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=3, max_kies=25):
        """Hỗ trợ cửa sổ ngắn linh hoạt: Chỉ cần từ 3-4 kỳ trở lên là có thể kích hoạt."""
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

    def execute_adaptive_decay_analysis(self, matrix: np.ndarray, decay_rate=0.6, patch_size=3):
        """
        Thuật toán Trọng số Suy giảm theo Hàm mũ & Cửa sổ Thích ứng Ngắn:
        - gán trọng số cực cao cho các kỳ sát hiện tại.
        - Triệt tiêu sức ì của các kỳ quá xa trong quá khứ.
        """
        T, D = matrix.shape
        
        # Tạo mảng trọng số suy giảm hàm mũ (Exponential Decay Weights)
        # Kỳ mới nhất (cuối cùng) có trọng số lớn nhất
        time_weights = np.array([decay_rate ** (T - 1 - t) for t in range(T)])
        time_weights /= np.sum(time_weights) # Chuẩn hóa về tổng = 1

        # Tính toán xung lượng năng lượng tức thời có gán trọng số thời gian
        weighted_matrix = matrix * time_weights[:, np.newaxis]
        
        # Đo lường biến động tức thời trong 2-3 kỳ gần nhất
        recent_burst = matrix[-min(2, T):].sum(axis=0) # Tần suất 2 kỳ gần nhất
        global_energy = weighted_matrix.sum(axis=0)

        # Tính điểm số năng lượng tức thời (Instantaneous Phase Score)
        # Bắt dính các số đang chớm bùng nổ hoặc có xung lượng cực ngắn
        instant_scores = global_energy * (1.0 + 0.5 * recent_burst)

        # Phân rã theo 8 phân thớ đầu số
        structure_groups = {
            f"Đầu {i} ({(i*10)+1:02d}-{(i+1)*10:02d})": list(range(i * 10, (i + 1) * 10))
            for i in range(8)
        }

        selected_cores = []
        structural_breakdown = []

        for group_name, indices in structure_groups.items():
            group_scores = {idx + 1: instant_scores[idx] for idx in indices}
            sorted_group = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            top_2_in_group = sorted_group[:2]
            
            for num, score in top_2_in_group:
                selected_cores.append((num, score, group_name))
            
            structural_breakdown.append({
                "Phân thớ cấu trúc": group_name,
                "Top 2 số tức thời": f"{sorted_group[0][0]:02d}, {sorted_group[1][0]:02d}",
                "Xung lượng ngắn hạn": round(float(np.mean([s for _, s in sorted_group[:2]])), 4)
            })

        selected_cores = sorted(selected_cores, key=lambda x: x[1], reverse=True)
        final_top_5 = [item[0] for item in selected_cores[:5]]
        adaptive_index = float(np.mean(instant_scores))

        return {
            "adaptive_index": adaptive_index,
            "final_top_5": final_top_5,
            "structural_breakdown": structural_breakdown,
            "scores": instant_scores
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("⚡ MDM-IDS v111.0: ADAPTIVE SHORT-WINDOW CORE")
st.caption("Cửa sổ thích ứng ngắn (3-4 kỳ) • Trọng số suy giảm hàm mũ (Exponential Decay) • Triệt tiêu trễ pha")

with st.sidebar:
    st.header("⚙️ Cấu Hình Độ Nhạy Tức Thời")
    target_order = st.slider("Số kỳ phân tích (Bậc T ngắn):", 3, 7, 4)
    decay_val = st.slider("Hệ số suy giảm (Decay Rate):", 0.3, 0.9, 0.6, step=0.05)
    st.info("Hệ số Decay càng nhỏ, mô hình càng tập trung tuyệt đối vào 1-2 kỳ sát hiện tại.")

raw_input = st.text_area(
    f"Dán thủ công dữ liệu Keno (Chỉ cần từ {target_order} kỳ):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = AdaptiveShortWindowEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        res = engine.execute_adaptive_decay_analysis(matrix, decay_rate=decay_val, patch_size=target_order)

        st.success(f"🚀 Đã xử lý với Cửa sổ thích ứng {total_kies} kỳ (Trọng số sát hiện tại: {100*(1-decay_val):.1f}%)!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🌀 Chỉ Số Tức Thời", value=f"{res['adaptive_index']:.4f}")
        with col2:
            st.metric(label="⚡ Chế Độ Cửa Sổ", value=f"SHORT-WINDOW ({total_kies} KỲ)")
        with col3:
            st.metric(label="🎯 Tập Giá Trị Lõi", value="TOP 5 INSTANT")

        st.markdown("---")
        st.subheader("🎯 TOP 5 CON SỐ T TỨC THỜI (LOẠI BỎ SỨC Ì LỊCH SỬ)")
        
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['final_top_5']])
        st.markdown(f"<div style='text-align:center; padding:25px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 CHI TIẾT PHÂN RÃ THEO XUNG LƯỢNG NGẮN HẠN")
        df_struct = pd.DataFrame(res['structural_breakdown'])
        st.table(df_struct)

    else:
        st.warning(f"Cần tối thiểu {target_order} kỳ dữ liệu để khởi chạy cửa sổ ngắn (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Nhập tay từ 3 đến 5 kỳ dữ liệu Keno mới nhất để trải nghiệm độ nhạy tức thời của v111.0!")
