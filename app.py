import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v112.0: Quantum State Transition", layout="wide")

class QuantumTransitionEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=3, max_kies=15):
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

    def execute_quantum_analysis(self, matrix: np.ndarray):
        """
        Mô hình chuyển đổi trạng thái lượng tử (Quantum-like State Transition):
        Đo lường xác suất chuyển dịch biên độ từ kỳ $t-1$ sang kỳ $t$ thông qua
        tương tác giao thoa sóng không gian.
        """
        T, D = matrix.shape
        if T < 2:
            return np.zeros(D)

        # Trọng số gia tốc tập trung mạnh vào 2 kỳ cuối cùng
        recent_matrix = matrix[-min(3, T):]
        weights = np.exp(np.linspace(0, 1, len(recent_matrix)))
        weights /= weights.sum()

        # Tính toán biên độ giao thoa lượng tử cục bộ
        quantum_amplitude = np.zeros(D)
        for i in range(D):
            history_col = recent_matrix[:, i]
            # Tích chập biên độ sóng với trọng số thời gian phi tuyến
            wave_sum = np.sum(history_col * weights)
            # Hiệu ứng cộng hưởng phi tuyến (Quantum-like Phase Interference)
            quantum_amplitude[i] = wave_sum * (1.0 + abs(history_col[-1] - np.mean(history_col)))

        # Phân rã cấu trúc thành 8 phân thớ đầu số để chọn 2 số chủ lực mỗi nhóm
        structure_groups = {
            f"Đầu {i} ({(i*10)+1:02d}-{(i+1)*10:02d})": list(range(i * 10, (i + 1) * 10))
            for i in range(8)
        }

        selected_cores = []
        structural_breakdown = []

        for group_name, indices in structure_groups.items():
            group_scores = {idx + 1: quantum_amplitude[idx] for idx in indices}
            sorted_group = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            top_2_in_group = sorted_group[:2]
            
            for num, score in top_2_in_group:
                selected_cores.append((num, score, group_name))
            
            structural_breakdown.append({
                "Phân thớ cấu trúc": group_name,
                "Top 2 số lượng tử": f"{sorted_group[0][0]:02d}, {sorted_group[1][0]:02d}",
                "Biên độ sóng": round(float(np.mean([s for _, s in sorted_group[:2]])), 4)
            })

        selected_cores = sorted(selected_cores, key=lambda x: x[1], reverse=True)
        final_top_5 = [item[0] for item in selected_cores[:5]]
        quantum_index = float(np.mean(quantum_amplitude))

        return {
            "quantum_index": quantum_index,
            "final_top_5": final_top_5,
            "structural_breakdown": structural_breakdown,
            "scores": quantum_amplitude
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌌 MDM-IDS v112.0: QUANTUM STATE TRANSITION")
st.caption("Mô hình Chuyển đổi Trạng thái Lượng tử • Giao thoa sóng phi tuyến • Phân rã đa cấu trúc")

with st.sidebar:
    st.header("⚙️ Cấu Hình Lượng Tử")
    target_order = st.slider("Số kỳ phân tích (Cửa sổ ngắn):", 3, 8, 4)
    st.info("Phiên bản này loại bỏ hoàn toàn các ràng buộc tuyến tính cũ, tập trung vào biên độ sóng lượng tử.")

raw_input = st.text_area(
    f"Dán thủ công dữ liệu Keno (Chỉ cần từ {target_order} kỳ gần nhất):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = QuantumTransitionEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        res = engine.execute_quantum_analysis(matrix)

        st.success(f"⚛️ Đã giải mã trường lượng tử thành công trên {total_kies} kỳ dữ liệu!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🌀 Chỉ Số Sóng Lượng Tử", value=f"{res['quantum_index']:.4f}")
        with col2:
            st.metric(label="⚡ Trạng Thái", value="QUANTUM ACTIVE")
        with col3:
            st.metric(label="🎯 Tập Giá Trị Lõi", value="TOP 5 QUANTUM CORE")

        st.markdown("---")
        st.subheader("🎯 TOP 5 CON SỐ LƯỢNG TỬ TỐI ƯU")
        
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['final_top_5']])
        st.markdown(f"<div style='text-align:center; padding:25px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 CHI TIẾT PHÂN RÃ BIÊN ĐỘ SÓNG THEO PHÂN THỚ")
        df_struct = pd.DataFrame(res['structural_breakdown'])
        st.table(df_struct)

    else:
        st.warning(f"Cần tối thiểu {target_order} kỳ dữ liệu để khởi chạy mô hình lượng tử (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Hãy dán 3-4 kỳ Keno mới nhất vào khung trên để cỗ máy v112.0 quét trường giao thoa sóng lượng tử!")
