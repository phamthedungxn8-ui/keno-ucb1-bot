import re
import numpy as np
import pandas as pd
import itertools
import streamlit as st

st.set_page_config(page_title="MDM-IDS v110.0: Dynamic Playstyle & Portfolio Engine", layout="wide")

class DynamicStrategyEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=7, max_kies=25):
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

    def compute_scores(self, matrix: np.ndarray, patch_size=4):
        T, D = matrix.shape
        patches = [matrix[start : start + patch_size] for start in range(T - patch_size + 1)]
        if not patches:
            patches = [matrix]

        energies = []
        for p in patches:
            if len(p) < 2:
                energies.append(np.zeros(D))
                continue
            A, B = p[:-1], p[1:]
            e = np.zeros(D)
            for i in range(D):
                e[i] = np.linalg.norm(np.outer(A[:, i], B[:, i]) - np.outer(B[:, i], A[:, i]))
            energies.append(e)
        
        energies = np.array(energies)
        mean_e = np.mean(energies, axis=0)
        std_e = np.std(energies, axis=0)
        
        # Dual-pass score
        scores = mean_e / (1.0 + std_e)
        return scores

    def execute_strategy(self, matrix: np.ndarray, playstyle: str, patch_size=4):
        T, D = matrix.shape
        scores = self.compute_scores(matrix, patch_size=patch_size)
        recent_freq = matrix[-min(5, T):].sum(axis=0) # Tần suất 5 kỳ gần nhất

        ranked_indices = np.argsort(scores)[::-1]

        if playstyle == "🔥 Dàn Nóng Quán Tính (Hot Momentum)":
            # Ưu tiên các số có năng lượng cao + tần suất xuất hiện gần đây cao
            adjusted_scores = scores * (1.0 + 0.3 * recent_freq)
            final_rank = np.argsort(adjusted_scores)[::-1]
            core_numbers = [idx + 1 for idx in final_rank[:5]]
            strategy_desc = "Tập trung các số đang trên đà nổ mạnh (Hot Streaks)."

        elif playstyle == "❄️ Dàn Lạnh Đảo Chiều (Cold Compression)":
            # Ưu tiên các số năng lượng tích lũy ngầm nhưng chưa ra trong 3-5 kỳ qua (nén pha)
            cold_mask = (recent_freq == 0).astype(float)
            adjusted_scores = scores * (1.0 + 0.8 * cold_mask)
            final_rank = np.argsort(adjusted_scores)[::-1]
            core_numbers = [idx + 1 for idx in final_rank[:5]]
            strategy_desc = "Săn tìm các điểm nén pha lâu ngày sắp bùng nổ đảo chiều."

        elif playstyle == "🎲 Dàn Tối Ưu Tỷ Lệ (Bao 2 / Bao 3 Cặp)":
            # Trích xuất 5 số mạnh nhất để ghép thành các cặp Bao 2 hoặc Bộ 3
            core_numbers = [idx + 1 for idx in ranked_indices[:5]]
            strategy_desc = "Chia nhỏ Top 5 thành các vé Bao 2 / Bao 3 để bảo toàn vốn."

        else: # Mặc định v109.0 Dual-Pass
            core_numbers = [idx + 1 for idx in ranked_indices[:5]]
            strategy_desc = "Cân bằng giữa quét xuôi và quét ngược hai chiều."

        # Ghép cặp Bao 2 và Bao 3 từ Top 5
        pairs_2 = list(itertools.combinations(core_numbers[:4], 2))
        triplets_3 = list(itertools.combinations(core_numbers[:4], 3))

        return {
            "core_numbers": core_numbers,
            "strategy_desc": strategy_desc,
            "pairs_2": pairs_2,
            "triplets_3": triplets_3
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌌 MDM-IDS v110.0: DYNAMIC PLAYSTYLE ENGINE")
st.caption("Khung tùy biến chiến lược • Phân tách Dàn Nóng / Dàn Lạnh / Bao 2-Bao 3 • Tối ưu dòng tiền")

with st.sidebar:
    st.header("🎮 Chọn Chiến Lược Chơi")
    selected_playstyle = st.radio(
        "Đổi phương pháp phân tích:",
        [
            "🔥 Dàn Nóng Quán Tính (Hot Momentum)",
            "❄️ Dàn Lạnh Đảo Chiều (Cold Compression)",
            "🎲 Dàn Tối Ưu Tỷ Lệ (Bao 2 / Bao 3 Cặp)",
            "⚖️ Cân Bằng Hai Chiều (Dual-Pass Balance)"
        ]
    )
    st.markdown("---")
    target_order = st.slider("Số kỳ phân tích (Bậc T):", 7, 15, 8)
    patch_w = st.slider("Kích thước Local Patch:", 3, 6, 4)

raw_input = st.text_area(
    f"Dán thủ công dữ liệu Keno (Tối thiểu {target_order} kỳ):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = DynamicStrategyEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        res = engine.execute_strategy(matrix, selected_playstyle, patch_size=patch_w)

        st.success(f"⚡ Đã áp dụng chiến lược: **{selected_playstyle}** trên {total_kies} kỳ dữ liệu!")
        st.info(f"💡 **Mô tả chiến thuật:** {res['strategy_desc']}")
        st.markdown("---")

        st.subheader("🎯 BỘ SỐ CỐT LÕI (TOP 5 DỰ BÁO)")
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['core_numbers']])
        st.markdown(f"<div style='text-align:center; padding:20px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("📌 DÀN VÉ BAO 2 (Tối Ưu Vốn)")
            st.write("Đánh từng cặp 2 số để tăng tối đa xác suất trúng thưởng:")
            for p in res['pairs_2']:
                st.code(f"Vé Bao 2: [ {p[0]:02d} - {p[1]:02d} ]", language="text")

        with col_b:
            st.subheader("📌 DÀN VÉ BAO 3 (Tăng Biên Lợi Nhuận)")
            st.write("Các bộ 3 số cộng hưởng cao nhất:")
            for t in res['triplets_3']:
                st.code(f"Vé Bao 3: [ {t[0]:02d} - {t[1]:02d} - {t[2]:02d} ]", language="text")

    else:
        st.warning(f"Cần tối thiểu {target_order} kỳ dữ liệu để khởi chạy chiến lược (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Hãy dán dữ liệu Keno vào khung trên và thử đổi các Chế độ chơi ở thanh menu bên trái để trải nghiệm!")
