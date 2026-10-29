import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-Swarm v1.0: Multi-Agent Research Core", layout="wide")

class MultiAgentSwarmEngine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def parse_data(self, raw_text: str, min_kies=3, max_kies=20):
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

    def agent_topologist(self, matrix):
        """Agent 1: Phân tích hình học tô-pô và năng lượng biến đổi không gian."""
        T, D = matrix.shape
        if T < 2:
            return np.zeros(D)
        diffs = np.abs(np.diff(matrix, axis=0))
        topo_scores = diffs.mean(axis=0) * (1.0 + matrix[-1])
        return topo_scores

    def agent_phase_shifter(self, matrix):
        """Agent 2: Đo lường độ trễ pha và xung lượng suy giảm hàm mũ."""
        T, D = matrix.shape
        weights = np.exp(np.linspace(0, 2, T))
        weights /= weights.sum()
        weighted = matrix * weights[:, np.newaxis]
        return weighted.sum(axis=0)

    def agent_cluster_miner(self, matrix):
        """Agent 3: Khai thác ma trận tương quan cặp và cụm bọc lót."""
        T, D = matrix.shape
        co_matrix = np.dot(matrix.T, matrix)
        np.fill_diagonal(co_matrix, 0)
        return co_matrix.sum(axis=1)

    def run_swarm_consensus(self, matrix):
        """Agent 4 (Arbiter): Tổng hợp ý kiến đa tác tử và bỏ phiếu đồng thuận."""
        score_topo = self.agent_topologist(matrix)
        score_phase = self.agent_phase_shifter(matrix)
        score_cluster = self.agent_cluster_miner(matrix)

        # Chuẩn hóa min-max các vector điểm số từ các agent
        def norm(arr):
            mx, mn = arr.max(), arr.min()
            return (arr - mn) / (mx - mn + 1e-9)

        consensus_scores = (0.33 * norm(score_topo)) + (0.34 * norm(score_phase)) + (0.33 * norm(score_cluster))

        # Phân rã cấu trúc thành các phân thớ đầu số
        structure_groups = {
            f"Đầu {i} ({(i*10)+1:02d}-{(i+1)*10:02d})": list(range(i * 10, (i + 1) * 10))
            for i in range(8)
        }

        selected_cores = []
        structural_breakdown = []

        for group_name, indices in structure_groups.items():
            group_scores = {idx + 1: consensus_scores[idx] for idx in indices}
            sorted_group = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            top_2_in_group = sorted_group[:2]
            
            for num, score in top_2_in_group:
                selected_cores.append((num, score, group_name))
            
            structural_breakdown.append({
                "Phân thớ cấu trúc": group_name,
                "Top 2 số đồng thuận": f"{sorted_group[0][0]:02d}, {sorted_group[1][0]:02d}",
                "Chỉ số đồng thuận Agent": round(float(np.mean([s for _, s in sorted_group[:2]])), 4)
            })

        selected_cores = sorted(selected_cores, key=lambda x: x[1], reverse=True)
        final_top_5 = [item[0] for item in selected_cores[:5]]
        swarm_index = float(np.mean(consensus_scores))

        return {
            "swarm_index": swarm_index,
            "final_top_5": final_top_5,
            "structural_breakdown": structural_breakdown,
            "scores": consensus_scores
        }

# ==============================================================================
# GIAO DIỆN STREAMLIT
# ==============================================================================
st.title("🌐 MDM-SWARM v1.0: MULTI-AGENT RESEARCH CORE")
st.caption("Hệ thống Đa tác tử phân tán • Top-p Consensus • Giải mã phi tuyến tính")

with st.sidebar:
    st.header("🤖 Trạng Thái Swarm Agents")
    st.markdown("- **Agent 1 (Topologist):** Active")
    st.markdown("- **Agent 2 (Phase Shifter):** Active")
    st.markdown("- **Agent 3 (Cluster Miner):** Active")
    st.markdown("- **Agent 4 (Arbiter):** Active")
    st.markdown("---")
    target_order = st.slider("Cửa sổ phân tích của Swarm:", 3, 10, 5)

raw_input = st.text_area(
    f"Dán thủ công dữ liệu Keno (Tối thiểu {target_order} kỳ):",
    placeholder="Ví dụ:\nKỳ 1: 06 11 13 18 21 22 25 31 43 44 45 46 53 54 55 64 65 68 76 77\nKỳ 2: ...",
    height=200
)

if raw_input.strip():
    engine = MultiAgentSwarmEngine()
    matrix, total_kies = engine.parse_data(raw_input, min_kies=target_order)

    if matrix is not None:
        res = engine.run_swarm_consensus(matrix)

        st.success(f"🧬 Hệ thống 4 Agent đã hoàn tất quá trình hội tụ và bỏ phiếu trên {total_kies} kỳ!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🌀 Chỉ Số Đồng Thuận Swarm", value=f"{res['swarm_index']:.4f}")
        with col2:
            st.metric(label="⚡ Trạng Thái Tác Tử", value="4 AGENTS SYNCHRONIZED")
        with col3:
            st.metric(label="🎯 Tập Giá Trị Lõi", value="TOP 5 SWARM CORE")

        st.markdown("---")
        st.subheader("🎯 TOP 5 CON SỐ ĐỒNG THUẬN TỪ HỆ THỐNG ĐA TÁC TỬ")
        
        core_str = "  •  ".join([f"**{n:02d}**" for n in res['final_top_5']])
        st.markdown(f"<div style='text-align:center; padding:25px; background:#0E1117; border-radius:10px; border:2px solid #00F0FF; font-size:28px; color:#00F0FF;'>{core_str}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 CHI TIẾT PHÂN RÃ THEO PHÂN THỚ CẤU TRÚC ĐẦU SỐ")
        df_struct = pd.DataFrame(res['structural_breakdown'])
        st.table(df_struct)

    else:
        st.warning(f"Cần tối thiểu {target_order} kỳ dữ liệu để khởi động các Agent (Hệ thống ghi nhận được {total_kies} kỳ).")
else:
    st.info("Hãy dán dữ liệu vào khung trên để đánh thức hệ thống các Agent nghiên cứu chuyên sâu!")
