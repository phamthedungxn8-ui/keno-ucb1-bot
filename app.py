import itertools
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="AlphaKeno Quantum Engine", page_icon="🪐", layout="wide"
)

st.title("🪐 ALPHAKENO QUANTUM ENGINE v2.3")
st.caption(
    "Hệ thống Tự học & Ghép cụm Bậc cao Keno dựa trên Ma trận Vướng víu Không gian"
)


# BỘ LỌC DỮ LIỆU CHUẨN
def parse_draws_from_dataframe(df):
    history_draws = []
    for _, row in df.iterrows():
        row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
        tokens = re.findall(r"\b\d+\b", row_str)
        valid_nums = [int(t) for t in tokens if 1 <= int(t) <= 80]
        seen = set()
        unique_nums = [n for n in valid_nums if not (n in seen or seen.add(n))]
        if len(unique_nums) >= 10:
            history_draws.append(unique_nums)
    return history_draws


class AlphaKenoStreamlitEngine:

    def __init__(self, total_numbers=80):
        self.N = total_numbers
        self.entanglement_matrix = np.ones((self.N + 1, self.N + 1)) * 0.5

    def train_on_history(self, history_draws: list, iterations=3):
        alpha = 0.08
        for _ in range(iterations):
            for t in range(len(history_draws) - 1):
                prev_draw = history_draws[t]
                curr_draw = history_draws[t + 1]

                for x in prev_draw:
                    for y in curr_draw:
                        if x <= 80 and y <= 80:
                            self.entanglement_matrix[x][y] = (
                                1 - alpha
                            ) * self.entanglement_matrix[x][y] + alpha * 1.0

            self.entanglement_matrix *= 0.99

    def apply_intuition_bias(self, intuition_numbers: list, bias_factor=1.35):
        for num in intuition_numbers:
            if 1 <= num <= 80:
                self.entanglement_matrix[num, :] *= bias_factor
                self.entanglement_matrix[:, num] *= bias_factor
        max_val = np.max(self.entanglement_matrix)
        if max_val > 0:
            self.entanglement_matrix /= max_val

    def generate_optimal_tickets(
        self,
        history_draws: list,
        seed_clusters: list = None,
        target_size=8,
        top_k_tickets=5,
    ):
        node_energies = np.sum(self.entanglement_matrix, axis=0)[1:]
        top_nodes = list(np.argsort(node_energies)[-25:] + 1)

        candidate_micro_clusters = []
        if seed_clusters:
            for seed in seed_clusters:
                valid_seed = [s for s in seed if 1 <= s <= 80]
                if len(valid_seed) >= 2:
                    candidate_micro_clusters.append(
                        tuple(sorted(valid_seed[:3]))
                    )

        generated_combos = list(itertools.combinations(top_nodes[:12], 3))
        candidate_micro_clusters.extend(generated_combos)

        scored_clusters = []
        for cluster in set(candidate_micro_clusters):
            score = 0
            for i, j in itertools.combinations(cluster, 2):
                score += (
                    self.entanglement_matrix[i][j]
                    + self.entanglement_matrix[j][i]
                )
            scored_clusters.append((cluster, score))

        scored_clusters.sort(key=lambda x: x[1], reverse=True)
        best_micro = [c[0] for c in scored_clusters[:15]]

        high_order_tickets = set()
        for c1 in best_micro:
            current_ticket = list(c1)
            for node in top_nodes:
                if len(current_ticket) == target_size:
                    break
                if node not in current_ticket:
                    current_ticket.append(node)

            if len(current_ticket) == target_size:
                high_order_tickets.add(tuple(sorted(current_ticket)))

        final_scored_tickets = []
        for ticket in high_order_tickets:
            t_score = 0
            for i, j in itertools.combinations(ticket, 2):
                t_score += self.entanglement_matrix[i][j]
            final_scored_tickets.append((ticket, round(t_score, 2)))

        final_scored_tickets.sort(key=lambda x: x[1], reverse=True)
        return top_nodes, final_scored_tickets[:top_k_tickets]


# SIDEBAR SETUP
st.sidebar.header("📥 Nạp Dữ Liệu & Cấu Hình")
data_source = st.sidebar.radio(
    "Nguồn dữ liệu:", ["Dùng Dữ Liệu Giả Lập Mẫu", "Tải file CSV/Excel"]
)

history_draws = []
if data_source == "Dùng Dữ Liệu Giả Lập Mẫu":
    np.random.seed(42)
    history_draws = [
        list(np.random.choice(range(1, 81), size=20, replace=False))
        for _ in range(60)
    ]
    st.sidebar.info("💡 Đang dùng 60 kỳ quay mẫu hệ thống.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Tải file Kết quả:", type=["csv", "xlsx", "txt"]
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, header=None, on_bad_lines="skip")
            else:
                df = pd.read_excel(uploaded_file, header=None)
            history_draws = parse_draws_from_dataframe(df)
            if history_draws:
                st.sidebar.success(f"✅ Đã nạp thành công {len(history_draws)} kỳ!")
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Trực Giác Tâm Thức")
intuition_input = st.sidebar.text_input("Các số trực giác (cách nhau bởi dấu phẩy):", value="37, 68")
intuition_numbers = [int(x.strip()) for x in intuition_input.split(",") if x.strip().isdigit()]

st.sidebar.subheader("🧬 Cụm Hạt Giống")
seed_input = st.sidebar.text_area("Cụm 3 số hạt giống:", value="36,43,62\n08,17,56")
seed_clusters = []
if seed_input.strip():
    for line in seed_input.strip().split("\n"):
        parts = [int(p.strip()) for p in re.findall(r"\d+", line)]
        if len(parts) >= 2:
            seed_clusters.append(parts)

target_k = st.sidebar.slider("Loại vé Keno (Bậc số):", 2, 10, 5)

# THỰC THI
if st.button("🚀 KÍCH HOẠT THUẬT TOÁN", type="primary"):
    if not history_draws:
        st.error("❌ Chưa có dữ liệu lịch sử!")
    else:
        with st.spinner("🌀 Đang xử lý Ma trận Vướng víu..."):
            engine = AlphaKenoStreamlitEngine()
            engine.train_on_history(history_draws, iterations=3)
            if intuition_numbers:
                engine.apply_intuition_bias(intuition_numbers, bias_factor=1.35)

            top_nodes, optimal_tickets = engine.generate_optimal_tickets(
                history_draws,
                seed_clusters=seed_clusters,
                target_size=target_k,
                top_k_tickets=5,
            )

        st.subheader(f"🔥 Top Vé Keno Bậc {target_k} Được Khuyên Nghị")
        for idx, (ticket, score) in enumerate(optimal_tickets, 1):
            st.markdown(
                f"""
            <div style="background-color: #1e222d; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #00ff88;">
                <h4 style="margin:0; color: #00ff88;">Bộ Vé #{idx} (Điểm Xung: {score})</h4>
                <h2 style="margin:5px 0; color: #ffffff; letter-spacing: 2px;">{list(ticket)}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )
