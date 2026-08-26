import io
import itertools
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="AlphaKeno Quantum Engine", page_icon="🪐", layout="wide"
)

st.title("🪐 ALPHAKENO QUANTUM ENGINE v2.1")
st.caption(
    "Hệ thống Tự học & Ghép cụm Bậc cao Keno dựa trên Ma trận Vướng víu Không gian - Thời gian"
)


# =============================================================================
# 2. BỘ ĐỌC DỮ LIỆU BỐ TRÍ THÔNG MINH (AUTO-PARSER)
# =============================================================================
def parse_draws_from_dataframe(df):
    """Bóc tách thông minh: Tự động trích xuất các số từ 1 đến 80 trên từng dòng

    Bất kể file có chứa chữ, ngày tháng hay dấu phân cách gì.
    """
    history_draws = []
    for _, row in df.iterrows():
        # Chuyển toàn bộ dòng thành chuỗi văn bản
        row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
        # Dùng Regex lấy toàn bộ các chuỗi chữ số
        tokens = re.findall(r"\b\d+\b", row_str)

        # Lọc ra các số hợp lệ của Keno (1 đến 80)
        valid_nums = [int(t) for t in tokens if 1 <= int(t) <= 80]

        # Loại bỏ trùng lặp trên cùng 1 kỳ nhưng giữ nguyên thứ tự
        seen = set()
        unique_nums = []
        for n in valid_nums:
            if n not in seen:
                seen.add(n)
                unique_nums.append(n)

        # Nếu dòng có từ 10 số trở lên -> Tính là 1 kỳ quay hợp lệ
        if len(unique_nums) >= 10:
            history_draws.append(unique_nums)

    return history_draws


# =============================================================================
# 3. CORE QUANTUM ENGINE
# =============================================================================
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

        # Thêm cụm hạt giống người dùng nạp vào
        if seed_clusters:
            for seed in seed_clusters:
                valid_seed = [s for s in seed if 1 <= s <= 80]
                if len(valid_seed) >= 2:
                    candidate_micro_clusters.append(
                        tuple(sorted(valid_seed[:3]))
                    )

        # Tạo thêm các cụm 3 số từ Top Vướng Víu
        generated_combos = list(itertools.combinations(top_nodes[:12], 3))
        candidate_micro_clusters.extend(generated_combos)

        # Chấm điểm cụm
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

        # Ghép thành vé Bậc mong muốn (target_size)
        high_order_tickets = set()

        for c1 in best_micro:
            current_ticket = list(c1)
            # Thêm dần các số có năng lượng vướng víu cao nhất cho đủ target_size
            for node in top_nodes:
                if len(current_ticket) == target_size:
                    break
                if node not in current_ticket:
                    current_ticket.append(node)

            if len(current_ticket) == target_size:
                high_order_tickets.add(tuple(sorted(current_ticket)))

        # Chấm điểm toàn bộ vé
        final_scored_tickets = []
        for ticket in high_order_tickets:
            t_score = 0
            for i, j in itertools.combinations(ticket, 2):
                t_score += self.entanglement_matrix[i][j]
            final_scored_tickets.append((ticket, round(t_score, 2)))

        final_scored_tickets.sort(key=lambda x: x[1], reverse=True)
        return top_nodes, final_scored_tickets[:top_k_tickets]


# =============================================================================
# 4. STREAMLIT SIDEBAR & DATA LOADING
# =============================================================================
st.sidebar.header("📥 Nạp Dữ Liệu & Cấu Hình")

data_source = st.sidebar.radio(
    "Nguồn dữ liệu lịch sử:",
    ["Tải file CSV/Excel", "Dùng Dữ Liệu Giả Lập Mẫu"],
)

history_draws = []
seed_clusters = []

if data_source == "Tải file CSV/Excel":
    uploaded_file = st.sidebar.file_uploader(
        "Tải file Kết quả Keno (Hỗ trợ file CSV, XLSX, TXT):",
        type=["csv", "xlsx", "txt"],
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, header=None, on_bad_lines="skip")
            else:
                df = pd.read_excel(uploaded_file, header=None)

            history_draws = parse_draws_from_dataframe(df)

            if len(history_draws) > 0:
                st.sidebar.success(
                    f"✅ Đã nạp thành công {len(history_draws)} kỳ quay!"
                )
            else:
                st.sidebar.error(
                    "❌ Không tìm thấy dãy số hợp lệ (1-80) trong file!"
                )
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file: {str(e)}")

elif data_source == "Dùng Dữ Liệu Giả Lập Mẫu":
    np.random.seed(42)
    history_draws = [
        list(np.random.choice(range(1, 81), size=20, replace=False))
        for _ in range(60)
    ]
    st.sidebar.info(f"💡 Đang dùng 60 kỳ quay mẫu.")

# NẠP CỤM HẠT GIỐNG
st.sidebar.markdown("---")
st.sidebar.subheader("🧬 Nạp Cụm Hạt Giống Tần Suất")
seed_input = st.sidebar.text_area(
    "Nhập các cụm số xuất hiện nhiều nhất (Mỗi cụm 1 dòng, cách nhau bằng dấu phẩy):",
    value="36,43,62\n08,17,56\n07,58,65",
)

if seed_input.strip():
    lines = seed_input.strip().split("\n")
    for line in lines:
        parts = [int(p.strip()) for p in re.findall(r"\d+", line)]
        if len(parts) >= 2:
            seed_clusters.append(parts)

# CẤU HÌNH THAM SỐ
st.sidebar.markdown("---")
target_k = st.sidebar.slider("Loại vé Keno muốn đánh (Bậc số):", 2, 10, 5)
train_iters = st.sidebar.slider("Số vòng tự học Vướng Víu:", 1, 10, 3)


# =============================================================================
# 5. KHU VỰC THỰC THI & HIỂN THỊ
# =============================================================================
if st.button("🚀 KÍCH HOẠT HỆ THỐNG TỰ HỌC & GHÉP CỤM BẬC CAO", type="primary"):
    if not history_draws:
        st.error(
            "❌ Chưa có dữ liệu lịch sử! Vui lòng tải file hợp lệ hoặc chọn dữ liệu giả lập."
        )
    else:
        with st.spinner("🌀 Đang xử lý Ma trận Vướng víu Lượng tử..."):
            engine = AlphaKenoStreamlitEngine()
            engine.train_on_history(history_draws, iterations=train_iters)

            top_nodes, optimal_tickets = engine.generate_optimal_tickets(
                history_draws,
                seed_clusters=seed_clusters,
                target_size=target_k,
                top_k_tickets=5,
            )

        tab1, tab2, tab3 = st.tabs(
            [
                "🎯 Vé Bậc Cao Tối Ưu",
                "🌌 Top Hạt Nhân Vướng Víu",
                "📊 Ma Trận Năng Lượng Không Gian",
            ]
        )

        with tab1:
            st.subheader(
                f"🔥 Top Vé Keno Bậc {target_k} Được Khuyên Nghị Cho Kỳ Tiếp Theo"
            )
            if optimal_tickets:
                for idx, (ticket, score) in enumerate(optimal_tickets, 1):
                    st.markdown(
                        f"""
                    <div style="background-color: #1e222d; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #00ff88;">
                        <h4 style="margin:0; color: #00ff88;">Bộ Vé #{idx} (Điểm Xung Vướng Víu: {score})</h4>
                        <h2 style="margin:5px 0; color: #ffffff; letter-spacing: 2px;">{list(ticket)}</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Chưa tạo được bộ vé. Hãy thử thay đổi tham số.")

        with tab2:
            st.subheader("📌 Top 20 Hạt Nhân Số Tích Tụ Năng Lượng Mạnh Nhất")
            cols = st.columns(5)
            for idx, node in enumerate(reversed(top_nodes[-20:])):
                cols[idx % 5].metric(
                    label=f"Hạt nhân #{idx+1}", value=f"Số [{node:02d}]"
                )

        with tab3:
            st.subheader("🌐 Heatmap Ma Trận Vướng Víu Lượng Tử (80 x 80)")
            fig = px.imshow(
                engine.entanglement_matrix[1:, 1:],
                labels=dict(
                    x="Số Kỳ Sau (T+1)", y="Số Kỳ Trước (T)", color="Cường Độ"
                ),
                x=[f"{i:02d}" for i in range(1, 81)],
                y=[f"{i:02d}" for i in range(1, 81)],
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)
