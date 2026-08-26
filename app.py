import io
import itertools
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN
# =============================================================================
st.set_page_config(
    page_title="AlphaKeno Quantum Engine", page_icon="🪐", layout="wide"
)

st.title("🪐 ALPHAKENO QUANTUM ENGINE v2.0")
st.caption(
    "Hệ thống Tự học & Ghép cụm Bậc cao Keno dựa trên Ma trận Vướng víu Không gian - Thời gian"
)


# =============================================================================
# 2. CORE ENGINE (MÔ HÌNH TOÁN VƯỚNG VÍU & GHÉP CỤM FREQUENCY SEED)
# =============================================================================
class AlphaKenoStreamlitEngine:

    def __init__(self, total_numbers=80):
        self.N = total_numbers
        # Ma trận Vướng víu Lượng tử (81 x 81)
        self.entanglement_matrix = np.ones((self.N + 1, self.N + 1)) * 0.5

    def train_on_history(self, history_draws: list, iterations=3):
        """Huấn luyện tự động trên chuỗi lịch sử kỳ quay."""
        alpha = 0.08  # Tốc độ học
        for _ in range(iterations):
            for t in range(len(history_draws) - 1):
                prev_draw = history_draws[t]
                curr_draw = history_draws[t + 1]

                for x in prev_draw:
                    for y in curr_draw:
                        self.entanglement_matrix[x][y] = (
                            1 - alpha
                        ) * self.entanglement_matrix[x][y] + alpha * 1.0

            # Suy giảm ma trận tự nhiên (Decay)
            self.entanglement_matrix *= 0.99

    def generate_optimal_tickets(
        self,
        history_draws: list,
        seed_clusters: list = None,
        target_size=8,
        top_k_tickets=5,
    ):
        """Ghép cụm bậc cao kết hợp Hạt giống Tần suất (Frequency Seeds)"""
        # 1. Tính tổng năng lượng vướng víu tích tụ cho từng nút (1-80)
        node_energies = np.sum(self.entanglement_matrix, axis=0)[1:]
        top_nodes = np.argsort(node_energies)[-20:] + 1  # Top 20 nút mạnh nhất

        # 2. Thu thập các cụm hạt nhân (Micro-Clusters 3 số)
        candidate_micro_clusters = []

        # Nếu người dùng nạp thêm Cụm Tần suất hạt giống
        if seed_clusters:
            for seed in seed_clusters:
                if len(seed) == 3:
                    candidate_micro_clusters.append(tuple(sorted(seed)))

        # Bổ sung các cụm 3 số từ Top Nút Vướng víu
        generated_combos = list(itertools.combinations(top_nodes, 3))
        candidate_micro_clusters.extend(generated_combos)

        # 3. Chấm điểm năng lượng cho từng cụm micro
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
        best_micro = [c[0] for c in scored_clusters[:10]]

        # 4. Ghép các cụm Micro thành BỘ SỐ BẬC CAO (VD: Keno 8 số)
        high_order_tickets = set()
        for c1, c2 in itertools.combinations(best_micro, 2):
            combined = tuple(sorted(list(set(c1 + c2))))
            if len(combined) == target_size:
                high_order_tickets.add(combined)
            elif len(combined) < target_size:
                # Bù số có vướng víu mạnh nhất vào cho đủ target_size
                for node in top_nodes:
                    if node not in combined:
                        new_t = tuple(sorted(list(combined + (node,))))
                        if len(new_t) == target_size:
                            high_order_tickets.add(new_t)
                            break

        # Chấm điểm toàn bộ vé bậc cao đã tạo
        final_scored_tickets = []
        for ticket in high_order_tickets:
            t_score = 0
            for i, j in itertools.combinations(ticket, 2):
                t_score += self.entanglement_matrix[i][j]
            final_scored_tickets.append((ticket, round(t_score, 2)))

        final_scored_tickets.sort(key=lambda x: x[1], reverse=True)
        return top_nodes, final_scored_tickets[:top_k_tickets]


# =============================================================================
# 3. GIAO DIỆN NẠP DỮ LIỆU STREAMLIT (DATA LOADING PIPELINE)
# =============================================================================
st.sidebar.header("📥 Nạp Dữ Liệu & Cấu Hình")

# Tab chọn nguồn nạp dữ liệu
data_source = st.sidebar.radio(
    "Nguồn dữ liệu lịch sử:",
    ["Tải file CSV/Excel", "Dùng Dữ Liệu Giả Lập Mẫu"],
)

history_draws = []
seed_clusters = []

if data_source == "Tải file CSV/Excel":
    uploaded_file = st.sidebar.file_uploader(
        "Tải file Kết quả Keno (Mỗi dòng là 1 kỳ quay gồm 20 số, phân cách bởi dấu phẩy):",
        type=["csv", "xlsx"],
    )
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=None)
        else:
            df = pd.read_excel(uploaded_file, header=None)

        # Chuyển đổi dataframe thành danh sách lịch sử
        for _, row in df.iterrows():
            numbers = [
                int(x)
                for x in row.dropna().values
                if isinstance(x, (int, float, str)) and str(x).isdigit()
            ]
            if len(numbers) >= 20:
                history_draws.append(numbers[:20])

        st.sidebar.success(f"✅ Đã nạp thành công {len(history_draws)} kỳ quay!")

elif data_source == "Dùng Dữ Liệu Giả Lập Mẫu":
    np.random.seed(42)
    # Giả lập 60 kỳ quay Keno (mỗi kỳ 20 số)
    history_draws = [
        list(np.random.choice(range(1, 81), size=20, replace=False))
        for _ in range(60)
    ]
    st.sidebar.info(f"💡 Đang dùng 60 kỳ quay giả lập hệ thống.")

# NẠP DỮ LIỆU CỤM HẠT GIỐNG TẦN SUẤT (FREQUENCY SEEDS)
st.sidebar.markdown("---")
st.sidebar.subheader("🧬 Nạp Cụm Hạt Giống Tần Suất")
seed_input = st.sidebar.text_area(
    "Nhập các cụm 3 số xuất hiện nhiều nhất (Mỗi cụm 1 dòng, cách nhau bằng dấu phẩy):",
    value="05, 12, 38\n18, 29, 45\n02, 33, 71",
)

if seed_input.strip():
    lines = seed_input.strip().split("\n")
    for line in lines:
        parts = [int(p.strip()) for p in line.split(",") if p.strip().isdigit()]
        if len(parts) == 3:
            seed_clusters.append(parts)

# CẤU HÌNH THAM SỐ KHAI THÁC
st.sidebar.markdown("---")
target_k = st.sidebar.slider("Loại vé Keno muốn đánh (Bậc số):", 4, 10, 8)
train_iters = st.sidebar.slider("Số vòng tự học Vướng Víu:", 1, 10, 3)

# =============================================================================
# 4. LUỒNG THỰC THI & HIỂN THỊ KẾT QUẢ
# =============================================================================
if st.button("🚀 KÍCH HOẠT HỆ THỐNG TỰ HỌC & GHÉP CỤM BẬC CAO", type="primary"):
    if not history_draws:
        st.error("❌ Chưa có dữ liệu lịch sử! Vui lòng tải file hoặc chọn dữ liệu giả lập.")
    else:
        with st.spinner("🌀 Đang xử lý Ma trận Vướng víu và Phân rã Cụm Hạt nhân..."):
            engine = AlphaKenoStreamlitEngine()
            engine.train_on_history(history_draws, iterations=train_iters)

            top_nodes, optimal_tickets = engine.generate_optimal_tickets(
                history_draws,
                seed_clusters=seed_clusters,
                target_size=target_k,
                top_k_tickets=5,
            )

        # TAB HIỂN THỊ KẾT QUẢ
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

        with tab2:
            st.subheader("📌 Top 20 Hạt Nhân Số Tích Tụ Năng Lượng Mạnh Nhất")
            cols = st.columns(5)
            for idx, node in enumerate(reversed(top_nodes)):
                cols[idx % 5].metric(
                    label=f"Hạt nhân #{idx+1}", value=f"Số [{node:02d}]"
                )

        with tab3:
            st.subheader("🌐 Heatmap Ma Trận Vướng Víu Lượng Tử (80 x 80)")
            fig = px.imshow(
                engine.entanglement_matrix[1:, 1:],
                labels=dict(x="Số Kỳ Sau (T+1)", y="Số Kỳ Trước (T)", color="Cường Độ"),
                x=[f"{i:02d}" for i in range(1, 81)],
                y=[f"{i:02d}" for i in range(1, 81)],
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)
