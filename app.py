import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import hilbert
import streamlit as st

st.set_page_config(
    page_title="Keno Brain Oscillators Engine", page_icon="🧠", layout="wide"
)

st.title("🧠 KENO BRAIN OSCILLATOR & ALPHA RHYTHM ENGINE")
st.caption(
    "Mô hình hóa 80 con số Keno thành hệ thống các bộ dao động liên kết"
    " (Coupled Oscillators) & Đồng bộ Pha Kuramoto"
)


# =============================================================================
# 1. GIẢ LẬP / CÀO DỮ LIỆU
# =============================================================================
@st.cache_data(ttl=300)
def generate_sample_history(num_draws=100):
    np.random.seed(42)
    draws = [
        sorted(list(np.random.choice(range(1, 81), 20, replace=False)))
        for _ in range(num_draws)
    ]
    return draws


history_data = generate_sample_history(100)

# =============================================================================
# 2. THUẬT TOÁN BỘ NÃO DAO ĐỘNG (CORE ENGINE)
# =============================================================================
num_draws = len(history_data)
matrix = np.zeros((80, num_draws))

for t, draw in enumerate(history_data):
    for num in draw:
        if 1 <= num <= 80:
            matrix[num - 1, t] = 1.0
matrix[matrix == 0] = -1.0

# Trích xuất pha tức thời bằng Hilbert Transform
analytic_signal = hilbert(matrix, axis=1)
phases = np.angle(analytic_signal)

# Tính Nhịp Alpha (Chỉ số trật tự Kuramoto R) qua từng kỳ
order_parameters = []
for t in range(num_draws):
    R_t = np.abs(np.mean(np.exp(1j * phases[:, t])))
    order_parameters.append(R_t)

current_alpha = order_parameters[-1]

# Tính Ma trận Khóa Pha (PLV) cho 30 kỳ gần nhất
recent_phases = phases[:, -30:]
plv_matrix = np.zeros((80, 80))
for i in range(80):
    for j in range(i + 1, 80):
        phase_diff = recent_phases[i, :] - recent_phases[j, :]
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        plv_matrix[i, j] = plv
        plv_matrix[j, i] = plv

# Tính Tổng lực kéo tần số (Coupling Force) của từng số
coupling_force = np.sum(plv_matrix, axis=1)
latest_phases = phases[:, -1]

# Mức độ sẵn sàng bùng nổ (Pha tiệm cận đỉnh cos(theta) -> 1)
readiness_score = coupling_force * np.cos(latest_phases)

# =============================================================================
# 3. HIỂN THỊ KẾT QUẢ KIỂM CHỨNG TRÊN GIAO DIỆN
# =============================================================================
st.sidebar.header("⚙️ CẤU HÌNH THUẬT TOÁN")
target_k = st.sidebar.slider("Chọn Bậc Keno (Số con/vé):", 2, 10, 6)
top_n = st.sidebar.slider("Số lượng bộ vé gợi ý:", 3, 10, 5)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tỉnh trạng Nhịp Alpha (Kuramoto R)", f"{current_alpha:.4f}")
with col2:
    st.metric(
        "Mức độ Cộng hưởng Hệ thống",
        "CAO 🔥" if current_alpha > 0.15 else "BÌNH THƯỜNG 🌊",
    )
with col3:
    st.metric("Dữ liệu Phân tích", f"{num_draws} Kỳ gần nhất")

st.markdown("---")

# Tab hiển thị chi tiết
tab1, tab2, tab3 = st.tabs(
    ["🎯 TOP BỘ VÉ TỐI ƯU", "📊 BẢN ĐỒ KHÓA PHA (PLV)", "📈 BIỂU ĐỒ NHỊP ALPHA"]
)

with tab1:
    st.subheader(f"🔥 TOP {top_n} BỘ VÉ BẬC {target_k} CÓ ĐỘ ĐỒNG BỘ PHA CAO NHẤT")

    # Sắp xếp các số theo Readiness Score
    ranked_numbers = np.argsort(readiness_score)[::-1] + 1

    selected_tickets = []
    for i in range(top_n):
        ticket = sorted(
            list(ranked_numbers[i * target_k : (i + 1) * target_k])
        )
        # Tính Sync Score nội bộ của bộ vé
        sub_plv = [
            plv_matrix[a - 1, b - 1]
            for a in ticket
            for b in ticket
            if a != b
        ]
        avg_sync = np.mean(sub_plv) if sub_plv else 0
        selected_tickets.append((ticket, avg_sync))

    for idx, (ticket, sync) in enumerate(selected_tickets, 1):
        st.markdown(
            f"#### #{idx}. `{ticket}` — **PLV Sync Score:** `{sync:.4f}`"
        )

with tab2:
    st.subheader("Bản đồ Nhiệt Khóa Pha (Phase-Locking Matrix - 80x80)")
    st.caption(
        "Các điểm màu sáng thể hiện cặp số có lực kéo tần số mạnh mẽ,"
        " thường xuyên khóa pha với nhau."
    )
    fig_heatmap = px.imshow(
        plv_matrix,
        labels=dict(x="Mã số Keno", y="Mã số Keno", color="Độ gắn kết PLV"),
        x=list(range(1, 81)),
        y=list(range(1, 81)),
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.subheader("Biến động Nhịp Alpha (Global Order Parameter) qua thời gian")
    st.caption(
        "Những đỉnh cao thể hiện thời điểm toàn bộ 80 bộ dao động đạt trạng"
        " thái đồng bộ cực đại."
    )
    df_alpha = pd.DataFrame(
        {"Kỳ quay": list(range(1, num_draws + 1)), "Năng lượng Alpha": order_parameters}
    )
    fig_alpha = px.line(
        df_alpha,
        x="Kỳ quay",
        y="Năng lượng Alpha",
        markers=True,
        line_shape="spline",
    )
    st.plotly_chart(fig_alpha, use_container_width=True)
