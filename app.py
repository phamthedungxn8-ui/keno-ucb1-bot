import time
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CORE MATH & AI ENGINES
# -----------------------------------------------------------------------------

class QFPSAIOptimizer:
    """Bộ Tối ưu hóa AI Động lực học Không gian Pha Lượng tử - Mờ (QFPS)."""
    def __init__(self, num_dim: int = 80, hbar: float = 0.1, gamma: float = 0.05, alpha: float = 2.0):
        self.D = num_dim
        self.hbar = hbar
        self.gamma = gamma
        self.alpha = alpha
        raw_psi = np.random.randn(self.D) + 1j * np.random.randn(self.D)
        self.psi = raw_psi / np.linalg.norm(raw_psi)

    def compute_renyi_entropy(self, prob_density: np.ndarray) -> float:
        p = prob_density[prob_density > 1e-12]
        sum_p_sq = np.sum(p ** self.alpha)
        if sum_p_sq <= 0:
            return 1.0
        renyi = (1.0 / (1.0 - self.alpha)) * np.log2(sum_p_sq)
        return float(np.clip(renyi / np.log2(self.D), 0.0, 1.0))

    def step_evolution(self, loss_gradient: np.ndarray, dt: float = 0.01) -> float:
        prob_density = np.abs(self.psi) ** 2
        renyi_s = self.compute_renyi_entropy(prob_density)
        h_eff = loss_gradient - 1j * self.gamma * renyi_s
        self.psi *= np.exp(-1j * h_eff * dt / self.hbar)
        
        grad_norm = np.abs(loss_gradient)
        fuzzy_mask = 1.0 / (1.0 + np.exp(-10 * (grad_norm - np.mean(grad_norm))))
        self.psi *= fuzzy_mask
        
        norm = np.linalg.norm(self.psi)
        if norm > 0:
            self.psi /= norm
        return renyi_s


class KenoMemoryPairEngine:
    """Bộ trích xuất Cặp Bậc 2 dựa trên Động lực học Ký nhớ Ngắn hạn (3-5 Kỳ quay)."""
    def __init__(self, window_size: int = 5, decay_lambda: float = 0.3):
        self.W = window_size
        self.weights = np.exp(-decay_lambda * np.arange(window_size))

    def extract_best_pair(self, history_matrix: np.ndarray, renyi_s: float) -> tuple[tuple[int, int], float, dict]:
        T, N = history_matrix.shape
        if T < self.W:
            return (1, 2), 0.0, {"Cảnh báo": "Không đủ 5 kỳ lịch sử để phân tích"}

        recent = history_matrix[-self.W:]
        m_t = np.dot(self.weights, recent)
        
        co_occurrence = np.dot(recent.T, recent)
        deg = np.diag(co_occurrence)
        norm_factor = np.sqrt(np.outer(deg, deg)) + 1e-9
        c_t = co_occurrence / norm_factor
        np.fill_diagonal(c_t, 0.0)

        m_matrix = np.add.outer(m_t, m_t)
        np.fill_diagonal(m_matrix, 0.0)
        
        score_matrix = (0.6 * m_matrix + 0.4 * c_t) * np.exp(-renyi_s)
        
        best_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        num1, num2 = int(best_idx[0] + 1), int(best_idx[1] + 1)
        
        stats = {
            f"Số {num1}": f"Về {int(np.sum(recent[:, best_idx[0]]))}/{self.W} kỳ gần nhất",
            f"Số {num2}": f"Về {int(np.sum(recent[:, best_idx[1]]))}/{self.W} kỳ gần nhất"
        }
        return (num1, num2), float(score_matrix[best_idx]), stats


class BQFSMEngine:
    """Bộ Quản trị Vốn Lượng tử - Logic Mờ & Sinh trắc học (B-QFSME)."""
    def __init__(self, initial_capital: float = 10000000.0, gamma: float = 0.15):
        self.V0 = initial_capital
        self.Vt = initial_capital
        self.peak_capital = initial_capital
        self.gamma = gamma
        self.fuzzy_vector = [1.0] * 6
        self.capital_history = [initial_capital]
        self.error_history = []

    def update_fuzzy_vector(self, current_mdd: float) -> list[float]:
        cap_ratio = self.Vt / self.V0
        self.fuzzy_vector[0] = 1.0 / (1.0 + np.exp(-15 * (cap_ratio - 0.8)))
        self.fuzzy_vector[1] = 1.0 / (1.0 + np.exp(30 * (current_mdd - 0.15)))
        self.fuzzy_vector[2] = 0.3 if (self.error_history and self.error_history[-1] == 1) else 1.0
        self.fuzzy_vector[3] = float(np.clip(cap_ratio, 0.0, 1.0))
        self.fuzzy_vector[4] = 1.0 / (1.0 + np.exp(20 * (current_mdd - 0.20)))
        self.fuzzy_vector[5] = 1.0 / (1.0 + np.exp(40 * (current_mdd - 0.25)))
        return self.fuzzy_vector

    def calculate_position_size(self, renyi_s: float, reaction_time: float, odds: float = 9.0, p_win: float = 0.0601) -> tuple[float, float, str]:
        current_mdd = (self.peak_capital - self.Vt) / self.peak_capital if self.peak_capital > 0 else 0.0
        self.update_fuzzy_vector(current_mdd)
        
        recent_errors = sum(self.error_history[-5:]) if self.error_history else 0
        if recent_errors >= 2 or self.fuzzy_vector[5] < 0.1:
            return 0.0, 0.0, "LOCKOUT: Kích hoạt Cầu chì Sinh học / Cầu chì Vốn!"
        
        beta_bio = 1.0
        status_msg = "OPTIMAL: Trạng thái ổn định."
        if reaction_time < 1.5:
            beta_bio = 0.3
            status_msg = "WARNING: Thao tác quá nhanh (<1.5s). Giảm 70% vị thế."
        elif recent_errors == 1:
            beta_bio = 0.5
            status_msg = "CAUTION: Có 1 vi phạm kỷ luật. Giảm 50% vị thế."

        b = odds - 1.0
        q = 1.0 - p_win
        base_kelly = max((p_win * b - q) / b, 0.01)
        fuzzy_weight = self.fuzzy_vector[0] * self.fuzzy_vector[1] * self.fuzzy_vector[5]
        
        f_star = base_kelly * fuzzy_weight * (1.0 - renyi_s) * beta_bio * self.gamma
        f_star = float(np.clip(f_star, 0.0, 0.025))
        
        bet_amount = np.floor((self.Vt * f_star) / 10000.0) * 10000.0
        return f_star, bet_amount, status_msg


# -----------------------------------------------------------------------------
# 2. STREAMLIT APPLICATION INTERFACE
# -----------------------------------------------------------------------------

st.set_page_config(page_title="QFPS Keno Engine", page_icon="🎲", layout="wide")

# State Initialization
if "ai_opt" not in st.session_state:
    st.session_state.ai_opt = QFPSAIOptimizer()
if "pair_engine" not in st.session_state:
    st.session_state.pair_engine = KenoMemoryPairEngine()
if "bqfsm" not in st.session_state:
    st.session_state.bqfsm = BQFSMEngine()

# Khởi tạo Mẫu DataFrame Lịch Sử Ban Đầu
if "df_history_editor" not in st.session_state:
    st.session_state.df_history_editor = pd.DataFrame([
        {"Kỳ Xổ": "#296688", "Chẵn/Lẻ": "Lẻ 12", "Lớn/Nhỏ": "Lớn 11", "20 Con Số Thực Tế (Phân cách dấu phẩy)": "01, 03, 04, 08, 19, 23, 26, 27, 30, 39, 43, 55, 57, 58, 67, 68, 70, 73, 75, 78"},
        {"Kỳ Xổ": "#296687", "Chẵn/Lẻ": "Chẵn 12", "Lớn/Nhỏ": "Nhỏ 11", "20 Con Số Thực Tế (Phân cách dấu phẩy)": "08, 18, 20, 21, 22, 23, 25, 28, 31, 32, 38, 48, 54, 55, 57, 62, 75, 77, 78, 80"},
        {"Kỳ Xổ": "#296686", "Chẵn/Lẻ": "Chẵn 12", "Lớn/Nhỏ": "Nhỏ 13", "20 Con Số Thực Tế (Phân cách dấu phẩy)": "03, 06, 11, 15, 16, 19, 20, 22, 26, 28, 34, 38, 40, 49, 57, 65, 66, 68, 73, 76"},
        {"Kỳ Xổ": "#296685", "Chẵn/Lẻ": "Lẻ 13", "Lớn/Nhỏ": "Lớn 11", "20 Con Số Thực Tế (Phân cách dấu phẩy)": "02, 03, 05, 07, 09, 11, 12, 15, 40, 45, 47, 49, 50, 51, 52, 59, 67, 69, 70, 80"},
        {"Kỳ Xổ": "#296684", "Chẵn/Lẻ": "Chẵn 14", "Lớn/Nhỏ": "Nhỏ 12", "20 Con Số Thực Tế (Phân cách dấu phẩy)": "02, 10, 14, 16, 18, 21, 22, 23, 24, 25, 26, 40, 43, 46, 47, 51, 52, 60, 74, 78"}
    ])

# Khởi tạo Ma trận Lịch sử 80 số từ Bảng
def build_matrix_from_df(df):
    rows = []
    for _, row in df.iterrows():
        raw_nums = str(row["20 Con Số Thực Tế (Phân cách dấu phẩy)"])
        parsed = [int(s.strip()) for s in raw_nums.replace(",", " ").split() if s.strip().isdigit()]
        if len(parsed) == 20:
            rows.append(parsed)
    
    T = len(rows)
    matrix = np.zeros((max(T, 1), 80))
    for t_idx, nums in enumerate(rows):
        for n in nums:
            if 1 <= n <= 80:
                matrix[t_idx, n - 1] = 1
    return matrix

if "history_matrix" not in st.session_state:
    st.session_state.history_matrix = build_matrix_from_df(st.session_state.df_history_editor)

if "last_pair" not in st.session_state:
    st.session_state.last_pair = (51, 61)
if "last_bet_amount" not in st.session_state:
    st.session_state.last_bet_amount = 10000.0
if "last_click_time" not in st.session_state:
    st.session_state.last_click_time = time.time()

bqfsm = st.session_state.bqfsm
ai_opt = st.session_state.ai_opt
pair_engine = st.session_state.pair_engine

st.title("🎲 QFPS Keno Engine: Tối Ưu Bậc 2 & Quản Lý Vốn")

# -----------------------------------------------------------------------------
# METRICS DASHBOARD
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
current_mdd = (bqfsm.peak_capital - bqfsm.Vt) / bqfsm.peak_capital if bqfsm.peak_capital > 0 else 0.0
m1.metric("Vốn Hiện Tại (Vt)", f"{bqfsm.Vt:,.0f} VNĐ", delta=f"MDD: {current_mdd*100:.1f}%")

fake_grad = np.sin(np.sum(st.session_state.history_matrix[-5:], axis=0) * 0.5)
renyi_s = ai_opt.step_evolution(fake_grad)

m2.metric("Rényi Entropy (α=2)", f"{renyi_s:.4f}")
m3.metric("Fuzzy Health", f"{bqfsm.fuzzy_vector[0]:.2f}")
m4.metric("Cầu Chì Vốn", "SAFE" if bqfsm.fuzzy_vector[5] > 0.1 else "LOCKED")

st.divider()

# -----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📋 Bảng Nạp Lịch Sử 5 Kỳ", "📥 Cập Nhật Kỳ Quay Hiện Tại", "🔬 Mổ Xẻ Chi Tiết Kỳ Quay"])

# TAB 1: BẢNG NẠP LỊCH SỬ DẠNG MINH CHÍNH / VIETLOTT
with tab1:
    st.subheader("📋 Bảng Quản Lý & Nạp Lịch Sử 5-10 Kỳ Quay Chi Tiết")
    st.caption("Cấu trúc dạng Bảng giúp đối soát mã kỳ quay, tỷ lệ Chẵn/Lẻ, Lớn/Nhỏ và 20 con số chuẩn xác.")

    edited_df = st.data_editor(
        st.session_state.df_history_editor,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Kỳ Xổ": st.column_config.TextColumn("Mã Kỳ Xổ", help="VD: #296688", width="small"),
            "Chẵn/Lẻ": st.column_config.TextColumn("Thống kê C/L", width="small"),
            "Lớn/Nhỏ": st.column_config.TextColumn("Thống kê L/N", width="small"),
            "20 Con Số Thực Tế (Phân cách dấu phẩy)": st.column_config.TextColumn("Danh sách 20 số trúng", width="large")
        }
    )

    if st.button("🚀 Đồng Bộ Vào Ma Trận AI 80 Số", type="primary"):
        errors = []
        valid_rows = []

        for idx, row in edited_df.iterrows():
            raw_nums = str(row["20 Con Số Thực Tế (Phân cách dấu phẩy)"])
            parsed = [int(s.strip()) for s in raw_nums.replace(",", " ").split() if s.strip().isdigit()]
            
            if len(parsed) != 20:
                errors.append(f"Kỳ `{row['Kỳ Xổ']}` đang có {len(parsed)} số (yêu cầu đúng 20 số).")
            else:
                if len(set(parsed)) != 20:
                    errors.append(f"Kỳ `{row['Kỳ Xổ']}` có số bị trùng lặp!")
                else:
                    valid_rows.append(parsed)

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            T = len(valid_rows)
            new_matrix = np.zeros((T, 80))
            for t_idx, nums in enumerate(valid_rows):
                for n in nums:
                    if 1 <= n <= 80:
                        new_matrix[t_idx, n - 1] = 1

            st.session_state.history_matrix = new_matrix
            st.session_state.df_history_editor = edited_df
            st.success(f"🎉 Đã chuyển đổi thành công {T} kỳ quay chuẩn hóa thành Ma trận Lịch sử Lượng tử!")

# TAB 2: CẬP NHẬT KỲ VỪA QUAY VÀ TÍNH LÃI/LỖ
with tab2:
    st.subheader("Cập nhật 20 số thực tế của kỳ vừa quay")
    
    with st.form("actual_draw_form"):
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            drawn_numbers = st.multiselect(
                "Chọn đủ 20 con số thực tế vừa mở thưởng:",
                options=list(range(1, 81)),
                default=list(range(1, 21))
            )
        
        with col_f2:
            fee_per_ticket = st.selectbox(
                "Phí dịch vụ mua hộ / vé 10k:",
                options=[200, 500, 800],
                index=1
            )
            violation = st.checkbox("Vi phạm kỷ luật (Tâm lý / Cược sai tiền)")

        submit_draw = st.form_submit_button("⚡ Tính Lãi/Lỗ Thực Tế & Tiến Hóa AI")

    if submit_draw:
        if len(drawn_numbers) != 20:
            st.error("❌ Vui lòng chọn đúng đủ 20 con số thực tế!")
        else:
            now = time.time()
            reaction_time = now - st.session_state.last_click_time
            st.session_state.last_click_time = now

            # 1. Nối kỳ quay thực tế mới vào Ma trận Lịch sử
            new_row = np.zeros((1, 80))
            for num in drawn_numbers:
                new_row[0, num - 1] = 1
            st.session_state.history_matrix = np.vstack([st.session_state.history_matrix, new_row])

            # 2. Đánh giá Cặp Bậc 2 đã cược ở kỳ trước
            pair_a, pair_b = st.session_state.last_pair
            is_win = (pair_a in drawn_numbers) and (pair_b in drawn_numbers)

            # 3. Tính Lãi/Lỗ thực tế (Keno Bậc 2: 10k -> nhận 90k)
            num_tickets = st.session_state.last_bet_amount / 10000.0
            total_fee = num_tickets * fee_per_ticket
            
            if is_win:
                payout = num_tickets * 90000.0
                net_pnl = payout - st.session_state.last_bet_amount - total_fee
                st.balloons()
                st.success(f"🎉 TRÚNG BẬC 2 (Cặp {pair_a} - {pair_b})! Lãi ròng: +{net_pnl:,.0f} VNĐ (Đã trừ {total_fee:,.0f} VNĐ phí dịch vụ)")
            else:
                net_pnl = -(st.session_state.last_bet_amount + total_fee)
                st.error(f"❌ TRẬT BẬC 2 (Cặp {pair_a} - {pair_b}). Lỗ: {net_pnl:,.0f} VNĐ (Gồm {total_fee:,.0f} VNĐ phí mua hộ)")

            # Cập nhật vốn
            bqfsm.Vt += net_pnl
            bqfsm.capital_history.append(bqfsm.Vt)
            if bqfsm.Vt > bqfsm.peak_capital:
                bqfsm.peak_capital = bqfsm.Vt
            bqfsm.error_history.append(1 if violation else 0)

            # 4. Trích xuất Cặp Bậc 2 Tối ưu cho Kỳ t+1
            next_pair, score, stats = pair_engine.extract_best_pair(st.session_state.history_matrix, renyi_s)
            f_star, next_bet_amt, status_msg = bqfsm.calculate_position_size(renyi_s, reaction_time)

            st.session_state.last_pair = next_pair
            st.session_state.last_bet_amount = next_bet_amt

            st.divider()
            st.markdown("### 🎯 GỢI Ý CẶP BẬC 2 & KHUYẾN NGHỊ VỐN KỲ t+1")
            
            res1, res2, res3 = st.columns(3)
            res1.metric("Cặp Bậc 2 Tối Ưu Mới", f"Số {next_pair[0]} - Số {next_pair[1]}")
            res2.metric("Điểm Cộng Hưởng Ký Nhớ", f"{score:.4f}")
            res3.metric("Tiền Nên Vào (Gốc)", f"{next_bet_amt:,.0f} VNĐ", delta=f"Tỷ lệ: {f_star*100:.2f}%")

            st.info(f"💬 **Trạng thái Quản trị Vốn:** {status_msg}")

# TAB 3: MỔ XẺ CHI TIẾT TẦN SUẤT VÀ CỤM LẶP
with tab3:
    st.subheader("🔬 Phân tích Tần suất Lặp (Window W = 5 Kỳ gần nhất)")
    
    recent_5 = st.session_state.history_matrix[-5:]
    counts = np.sum(recent_5, axis=0)
    
    df_analysis = pd.DataFrame({
        "Con Số": list(range(1, 81)),
        "Số lần về (5 kỳ)": counts,
        "Trạng Thái Lặp": ["🔥🔥 Siêu Hot" if c >= 3 else ("🔥 Hot" if c == 2 else "Bình thường") for c in counts]
    })
    
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        st.write("Top Con Số Lặp Mạnh Nhất:")
        st.dataframe(df_analysis[df_analysis["Số lần về (5 kỳ)"] >= 2].sort_values(by="Số lần về (5 kỳ)", ascending=False), hide_index=True)
    
    with col_a2:
        st.write("Biểu đồ Tần suất Lặp 80 Số")
        st.bar_chart(df_analysis.set_index("Con Số")["Số lần về (5 kỳ)"])

# BIỂU ĐỒ TĂNG TRƯỞNG VỐN
if len(bqfsm.capital_history) > 1:
    st.divider()
    st.subheader("📈 Biểu Đồ Tăng Trưởng Vốn Thực Tế")
    st.line_chart(pd.DataFrame({"Vốn Thực Tế (VNĐ)": bqfsm.capital_history}))
