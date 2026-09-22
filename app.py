import time
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="B-QFSME Next-Gen Keno Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# CORE ENGINE: BIOMETRIC-QUANTUM-FUZZY STATE MACHINE (B-QFSME)
# -----------------------------------------------------------------------------
class BQFSMEngine:

    def __init__(
        self,
        initial_capital: float = 10000000.0,
        gamma: float = 0.15,
        alpha: float = 2.0,
    ):
        self.V0 = initial_capital
        self.Vt = initial_capital
        self.gamma = gamma
        self.alpha = alpha  # Order Rényi Entropy
        self.peak_capital = initial_capital

        # Fuzzy State Vector continuous in [0.0, 1.0]
        self.fuzzy_vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.capital_history = [initial_capital]
        self.error_history = []  # Lịch sử ghi nhận vi phạm kỷ luật (0 hoặc 1)
        self.reaction_times = []  # Lịch sử độ trễ thao tác (giây)

    def calculate_renyi_entropy(self, window_size: int = 5) -> float:
        """Tính Rényi Entropy (alpha=2) cửa sổ N=5 nhằm triệt tiêu trễ tín

        hiệu.
        """
        if len(self.capital_history) <= window_size:
            return 0.0

        recent = self.capital_history[-window_size:]
        returns = np.diff(recent) / (recent[:-1] + 1e-9)

        if np.all(returns == 0):
            return 0.0

        hist, _ = np.histogram(returns, bins=3)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]

        sum_p_alpha = np.sum(probs**self.alpha)
        if sum_p_alpha <= 0:
            return 1.0

        renyi = (1.0 / (1.0 - self.alpha)) * np.log2(sum_p_alpha)
        max_renyi = np.log2(3)
        return float(min(max(renyi / max_renyi, 0.0), 1.0))

    def update_fuzzy_vector(self, current_mdd: float) -> list:
        """Chuyển đổi trạng thái nhị phân sang Logic Mờ Sigmoid liên tục."""
        # b1: Bảo toàn Vốn khả dụng quanh 0.8 * V0
        cap_ratio = self.Vt / self.V0
        self.fuzzy_vector[0] = 1.0 / (1.0 + np.exp(-15 * (cap_ratio - 0.8)))

        # b2: Giới hạn Maximum Drawdown quanh 15%
        self.fuzzy_vector[1] = 1.0 / (1.0 + np.exp(30 * (current_mdd - 0.15)))

        # b3: Trạng thái thao tác sinh trắc học
        # b4: Tốc độ hồi phục dòng tiền
        # b5: Ngưỡng an toàn Kelly
        self.fuzzy_vector[2] = 1.0 if len(self.error_history) == 0 or self.error_history[-1] == 0 else 0.3
        self.fuzzy_vector[3] = 1.0 if cap_ratio >= 1.0 else cap_ratio
        self.fuzzy_vector[4] = 1.0 / (1.0 + np.exp(20 * (current_mdd - 0.20)))

        # b6: Cầu chì ngắt cưỡng bức quanh 25% MDD
        self.fuzzy_vector[5] = 1.0 / (1.0 + np.exp(40 * (current_mdd - 0.25)))

        return self.fuzzy_vector

    def calculate_biometric_factor(self, current_reaction_time: float) -> tuple[float, str]:
        """Đo lường ma sát sinh trắc học & tâm lý dựa trên nhịp độ thao tác và

        Tilt index.
        """
        self.reaction_times.append(current_reaction_time)
        recent_errors = sum(self.error_history[-5:]) if self.error_history else 0

        # Khóa cưỡng bức do vi phạm kỷ luật liên tiếp (Tilt Lockout)
        if recent_errors >= 2:
            return 0.0, "LOCKOUT: Phát hiện Tilt/Tâm lý kém. Cầu chì sinh học kích hoạt!"

        # Kiểm tra thao tác quá nhanh (Spamming / Cảm xúc cuống quýt)
        if current_reaction_time < 1.5:
            return 0.3, "WARNING: Nhập liệu quá nhanh (< 1.5s). Giảm 70% vị thế do nguy cơ cảm xúc."

        if recent_errors == 1:
            return 0.5, "CAUTION: Phát hiện 1 lần vi phạm kỷ luật. Giảm 50% vị thế."

        return 1.0, "OPTIMAL: Trạng thái tâm lý & thao tác ổn định."

    def detect_markov_regime(self, recent_outcomes: list) -> str:
        if not recent_outcomes or len(recent_outcomes) < 5:
            return "S3"
        positive_count = sum(1 for x in recent_outcomes[-10:] if x > 0)
        ratio = positive_count / len(recent_outcomes[-10:])
        if ratio >= 0.65:
            return "S1"
        elif ratio <= 0.35:
            return "S2"
        return "S3"

    def select_payoff_target(self, regime: str) -> dict:
        if regime == "S2" or self.fuzzy_vector[0] < 0.5:
            return {"type": "Keno Bậc 2 / Chẵn Lẻ", "p_win": 0.1739, "odds": 6.0}
        elif regime == "S1" and self.fuzzy_vector[4] > 0.7:
            return {"type": "Keno Bậc 4", "p_win": 0.0264, "odds": 100.0}
        return {"type": "Keno Bậc 2", "p_win": 0.1739, "odds": 6.0}

    def process_cycle(
        self,
        last_outcome: float,
        recent_outcomes: list,
        execution_error: bool,
        reaction_time: float,
    ) -> dict:
        self.Vt += last_outcome
        self.capital_history.append(self.Vt)
        self.error_history.append(1 if execution_error else 0)

        if self.Vt > self.peak_capital:
            self.peak_capital = self.Vt

        current_mdd = (
            (self.peak_capital - self.Vt) / self.peak_capital
            if self.peak_capital > 0
            else 0.0
        )

        self.update_fuzzy_vector(current_mdd)
        renyi_s = self.calculate_renyi_entropy()
        beta_bio, bio_status = self.calculate_biometric_factor(reaction_time)

        # Cầu chì tổng ngắt nếu Fuzzy Vector b6 quá thấp hoặc Beta Bio = 0
        if self.fuzzy_vector[5] < 0.1 or beta_bio == 0.0:
            return {
                "action": "CIRCUIT_BREAKER_HALT",
                "fuzzy_vector": [round(b, 2) for b in self.fuzzy_vector],
                "bet_amount": 0.0,
                "f_star_pct": "0.0%",
                "renyi_entropy": round(renyi_s, 4),
                "bio_status": bio_status,
                "target_game": "KHÔNG ĐẶT CƯỢC",
            }

        regime = self.detect_markov_regime(recent_outcomes)
        payoff = self.select_payoff_target(regime)

        b = payoff["odds"] - 1.0
        q = 1.0 - payoff["p_win"]
        base_kelly = max((payoff["p_win"] * b - q) / b, 0.01)

        # Tổng hợp trọng số Mờ
        fuzzy_weight = (
            self.fuzzy_vector[0] * self.fuzzy_vector[1] * self.fuzzy_vector[5]
        )

        # Công thức Vị thế B-QFSME
        f_qfsm = base_kelly * fuzzy_weight * (1.0 - renyi_s) * beta_bio * self.gamma
        f_star = float(min(max(f_qfsm, 0.0), 0.025))
        bet_amount = self.Vt * f_star

        return {
            "action": "EXECUTE_BET",
            "fuzzy_vector": [round(b, 2) for b in self.fuzzy_vector],
            "regime": regime,
            "target_game": payoff["type"],
            "renyi_entropy": round(renyi_s, 4),
            "beta_bio": round(beta_bio, 2),
            "bio_status": bio_status,
            "f_star_pct": f"{round(f_star * 100, 2)}%",
            "bet_amount": round(bet_amount, 2),
            "current_capital": round(self.Vt, 2),
        }


# -----------------------------------------------------------------------------
# STREAMLIT UI & INTERACTION LAYER
# -----------------------------------------------------------------------------
st.title("⚡ B-QFSME Keno Next-Gen Engine")
st.caption("Khung Quản trị Vốn Lượng tử - Logic Mờ & Ma sát Sinh trắc học Thực chiến")

# Sidebar Configuration
st.sidebar.header("⚙️ Cấu hình Hệ thống")
initial_cap = st.sidebar.number_input(
    "Vốn Ban Đầu V0 (VNĐ)", value=10000000, step=1000000
)
gamma_val = st.sidebar.slider("Hệ số Kelly Gamma", 0.05, 0.50, 0.15)
alpha_val = st.sidebar.slider("Order Rényi Entropy (Alpha)", 1.5, 3.0, 2.0)

# Session States Initialization
if "engine" not in st.session_state:
    st.session_state.engine = BQFSMEngine(
        initial_capital=initial_cap, gamma=gamma_val, alpha=alpha_val
    )
if "live_outcomes" not in st.session_state:
    st.session_state.live_outcomes = []
if "last_render_time" not in st.session_state:
    st.session_state.last_render_time = time.time()

engine = st.session_state.engine

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
current_mdd = (
    (engine.peak_capital - engine.Vt) / engine.peak_capital
    if engine.peak_capital > 0
    else 0.0
)
renyi_s = engine.calculate_renyi_entropy()

col1.metric("Số Dư Vốn (Vt)", f"{engine.Vt:,.0f} VNĐ", delta=f"MDD: {current_mdd*100:.1f}%")
col2.metric("Rényi Entropy (α=2)", f"{renyi_s:.4f}")
col3.metric("Fuzzy Vector State", f"|{' '.join(map(str, [round(b,1) for b in engine.fuzzy_vector]))}>")
col4.metric(
    "Trạng Thái Cầu Chì",
    "HOẠT ĐỘNG" if engine.fuzzy_vector[5] > 0.1 else "LOCKOUT / HALT",
)

st.divider()

# Navigation Tabs
tab_live, tab_sim = st.tabs(["🎯 Thực Chiến B-QFSME", "📊 Mô Phỏng Monte Carlo"])

# -----------------------------------------------------------------------------
# TAB 1: LIVE MODE
# -----------------------------------------------------------------------------
with tab_live:
    st.subheader("📥 Cập nhật Kỳ Quay Thực Tế")

    with st.form("live_update_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            outcome_str = st.radio("Kết quả kỳ quay:", ["Thắng (Win)", "Thua (Loss)"])
            pl_amount = st.number_input("Tiền Lãi / Lỗ (+/- VNĐ)", value=0, step=10000)

        with col_b:
            exec_err = st.checkbox("Vi phạm kỷ luật (Cảm xúc / Đặt sai số tiền)")
            st.info("💡 Hệ thống đang ghi nhận tốc độ phản xạ để phát hiện Tilt.")

        submit_btn = st.form_submit_button("🔄 Tính Toán Lệnh Kỳ Tiếp Theo (t+1)")

    if submit_btn:
        # Tính toán reaction time từ lần render trước
        now = time.time()
        reaction_time = now - st.session_state.last_render_time
        st.session_state.last_render_time = now

        delta = pl_amount if "Thắng" in outcome_str else -abs(pl_amount)
        st.session_state.live_outcomes.append(1 if "Thắng" in outcome_str else -1)

        decision = engine.process_cycle(
            last_outcome=delta,
            recent_outcomes=st.session_state.live_outcomes,
            execution_error=exec_err,
            reaction_time=reaction_time,
        )

        st.divider()
        st.markdown("### ⚡ Kết Quả Phân Tích Lệnh t+1")

        # Cảnh báo Sinh trắc
        if "LOCKOUT" in decision["bio_status"] or "WARNING" in decision["bio_status"]:
            st.warning(f"🧬 **Biometric Feedback:** {decision['bio_status']}")
        else:
            st.success(f"🧬 **Biometric Feedback:** {decision['bio_status']}")

        if decision["action"] == "CIRCUIT_BREAKER_HALT":
            st.error("🚨 **CẦU CHÌ ĐÃ KÍCH HOẠT:** HỆ THỐNG KHÓA LỆNH ĐẶT CƯỢC ĐỂ BẢO VỆ TÀI KHOẢN!")
        else:
            res1, res2, res3 = st.columns(3)
            res1.metric("Loại Cược Khuyên Dùng", decision["target_game"])
            res2.metric("Tỷ Lệ Vốn Tối Ưu (f*)", decision["f_star_pct"])
            res3.metric("Số Tiền Đặt Cược", f"{decision['bet_amount']:,.0f} VNĐ")

    # Capital Chart
    if len(engine.capital_history) > 1:
        st.markdown("#### Biểu đồ Tăng trưởng Vốn Thực chiến")
        st.line_chart(pd.DataFrame({"Vốn (VNĐ)": engine.capital_history}))

# -----------------------------------------------------------------------------
# TAB 2: MONTE CARLO SIMULATION
# -----------------------------------------------------------------------------
with tab_sim:
    st.subheader("Mô phỏng Kiểm thử Kiểm chứng B-QFSME")
    n_rounds = st.slider("Số kỳ quay mô phỏng", 50, 500, 100)

    if st.button("🚀 Chạy Mô Phỏng Monte Carlo"):
        sim_engine = BQFSMEngine(initial_capital=initial_cap, gamma=gamma_val, alpha=alpha_val)
        sim_outcomes = []

        for r in range(n_rounds):
            win = np.random.rand() < 0.1739
            sim_outcomes.append(1.0 if win else -1.0)

            # Giả lập reaction time ngẫu nhiên
            sim_rt = np.random.uniform(1.2, 5.0)
            dec = sim_engine.process_cycle(
                last_outcome=0.0,
                recent_outcomes=sim_outcomes,
                execution_error=False,
                reaction_time=sim_rt,
            )

            if dec["bet_amount"] > 0:
                sim_engine.Vt += (dec["bet_amount"] * 5) if win else -dec["bet_amount"]

        st.line_chart(pd.DataFrame({"Vốn Mô Phỏng (VNĐ)": sim_engine.capital_history}))
        st.success(f"Hoàn tất mô phỏng {n_rounds} kỳ. Vốn cuối cùng: {sim_engine.Vt:,.0f} VNĐ")
