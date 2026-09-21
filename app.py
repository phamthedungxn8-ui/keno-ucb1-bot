import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BCSM Keno System Engine", page_icon="⚡", layout="wide"
)


# --- CORE ENGINE CLASS ---
class BCSMKenoEngine:

    def __init__(self, initial_capital: float = 10000000.0, gamma: float = 0.15):
        self.V0 = initial_capital
        self.Vt = initial_capital
        self.gamma = gamma
        self.bit_vector = [1, 1, 1, 1, 1, 1]
        self.capital_history = [initial_capital]
        self.peak_capital = initial_capital

    def calculate_shannon_entropy(self, window_size: int = 10) -> float:
        if len(self.capital_history) <= window_size:
            return 0.0
        recent_history = self.capital_history[-window_size:]
        returns = np.diff(recent_history) / (recent_history[:-1] + 1e-9)
        if np.all(returns == 0):
            return 0.0
        hist, _ = np.histogram(returns, bins=5)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return float(min(max(entropy / np.log2(5), 0.0), 1.0))

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
        if regime == "S2" or self.bit_vector[0] == 0:
            return {"type": "Keno Bậc 2 / Chẵn Lẻ", "p_win": 0.1739, "odds": 6.0}
        elif regime == "S1" and self.bit_vector[4] == 1:
            return {"type": "Keno Bậc 4", "p_win": 0.0264, "odds": 100.0}
        return {"type": "Keno Bậc 2", "p_win": 0.1739, "odds": 6.0}

    def update_state_vector(self, execution_error: bool = False) -> bool:
        if self.Vt > self.peak_capital:
            self.peak_capital = self.Vt
        current_mdd = (
            (self.peak_capital - self.Vt) / self.peak_capital
            if self.peak_capital > 0
            else 0
        )

        self.bit_vector[0] = 1 if self.Vt >= 0.8 * self.V0 else 0
        self.bit_vector[1] = 1 if current_mdd <= 0.15 else 0
        self.bit_vector[2] = 0 if execution_error else 1
        self.bit_vector[4] = 1 if current_mdd <= 0.20 else 0

        bifurcation_risk = (self.bit_vector[1] == 0) and (
            self.bit_vector[2] == 0
        )
        if bifurcation_risk or current_mdd >= 0.25:
            self.bit_vector[3] = 0
            self.bit_vector[5] = 0
            return True

        self.bit_vector[3] = 1
        self.bit_vector[5] = 1
        return False

    def compute_position_size(
        self, p: float, odds: float, S_cap: float
    ) -> float:
        if self.bit_vector[5] == 0 or self.bit_vector[3] == 0:
            return 0.0
        b = odds - 1.0
        q = 1.0 - p
        kelly_f = (p * b - q) / b
        if kelly_f <= 0:
            kelly_f = 0.01
        f_star = kelly_f * (1.0 - S_cap) * self.gamma
        return float(min(max(f_star, 0.0), 0.02))

    def process_cycle(
        self,
        last_outcome: float = 0.0,
        recent_history_outcomes: list = None,
        execution_error: bool = False,
    ) -> dict:
        self.Vt += last_outcome
        self.capital_history.append(self.Vt)

        is_halted = self.update_state_vector(execution_error)
        if is_halted:
            return {
                "action": "CIRCUIT_BREAKER_HALT",
                "state_vector": "|0 0 0 0 0 0>",
                "bet_amount": 0.0,
                "f_star_pct": "0.0%",
                "shannon_entropy": 1.0,
                "regime": "CRITICAL",
                "target_game": "NONE",
            }

        S_cap = self.calculate_shannon_entropy()
        regime = self.detect_markov_regime(recent_history_outcomes or [])
        payoff = self.select_payoff_target(regime)
        f_star = self.compute_position_size(
            payoff["p_win"], payoff["odds"], S_cap
        )
        bet_amount = self.Vt * f_star

        return {
            "action": "EXECUTE_BET",
            "state_vector": f'|{" ".join(map(str, self.bit_vector))}>',
            "regime": regime,
            "target_game": payoff["type"],
            "shannon_entropy": round(S_cap, 4),
            "f_star_pct": f"{round(f_star * 100, 2)}%",
            "bet_amount": round(bet_amount, 2),
            "current_capital": round(self.Vt, 2),
        }


# --- UI INTERFACE ---
st.title("⚡ BCSM Keno System Engine")

# Sidebar
st.sidebar.header("⚙️ Thiết lập Hệ thống")
initial_cap = st.sidebar.number_input(
    "Vốn Ban Đầu V0 (VNĐ)", value=10000000, step=1000000
)
gamma_val = st.sidebar.slider("Hệ số An toàn Gamma", 0.05, 0.50, 0.15)

if "live_engine" not in st.session_state:
    st.session_state.live_engine = BCSMKenoEngine(
        initial_capital=initial_cap, gamma=gamma_val
    )
if "live_history" not in st.session_state:
    st.session_state.live_history = []

# Tabs Navigation
tab1, tab2 = st.tabs(["🎯 Chế độ Thực Chiến (Live Mode)", "📊 Chế độ Mô Phỏng (Monte Carlo)"])

# TAB 1: LIVE MODE
with tab1:
    st.subheader("Trợ lý Quản trị Vốn Thời gian thực")

    engine = st.session_state.live_engine

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vốn Hiện Tại", f"{engine.Vt:,.0f} VNĐ")
    m2.metric("Shannon Entropy", f"{engine.calculate_shannon_entropy():.4f}")
    m3.metric("State Vector", f'|{" ".join(map(str, engine.bit_vector))}>')
    m4.metric(
        "Trạng thái",
        "HOẠT ĐỘNG" if engine.bit_vector[5] == 1 else "CẦU CHÌ NGẮT",
    )

    st.divider()

    # Form nhập kết quả kỳ vừa xong
    st.markdown("### 📥 Cập nhật Kết quả Kỳ quay vừa qua")
    col_a, col_b = st.columns(2)

    with col_a:
        outcome_type = st.radio("Kết quả kỳ trước:", ["Thắng (Win)", "Thua (Loss)"])
        profit_loss = st.number_input("Số tiền Lãi / Lỗ (+/- VNĐ)", value=0, step=10000)

    with col_b:
        exec_error = st.checkbox("Vi phạm kỷ luật (Đặt sai tiền / Cảm xúc)")
        submit_btn = st.button("🔄 Tính toán Kỳ Tiếp Theo (t+1)")

    if submit_btn:
        actual_delta = (
            profit_loss
            if "Thắng" in outcome_type
            else -abs(profit_loss)
        )
        st.session_state.live_history.append(1 if "Thắng" in outcome_type else -1)

        decision = engine.process_cycle(
            last_outcome=actual_delta,
            recent_history_outcomes=st.session_state.live_history,
            execution_error=exec_error,
        )

        st.divider()
        st.markdown("### ⚡ Lệnh Đặt Cược Kỳ Tiếp Theo (t+1)")

        if decision["action"] == "CIRCUIT_BREAKER_HALT":
            st.error("🚨 CẦU CHÌ ĐÃ KÍCH HOẠT: Dừng đặt cược ngay lập tức để bảo vệ vốn!")
        else:
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Loại Cược Khuyên Dùng", decision["target_game"])
            res_col2.metric("Tỷ lệ Vốn (% f*)", decision["f_star_pct"])
            res_col3.metric("Số Tiền Cược Tối Ưu", f"{decision['bet_amount']:,.0f} VNĐ")

    # Chart
    if len(engine.capital_history) > 1:
        st.line_chart(pd.DataFrame({"Số dư Vốn": engine.capital_history}))

# TAB 2: MONTE CARLO SIMULATION
with tab2:
    st.subheader("Mô phỏng Kiểm chứng Chiến lược")
    n_sim_rounds = st.slider("Số kỳ mô phỏng", 50, 500, 100)
    if st.button("🚀 Kích hoạt Mô phỏng Monte Carlo"):
        sim_engine = BCSMKenoEngine(initial_capital=initial_cap, gamma=gamma_val)
        outcomes = []
        for r in range(n_sim_rounds):
            win = np.random.rand() < 0.1739
            outcomes.append(1.0 if win else -1.0)
            dec = sim_engine.process_cycle(
                last_outcome=0.0, recent_history_outcomes=outcomes
            )
            if dec["bet_amount"] > 0:
                sim_engine.Vt += (dec["bet_amount"] * 5) if win else -dec["bet_amount"]

        st.line_chart(pd.DataFrame({"Vốn Mô Phỏng": sim_engine.capital_history}))
        st.success(f"Kết quả sau {n_sim_rounds} kỳ: {sim_engine.Vt:,.0f} VNĐ")
