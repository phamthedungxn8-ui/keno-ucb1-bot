import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BCSM Keno System Engine", page_icon="⚡", layout="wide"
)


# -----------------------------------------------------------------------------
# CORE ENGINE CLASS
# -----------------------------------------------------------------------------
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
        returns = np.diff(recent_history) / recent_history[:-1]

        if np.all(returns == 0):
            return 0.0

        hist, _ = np.histogram(returns, bins=5)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]

        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(5)
        return float(min(max(entropy / max_entropy, 0.0), 1.0))

    def detect_markov_regime(self, recent_outcomes: list) -> str:
        if not recent_outcomes or len(recent_outcomes) < 5:
            return "S3"
        positive_count = sum(1 for x in recent_outcomes[-10:] if x > 0)
        ratio = positive_count / len(recent_outcomes[-10:])

        if ratio >= 0.65:
            return "S1"
        elif ratio <= 0.35:
            return "S2"
        else:
            return "S3"

    def select_payoff_target(self, regime: str) -> dict:
        if regime == "S2" or self.bit_vector[0] == 0:
            return {"type": "Keno_Level_2", "p_win": 0.1739, "odds": 6.0}
        elif regime == "S1" and self.bit_vector[4] == 1:
            return {"type": "Keno_Level_4", "p_win": 0.0264, "odds": 100.0}
        else:
            return {"type": "Keno_Level_2", "p_win": 0.1739, "odds": 6.0}

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


# -----------------------------------------------------------------------------
# STREAMLIT UI INTERFACE
# -----------------------------------------------------------------------------
st.title("⚡ BCSM Keno System Physics Engine")
st.markdown(
    "**Khung Quản trị Trạng thái Vốn & Entropy Dòng tiền** dựa trên Hệ thống Phức hợp."
)

# Sidebar Control Panel
st.sidebar.header("⚙️ Tham số Đầu vào (Input)")
initial_cap = st.sidebar.number_input(
    "Vốn Ban Đầu (V0)", value=10000000, step=1000000
)
gamma_val = st.sidebar.slider("Hệ số An toàn Gamma (Kelly)", 0.05, 0.50, 0.15)
n_sim_rounds = st.sidebar.slider("Số kỳ mô phỏng (Rounds)", 50, 500, 100)

run_sim_button = st.sidebar.button("🚀 Kích hoạt Mô phỏng")

# Session State Initialization
if "engine" not in st.session_state:
    st.session_state.engine = BCSMKenoEngine(
        initial_capital=initial_cap, gamma=gamma_val
    )

# Real-time Metrics Dashboard
col1, col2, col3, col4 = st.columns(4)
current_cap = st.session_state.engine.Vt
entropy_val = st.session_state.engine.calculate_shannon_entropy()

col1.metric("Vốn Hiện Tại (Vt)", f"{current_cap:,.0f} VNĐ")
col2.metric("Shannon Entropy (S)", f"{entropy_val:.4f}")
col3.metric(
    "State Vector",
    f'|{" ".join(map(str, st.session_state.engine.bit_vector))}>',
)
col4.metric(
    "Trạng thái Hệ thống",
    (
        "RUNNING"
        if st.session_state.engine.bit_vector[5] == 1
        else "HALTED (Cầu chì)"
    ),
)

st.divider()

# Simulation Execution Block
if run_sim_button:
    st.subheader("📊 Kết quả Mô phỏng Chuỗi Kỳ quay Real-time")

    engine = BCSMKenoEngine(initial_capital=initial_cap, gamma=gamma_val)
    outcomes = []
    p_win = 0.1739
    odds = 6.0

    progress_bar = st.progress(0)

    for r in range(n_sim_rounds):
        win = np.random.rand() < p_win
        outcome_val = 1.0 if win else -1.0
        outcomes.append(outcome_val)

        decision = engine.process_cycle(
            last_outcome=0.0, recent_history_outcomes=outcomes
        )
        bet_b = decision["bet_amount"]

        if bet_b > 0:
            delta = (bet_b * (odds - 1)) if win else -bet_b
            engine.Vt += delta

        progress_bar.progress((r + 1) / n_sim_rounds)

    # Plot Capital Growth Chart
    df_chart = pd.DataFrame(
        {"Kỳ quay": range(len(engine.capital_history)), "Vốn (VNĐ)": engine.capital_history}
    )
    st.line_chart(df_chart, x="Kỳ quay", y="Vốn (VNĐ)")

    st.success(f"Mô phỏng hoàn tất! Vốn cuối cùng: {engine.Vt:,.0f} VNĐ")
