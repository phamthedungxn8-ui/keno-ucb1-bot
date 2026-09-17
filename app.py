import numpy as np
import pandas as pd


class BCSMKenoEngine:
    """BCSM (Biometric-Contextual State Mapping) Engine for Keno Capital Management.

    Integrates 3-Axis Architecture:
    - Axis 1: Capital State Vector 6-D, Shannon Entropy & Dynamic Position
    Sizing (Kelly-Entropy).
    - Axis 2: Payoff Matrix & Minimax Variance Minimization.
    - Axis 3: Markov Chain Regime Switching Detection.
    """

    def __init__(self, initial_capital: float = 10000000.0, gamma: float = 0.15):
        """Khởi tạo Engine với Vốn ban đầu V0 và Hệ số an toàn Gamma (Fractional

        Kelly).
        """
        self.V0 = initial_capital
        self.Vt = initial_capital
        self.gamma = gamma

        # Vector Trạng thái 6-Chiều: |b1 b2 b3 b4 b5 b6>
        # b1: Capital Reserve Rate (Vt >= 0.8 * V0)
        # b2: Maximum Drawdown Boundary (MDD <= 15%)
        # b3: Execution Discipline (1: Tuân thủ, 0: Lỗi cược)
        # b4: Regime Exposure (1: <= 2% Position, 0: Clear Position)
        # b5: Profit Lock-in / Stop-loss Check
        # b6: Circuit Breaker Adaptability (1: Active, 0: HALT)
        self.bit_vector = [1, 1, 1, 1, 1, 1]
        self.capital_history = [initial_capital]
        self.peak_capital = initial_capital

    def calculate_shannon_entropy(self, window_size: int = 10) -> float:
        """Trục 1: Tính toán Shannon Entropy dòng vốn (S_capital) trên cửa sổ mẫu N

        kỳ gần nhất.

        Entropy S -> 0: Dòng vốn ổn định. Entropy S -> 1: Dòng vốn hỗn loạn.
        """
        if len(self.capital_history) <= window_size:
            return 0.0

        recent_history = self.capital_history[-window_size:]
        returns = np.diff(recent_history) / recent_history[:-1]

        if np.all(returns == 0):
            return 0.0

        # Chia phân phối tỷ suất lợi nhuận vào 5 nhóm (bins)
        hist, _ = np.histogram(returns, bins=5)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]  # Lọc xác suất bằng 0 để tránh log2(0)

        entropy = -np.sum(probs * np.log2(probs))
        # Standardize Entropy về đoạn [0.0, 1.0]
        max_entropy = np.log2(5)
        normalized_entropy = min(max(entropy / max_entropy, 0.0), 1.0)
        return float(normalized_entropy)

    def detect_markov_regime(self, recent_outcomes: list) -> str:
        """Trục 3: Mô hình hóa Chuỗi Markov về Trạng thái Không gian (Regime

        Switching).

        - S1: Over-represented (Lệch Dương)
        - S2: Under-represented (Lệch Âm)
        - S3: Equilibrium (Cân bằng / Hòa)
        """
        if not recent_outcomes or len(recent_outcomes) < 5:
            return 'S3'  # Mặc định Cân bằng nếu thiếu dữ liệu

        positive_count = sum(1 for x in recent_outcomes[-10:] if x > 0)
        ratio = positive_count / len(recent_outcomes[-10:])

        if ratio >= 0.65:
            return 'S1'  # Trend thắng ngắn hạn
        elif ratio <= 0.35:
            return 'S2'  # Trend thua ngắn hạn
        else:
            return 'S3'  # Cân bằng

    def select_payoff_target(self, regime: str) -> dict:
        """Trục 2: Tối ưu Payoff Matrix & Minimax Variance dựa trên Trạng thái

        Regime.
        """
        if regime == 'S2' or self.bit_vector[0] == 0:
            # Vùng S1 (Defensive): Keno Bậc 2 hoặc Chẵn/Lẻ (Variance cực thấp để bảo vệ vốn)
            return {'type': 'Keno_Level_2', 'p_win': 0.1739, 'odds': 6.0}
        elif regime == 'S1' and self.bit_vector[4] == 1:
            # Vùng S2 (Aggressive): Keno Bậc 4 (Tận dụng Profit Buffer khi b5=1)
            return {'type': 'Keno_Level_4', 'p_win': 0.0264, 'odds': 100.0}
        else:
            # Vùng Tiêu chuẩn
            return {'type': 'Keno_Level_2', 'p_win': 0.1739, 'odds': 6.0}

    def update_state_vector(self, execution_error: bool = False) -> bool:
        """Cập nhật Vector Trạng thái 6-D & Kiểm tra Cảnh báo Điểm gẫy (Bifurcation

        Warning).

        Return: True nếu kích hoạt CIRCUIT BREAKER, ngược lại False.
        """
        # Cập nhật Peak Capital & Current MDD
        if self.Vt > self.peak_capital:
            self.peak_capital = self.Vt
        current_mdd = (self.peak_capital - self.Vt) / self.peak_capital

        # Bit b1: Capital Reserve Rate (Vt >= 80% V0)
        self.bit_vector[0] = 1 if self.Vt >= 0.8 * self.V0 else 0

        # Bit b2: Maximum Drawdown Boundary (MDD <= 15%)
        self.bit_vector[1] = 1 if current_mdd <= 0.15 else 0

        # Bit b3: Execution Discipline
        self.bit_vector[2] = 0 if execution_error else 1

        # Bit b5: Profit Lock-in / Stop Loss Check
        # Khóa nếu sụt giảm quá 30% từ đỉnh hoặc mất sạch lãi
        self.bit_vector[4] = 1 if current_mdd <= 0.20 else 0

        # CẢNH BÁO ĐIỂM GẪY: Bifurcation Risk = ~b2 AND ~b3 (Mất kỷ luật VÀ Vốn giảm sâu)
        bifurcation_risk = (self.bit_vector[1] == 0) and (
            self.bit_vector[2] == 0
        )

        if bifurcation_risk or current_mdd >= 0.25:
            # KÍCH HOẠT CẦU CHÌ HỆ THỐNG
            self.bit_vector[3] = 0  # b4 -> 0: Thu vị thế cược về 0
            self.bit_vector[5] = 0  # b6 -> 0: CIRCUIT BREAKER HALT
            return True

        self.bit_vector[3] = 1
        self.bit_vector[5] = 1
        return False

    def compute_position_size(self, p: float, odds: float, S_cap: float) -> float:
        """Tính toán Quy mô Vị thế Tối ưu f* (Kelly - Entropy Adjusted).

        f* = [(p * b - q) / b] * (1 - S_capital) * gamma
        """
        # Nếu Circuit Breaker ngắt (b6 = 0) hoặc Bit b4 = 0 -> Khóa cược
        if self.bit_vector[5] == 0 or self.bit_vector[3] == 0:
            return 0.0

        b = odds - 1.0  # Net odds
        q = 1.0 - p

        kelly_f = (p * b - q) / b

        # Nếu Kelly ra kết quả âm (EV < 0 tiêu chuẩn), thiết lập baseline tối thiểu an toàn để duy trì chuỗi
        if kelly_f <= 0:
            kelly_f = 0.01  # Baseline Risk Factor

        # Công thức BCSM: Chiết khấu vị thế theo Shannon Entropy dòng vốn
        f_star = kelly_f * (1.0 - S_cap) * self.gamma

        # Giới hạn cứng: Không cược quá 2% vốn/kỳ
        return float(min(max(f_star, 0.0), 0.02))

    def process_cycle(
        self,
        last_outcome: float = 0.0,
        recent_history_outcomes: list = None,
        execution_error: bool = False,
    ) -> dict:
        """Quy trình Vận hành Chuẩn (SOP) cho mỗi kỳ quay t+1."""
        # 1. Cập nhật Số dư dòng vốn từ kết quả kỳ t
        self.Vt += last_outcome
        self.capital_history.append(self.Vt)

        # 2. Cập nhật Trạng thái Vector Bit & Kiểm tra Cầu chì
        is_halted = self.update_state_vector(execution_error)
        if is_halted:
            return {
                'action': 'CIRCUIT_BREAKER_HALT',
                'state_vector': self.bit_vector,
                'bet_amount': 0.0,
                'f_star': 0.0,
                'status': 'Dừng giao dịch khẩn cấp để bảo vệ vốn.',
            }

        # 3. Trục 1: Tính Shannon Entropy
        S_cap = self.calculate_shannon_entropy()

        # 4. Trục 3: Đo lường Chuỗi Markov
        regime = self.detect_markov_regime(recent_history_outcomes or [])

        # 5. Trục 2: Chọn Vùng cược
        payoff = self.select_payoff_target(regime)

        # 6. Tính toán Size Vị thế cược
        f_star = self.compute_position_size(
            payoff['p_win'], payoff['odds'], S_cap
        )
        bet_amount = self.Vt * f_star

        return {
            'action': 'EXECUTE_BET',
            'state_vector': f'|{" ".join(map(str, self.bit_vector))}>',
            'regime': regime,
            'target_game': payoff['type'],
            'shannon_entropy': round(S_cap, 4),
            'f_star_pct': f'{round(f_star * 100, 2)}%',
            'bet_amount': round(bet_amount, 2),
            'current_capital': round(self.Vt, 2),
        }


# =====================================================================
# BỘ MÔ PHỎNG MONTE CARLO KIỂM CHỨNG TỶ LỆ TỒN TẠI (SURVIVAL RATE)
# =====================================================================


def run_monte_carlo_validation(n_simulations: int = 500, n_rounds: int = 1000):
    """Chạy mô phỏng Monte Carlo so sánh Strategy A (Fixed 2%) vs Strategy B

    (BCSM Engine).
    """
    initial_cap = 10000000.0
    p_win_k2 = 0.1739
    odds_k2 = 6.0

    survived_fixed = 0
    survived_bcsm = 0

    print(
        f'=== KÍCH HOẠT MÔ PHỎNG MONTE CARLO ({n_simulations} Simulations x'
        f' {n_rounds} Rounds) ===\n'
    )

    for sim in range(n_simulations):
        cap_a = initial_cap
        engine_b = BCSMKenoEngine(initial_capital=initial_cap)

        outcomes_history = []

        for r in range(n_rounds):
            win = np.random.rand() < p_win_k2
            outcome_val = 1.0 if win else -1.0
            outcomes_history.append(outcome_val)

            # --- STRATEGY A (Fixed 2% Allocation) ---
            if cap_a > 0:
                bet_a = cap_a * 0.02
                cap_a = cap_a + (bet_a * (odds_k2 - 1)) if win else cap_a - bet_a

            # --- STRATEGY B (BCSM Engine) ---
            if engine_b.Vt > 0:
                # Lấy quyết định từ Engine
                decision = engine_b.process_cycle(
                    last_outcome=0.0, recent_history_outcomes=outcomes_history
                )
                bet_b = decision['bet_amount']

                if bet_b > 0:
                    delta_capital = (
                        (bet_b * (odds_k2 - 1)) if win else -bet_b
                    )
                    engine_b.Vt += delta_capital
                    engine_b.capital_history.append(engine_b.Vt)

        # Tiêu chí Tồn tại: Giữ được ít nhất 20% vốn ban đầu sau 1,000 kỳ
        if cap_a >= initial_cap * 0.20:
            survived_fixed += 1
        if engine_b.Vt >= initial_cap * 0.20:
            survived_bcsm += 1

    rate_a = (survived_fixed / n_simulations) * 100
    rate_b = (survived_bcsm / n_simulations) * 100

    print('=== KẾT QUẢ MÔ PHỎNG MONTE CARLO ===')
    print(f'Strategy A (Fixed 2% Capital Allocation) Survival Rate : {rate_a:.2f}%')
    print(f'Strategy B (BCSM Engine Dynamic Sizing) Survival Rate  : {rate_b:.2f}%')
    print('=====================================================\n')


# Execute Test Run
if __name__ == '__main__':
    # 1. Chạy thử nghiệm 1 chu kỳ vận hành Engine
    engine = BCSMKenoEngine(initial_capital=10000000.0)
    sample_decision = engine.process_cycle(
        last_outcome=0.0, recent_history_outcomes=[1, -1, -1, 1, -1]
    )
    print('Quyết định Mẫu từ BCSM Engine:')
    for k, v in sample_decision.items():
        print(f'  {k}: {v}')
    print('\n' + '=' * 50 + '\n')

    # 2. Chạy Mô phỏng Monte Carlo
    run_monte_carlo_validation(n_simulations=200, n_rounds=500)
