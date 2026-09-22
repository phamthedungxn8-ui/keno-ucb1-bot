import time
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CORE AI & MATHEMATICAL ENGINES
# -----------------------------------------------------------------------------

class QFPSAIOptimizer:
    """Bộ Tối ưu hóa AI Động lực học Không gian Pha Lượng tử - Mờ (QFPS).
    Tích hợp Xuyên hầm Lượng tử và Rényi Entropy (alpha=2) để lọc nhiễu trắng.
    """
    def __init__(self, num_dim: int = 80, hbar: float = 0.1, gamma: float = 0.05, alpha: float = 2.0):
        self.D = num_dim
        self.hbar = hbar
        self.gamma = gamma
        self.alpha = alpha
        # Khởi tạo Hàm Sóng Lượng tử Phức được chuẩn hóa
        raw_psi = np.random.randn(self.D) + 1j * np.random.randn(self.D)
        self.psi = raw_psi / np.linalg.norm(raw_psi)

    def compute_renyi_entropy(self, prob_density: np.ndarray) -> float:
        """Tính Rényi Entropy (alpha=2) dạng Vector hóa."""
        p = prob_density[prob_density > 1e-12]
        sum_p_sq = np.sum(p ** self.alpha)
        if sum_p_sq <= 0:
            return 1.0
        renyi = (1.0 / (1.0 - self.alpha)) * np.log2(sum_p_sq)
        return float(np.clip(renyi / np.log2(self.D), 0.0, 1.0))

    def step_evolution(self, loss_gradient: np.ndarray, dt: float = 0.01) -> float:
        """Thực hiện 1 bước Tiến hóa Toán tử Hamilton và trả về Rényi Entropy."""
        prob_density = np.abs(self.psi) ** 2
        renyi_s = self.compute_renyi_entropy(prob_density)
        
        # Hamilton hiệu dụng: H_eff = V_w - i * gamma * S_renyi
        h_eff = loss_gradient - 1j * self.gamma * renyi_s
        
        # Tiến hóa Hàm Sóng Lượng tử
        self.psi *= np.exp(-1j * h_eff * dt / self.hbar)
        
        # Áp dụng Fuzzy Mask
        grad_norm = np.abs(loss_gradient)
        fuzzy_mask = 1.0 / (1.0 + np.exp(-10 * (grad_norm - np.mean(grad_norm))))
        self.psi *= fuzzy_mask
        
        # Chuẩn hóa lại Hàm Sóng
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
        """Tối ưu hóa tìm cặp số bằng Vectorization Ma trận 80x80."""
        T, N = history_matrix.shape
        if T < self.W:
            return (1, 2), 0.0, {}

        # 1. Trích xuất cửa sổ W kỳ gần nhất (Wx80)
        recent = history_matrix[-self.W:]
        
        # 2. Vector Tốc độ Lặp M_t (Shape: 80,)
        m_t = np.dot(self.weights, recent)
        
        # 3. Ma trận Tương quan Cặp C_t (Shape: 80x80)
        co_occurrence = np.dot(recent.T, recent)
        deg = np.diag(co_occurrence)
        norm_factor = np.sqrt(np.outer(deg, deg)) + 1e-9
        c_t = co_occurrence / norm_factor
        np.fill_diagonal(c_t, 0.0)

        # 4. Tính Score Matrix: Ma trận Điểm Cộng Hưởng
        m_matrix = np.add.outer(m_t, m_t)
        np.fill_diagonal(m_matrix, 0.0)
        
        score_matrix = (0.6 * m_matrix + 0.4 * c_t) * np.exp(-renyi_s)
        
        # 5. Lọc lấy cặp điểm cao nhất
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

    def calculate_position_size(self, renyi_s: float, reaction_time: float, odds: float = 6.0, p_win: float = 0.1739) -> tuple[float, float, str]:
        current_mdd = (self.peak_capital - self.Vt) / self.peak_capital if self.peak_capital > 0 else 0.0
        self.update_fuzzy_vector(current_mdd)
        
        # Biometric Factor
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

        # Kelly Formula thích ứng QFPS
        b = odds - 1.0
        q = 1.0 - p_win
        base_kelly = max((p_win * b - q) / b, 0.01)
        fuzzy_weight = self.fuzzy_vector[0] * self.fuzzy_vector[1] * self.fuzzy_vector[5]
        
        f_star = base_kelly * fuzzy_weight * (1.0 - renyi_s) * beta_bio * self.gamma
        f_star = float(np.clip(f_star, 0.0, 0.025))
        
        return f_star, self.Vt * f_star, status_msg


# -----------------------------------------------------------------------------
# 2. STREAMLIT USER INTERFACE
# -----------------------------------------------------------------------------

st.set_page_config(page_title="QFPS Keno Engine", page_icon="⚡", layout="wide")

# Session State Initialization
if "ai_opt" not in st.session_state:
    st.session_state.ai_opt = QFPSAIOptimizer()
if "pair_engine" not in st.session_state:
    st.session_state.pair_engine = KenoMemoryPairEngine()
if "bqfsm" not in st.session_state:
    st.session_state.bqfsm = BQFSMEngine()
if "history_matrix" not in st.session_state:
    # Khởi tạo ngẫu nhiên 10 kỳ ban đầu
    init_hist = np.zeros((10, 80))
    for t in range(10):
        init_hist[t, np.random.choice(80, 20, replace=False)] = 1
    st.session_state.history_matrix = init_hist
if "last_click_time" not in st.session_state:
    st.session_state.last_click_time = time.time()

bqfsm = st.session_state.bqfsm
ai_opt = st.session_state.ai_opt
pair_engine = st.session_state.pair_engine

st.title("⚡ QFPS Keno Engine - Hợp nhất AI Lượng tử & Quản lý Vốn")

# Top Dashboard Metrics
m1, m2, m3, m4 = st.columns(4)
current_mdd = (bqfsm.peak_capital - bqfsm.Vt) / bqfsm.peak_capital if bqfsm.peak_capital > 0 else 0.0
m1.metric("Vốn Hiện Tại (Vt)", f"{bqfsm.Vt:,.0f} VNĐ", delta=f"MDD: {current_mdd*100:.1f}%")

# Chạy 1 bước tiến hóa AI để lấy Rényi Entropy thời gian thực
fake_grad = np.sin(np.sum(st.session_state.history_matrix[-5:], axis=0) * 0.5)
renyi_s = ai_opt.step_evolution(fake_grad)

m2.metric("Rényi Entropy (α=2)", f"{renyi_s:.4f}")
m3.metric("Trạng Thái Fuzzy Vector", f"|{' '.join([str(round(b,1)) for b in bqfsm.fuzzy_vector])}>")
m4.metric("Cầu Chì An Toàn", "HOẠT ĐỘNG" if bqfsm.fuzzy_vector[5] > 0.1 else "LOCKED")

st.divider()

# Main Interactive Form
st.subheader("📥 Cập nhật Kỳ Quay Thực Tế")
with st.form("keno_update_form"):
    col_left, col_right = st.columns(2)
    with col_left:
        result_type = st.radio("Kết quả kỳ trước:", ["Thắng (Win)", "Thua (Loss)"])
        pl_val = st.number_input("Tiền Lãi/Lỗ (+/- VNĐ):", value=0, step=10000)
    with col_right:
        violation = st.checkbox("Vi phạm kỷ luật (Vào tiền sai / Tâm lý)")
        st.info("💡 Bấm nút bên dưới để AI chạy Xuyên hầm Lượng tử trích xuất cặp Bậc 2 cho kỳ tiếp theo.")

    submit = st.form_submit_button("🔄 Tiến Hóa AI & Lọc Cặp Bậc 2 (t+1)")

if submit:
    now = time.time()
    reaction_time = now - st.session_state.last_click_time
    st.session_state.last_click_time = now

    # Cập nhật số dư vốn
    delta = pl_val if "Thắng" in result_type else -abs(pl_val)
    bqfsm.Vt += delta
    bqfsm.capital_history.append(bqfsm.Vt)
    if bqfsm.Vt > bqfsm.peak_capital:
        bqfsm.peak_capital = bqfsm.Vt
    bqfsm.error_history.append(1 if violation else 0)

    # Giả lập thêm 1 kết quả kỳ mới vào History Matrix
    new_draw = np.zeros((1, 80))
    new_draw[0, np.random.choice(80, 20, replace=False)] = 1
    st.session_state.history_matrix = np.vstack([st.session_state.history_matrix, new_draw])

    # 1. Trích xuất Cặp Bậc 2
    best_pair, score, stats = pair_engine.extract_best_pair(st.session_state.history_matrix, renyi_s)

    # 2. Tính toán Quản lý vốn
    f_star, bet_amt, status_msg = bqfsm.calculate_position_size(renyi_s, reaction_time)

    # Hiển thị Kết quả
    st.markdown("### 🎯 Dự Đóa Bậc 2 & Vị Thế Cược Kỳ t+1")
    
    if "LOCKOUT" in status_msg:
        st.error(f"🚨 {status_msg}")
    elif "WARNING" in status_msg:
        st.warning(f"⚠️ {status_msg}")
    else:
        st.success(f"✅ {status_msg}")

    res1, res2, res3 = st.columns(3)
    res1.metric("Cặp Bậc 2 Tối Ưu (QFPS)", f"Số {best_pair[0]} - Số {best_pair[1]}")
    res2.metric("Điểm Cộng Hưởng Ký Nhớ", f"{score:.4f}")
    res3.metric("Số Tiền Đặt Cược (f*)", f"{bet_amt:,.0f} VNĐ ({f_star*100:.2f}%)")

    st.json(stats)

# Chart
if len(bqfsm.capital_history) > 1:
    st.line_chart(pd.DataFrame({"Vốn (VNĐ)": bqfsm.capital_history}))
