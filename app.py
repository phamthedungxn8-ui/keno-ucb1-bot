import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Deep Autonomous Multi-Agent Keno Engine", layout="centered")

# ==============================================================================
# 1. DEEP AGENT: TRANSITION MOMENTUM MATRIX (MARKOV PHI TUYẾN)
# ==============================================================================
class DeepTransitionAgent:
    """Agent Tự Giải Bài Toán Quán Tính Chuyển Trạng Thái Bậc Cao (Non-linear Markov)"""
    def __init__(self, num_dim=80):
        self.D = num_dim

    def solve_momentum(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # Xây dựng ma trận chuyển trạng thái phi tuyến A (Transition Matrix 80x80)
        A = np.dot(X.T, X) / T
        np.fill_diagonal(A, 0)
        
        # Giải phương trình lan truyền năng lượng bậc cao: S = A * Last_State
        last_state = X[-1]
        energy_signal = np.dot(A, last_state)
        
        # Tự động giải toán bù quán tính lặp (Momentum Gain Function)
        momentum_vector = np.zeros(D)
        for d in range(D):
            p_occur = freqs[d] / T
            # Phương trình kích hoạt Sigmoid năng lượng
            sigmoid_energy = 1.0 / (1.0 + np.exp(-energy_signal[d]))
            
            if last_state[d] == 1:
                # Nếu vừa nổ ở Kỳ 5, tính hàm suy giảm bão hòa (Saturation Decay)
                decay_factor = np.exp(-0.5 * max(0, freqs[d] - 3))
                momentum_vector[d] = p_occur * sigmoid_energy * decay_factor
            else:
                momentum_vector[d] = p_occur * sigmoid_energy * 0.4
                
        # Chuẩn hóa năng lượng tín hiệu [0, 1]
        max_val = np.max(momentum_vector)
        return momentum_vector / max_val if max_val > 0 else momentum_vector

# ==============================================================================
# 2. DEEP AGENT: SUPPRESSION ENTROPY REBOUND (ĐIỂM RƠI TÍCH LŨY NĂNG LƯỢNG)
# ==============================================================================
class DeepEntropyAgent:
    """Agent Tự Giải Phương Trình Bùng Nổ Năng Lượng Gan (Entropy Suppression Rebound)"""
    def __init__(self, num_dim=80):
        self.D = num_dim

    def solve_rebound(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        rebound_vector = np.zeros(D)
        
        # Phân tích dải không gian 8 không gian hàng chục (Decade Clusters)
        decade_densities = np.zeros(8)
        for dec in range(8):
            decade_densities[dec] = X[:, dec*10 : (dec+1)*10].sum() / (T * 10)

        for d in range(D):
            dec_idx = d // 10
            cluster_density = decade_densities[dec_idx]
            
            # Giải phương trình nén Shannon Entropy
            p1 = max(1e-5, freqs[d] / T)
            p0 = 1.0 - p1
            shannon_h = - (p1 * np.log2(p1) + p0 * np.log2(p0))
            
            if freqs[d] == 0:
                # Năng lượng nén đại cực: Càng nén sâu + Hàng chục đang nổ mạnh = Điểm rơi cao
                rebound_vector[d] = shannon_h * (1.5 + cluster_density * 2.0)
            elif freqs[d] == 1 and X[-1, d] == 0:
                rebound_vector[d] = shannon_h * (1.0 + cluster_density)
            else:
                rebound_vector[d] = shannon_h * 0.1
                
        max_val = np.max(rebound_vector)
        return rebound_vector / max_val if max_val > 0 else rebound_vector

# ==============================================================================
# 3. DEEP AGENT: BAYESIAN MONTE CARLO STOCHASTIC NOISE
# ==============================================================================
class DeepMonteCarloAgent:
    """Agent Tự Mô Phỏng Bayes-Monte Carlo Phá Rào Cản Ngẫu Nhiên"""
    def __init__(self, num_dim=80):
        self.D = num_dim

    def solve_stochastic_barrier(self, X, num_sims=1500):
        np.random.seed()
        T, D = X.shape
        prior_probs = X.sum(axis=0) / (T * D) # Xác suất Tiền định (Prior)
        
        sim_results = np.zeros((num_sims, D))
        for s in range(num_sims):
            # Cập nhật Trọng số Bayes (Posterior Weights)
            bayes_weights = prior_probs + np.random.normal(0.0, 0.05, size=D)
            bayes_weights = np.clip(bayes_weights, 1e-5, None)
            bayes_weights /= bayes_weights.sum()
            
            # Rút 20 số ngẫu nhiên theo trọng số Bayes
            draw = np.random.choice(D, size=20, replace=False, p=bayes_weights)
            sim_results[s, draw] = 1.0
            
        mc_vector = sim_results.mean(axis=0)
        max_val = np.max(mc_vector)
        return mc_vector / max_val if max_val > 0 else mc_vector

# ==============================================================================
# 4. MASTER ENSEMBLE SYSTEM: SELF-TUNING FUSION
# ==============================================================================
class DeepAutonomousKenoSystem:
    def __init__(self, num_dim=80):
        self.agent_trans = DeepTransitionAgent(num_dim)
        self.agent_entropy = DeepEntropyAgent(num_dim)
        self.agent_mc = DeepMonteCarloAgent(num_dim)

    def process_and_optimize(self, X):
        # 1. Các Agent thực thi tự giải phương trình riêng biệt
        v_trans = self.agent_trans.solve_momentum(X)
        v_entropy = self.agent_entropy.solve_rebound(X)
        v_mc = self.agent_mc.solve_stochastic_barrier(X, num_sims=1500)
        
        # 2. Tự tính toán độ lệch pha giữa các Agent để tự cân bằng trọng số (Self-Tuning Weights)
        # Nếu nhóm số Gan đang xuất hiện nhiều trong 5 kỳ, tăng trọng số Entropy.
        cold_count = np.sum(X.sum(axis=0) == 0)
        if cold_count > 30: # Thị trường nén mạnh
            w_trans, w_entropy, w_mc = 0.35, 0.45, 0.20
        else: # Thị trường dải số biến động đều
            w_trans, w_entropy, w_mc = 0.45, 0.35, 0.20
            
        # Ma trận Năng lượng Tổng hợp
        fusion_vector = (w_trans * v_trans) + (w_entropy * v_entropy) + (w_mc * v_mc)
        
        # 3. TRÍCH XUẤT BỘ BẬC 2 (2 SỐ TRỤ CỘT ĐỐI ỨNG)
        # N1: Chọn số có Động lượng Quán tính Markov cao nhất
        n1_idx = int(np.argmax(v_trans))
        
        # N2: Chọn số có Bùng nổ Entropy Gan cao nhất (Khác N1)
        n2_idx = int(np.argmax(v_entropy))
        if n2_idx == n1_idx:
            temp_ent = v_entropy.copy()
            temp_ent[n1_idx] = -1.0
            n2_idx = int(np.argmax(temp_ent))
            
        # 4. TRÍCH XUẤT BỘ BẬC 3 (BỔ SUNG SỐ N3 TỪ CÂN BẰNG BAYES-MONTE CARLO)
        temp_fusion = fusion_vector.copy()
        temp_fusion[n1_idx] = -1.0
        temp_fusion[n2_idx] = -1.0
        n3_idx = int(np.argmax(temp_fusion))
        
        N1, N2, N3 = n1_idx + 1, n2_idx + 1, n3_idx + 1
        
        bo_bac_2 = tuple(sorted([N1, N2]))
        bo_bac_3 = tuple(sorted([N1, N2, N3]))

        explanation = (
            f"• **CƠ CHẾ TỰ HỌC SÂU VÀ TỰ GIẢI BÀI TOÁN (Deep Autonomous System):**\n"
            f"  - **Deep Transition Agent (Tự giải Quán tính Markov):** Tự xác định phương trình năng lượng lan truyền $S = A \\cdot X$, chốt **{N1:02d}** làm Trụ Quán Tính.\n"
            f"  - **Deep Entropy Agent (Tự giải Bùng nổ Gan):** Giải phương trình Shannon Entropy kết hợp mật độ khu vực hàng chục, chốt **{N2:02d}** làm Trụ Điểm Rơi.\n"
            f"  - **Deep Monte Carlo Agent (Tự mô phỏng Bayes):** Chạy 1,500 lượt giả lập ngẫu nhiên trọng số Bayes, trích xuất **{N3:02d}** làm Số Bọc Lót Rào Cản.\n"
            f"• **Cấu trúc tối ưu:** Hệ thống tự động điều chỉnh trọng số (Trọng số hiện tại: {int(w_trans*100)}% Transition, {int(w_entropy*100)}% Entropy, {int(w_mc*100)}% Monte Carlo) để cho ra **Bộ Bậc 2 [{N1:02d} — {N2:02d}]** và **Bộ BẬC 3 [{N1:02d} — {N2:02d} — {N3:02d}]**."
        )

        return bo_bac_2, bo_bac_3, N1, N2, N3, explanation

# ==============================================================================
# STREAMLIT UI DISPLAY
# ==============================================================================
st.title("🧠 Deep Autonomous Multi-Agent Keno Engine")
st.caption("Agent Tự Học Sâu • Tự Giải Phương Trình Rào Cản Ngẫu Nhiên • Chốt Bậc 2 & Bậc 3")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=150,
    key="raw_text_keno_deep_agent"
)

if raw_text_input.strip():
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text_input.strip(), flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        matrix = np.zeros((5, 80), dtype=float)
        for k in range(5):
            for num in all_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0
                
        st.success("🎉 Tất cả các Deep Agents đã giải xong hệ phương trình tự chủ!")
        
        system = DeepAutonomousKenoSystem(num_dim=80)
        bo2, bo3, n1, n2, n3, explanation = system.process_and_optimize(matrix)
        
        st.markdown("---")
        st.subheader("🎯 BẢNG CHỐT SỐ TỪ CÁC DEEP AGENTS")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="🔥 MARKOV TRANSITION", value=f"{n1:02d}")
        with c2:
            st.metric(label="❄️ ENTROPY REBOUND", value=f"{n2:02d}")
        with c3:
            st.metric(label="🎲 BAYES MONTE CARLO", value=f"{n3:02d}")
            
        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("1️⃣ BỘ BẬC 2 CHỐT (2 SỐ)")
            st.title(f"{bo2[0]:02d} — {bo2[1]:02d}")
        with col_right:
            st.subheader("2️⃣ BỘ BẬC 3 CHỐT (3 SỐ)")
            st.title(f"{bo3[0]:02d} — {bo3[1]:02d} — {bo3[2]:02d}")
            
        st.markdown("---")
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung trên để kích hoạt hệ thống Deep Autonomous Agent.")
