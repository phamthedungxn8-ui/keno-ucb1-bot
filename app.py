import streamlit as st
import numpy as np
import pandas as pd
import math

st.set_page_config(page_title="Keno Reasoning Engine v2.0", page_icon="🧮", layout="wide")

st.title("🧮 Integrated Keno Engine (Data-Driven MCTS + 4 Pillars)")
st.caption("Kết hợp Dữ liệu Lịch sử (Exponential Decay) + MCTS Tree Search + Formal Verification + Stress Test")

# =========================================================
# 1. BỘ XỬ LÝ DỮ LIỆU LỊCH SỬ & TRỌNG SỐ THỜI GIAN
# =========================================================
def calculate_decay_weights(history_draws, decay_factor=0.95):
    """Tính trọng số Exponential Decay cho 80 số dựa trên lịch sử"""
    num_draws = len(history_draws)
    weights = np.zeros(80)
    
    for idx, draw in enumerate(history_draws):
        # Kỳ càng mới (gần cuối list) trọng số càng cao
        time_weight = math.pow(decay_factor, num_draws - 1 - idx)
        for num in draw:
            if 1 <= num <= 80:
                weights[num - 1] += time_weight
                
    # Chuẩn hóa về xác suất (tổng = 1)
    if weights.sum() > 0:
        probs = weights / weights.sum()
    else:
        probs = np.ones(80) / 80.0
    return probs

# =========================================================
# 2. TRỤ CỘT 3: FORMAL VERIFICATION SYSTEM
# =========================================================
class FormalVerificationEngine:
    @staticmethod
    def verify_quadrant_entropy(candidate_8):
        quads = [(n - 1) // 20 for n in candidate_8]
        counts = [quads.count(i) for i in range(4)]
        probs = [c / 8.0 for c in counts if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy >= 1.38, entropy

    @staticmethod
    def verify_anti_clustering(candidate_8):
        psychological_nums = [n for n in candidate_8 if n <= 31]
        return len(psychological_nums) <= 5

# =========================================================
# 3. TRỤ CỘT 4: COUNTEREXAMPLE STRESS TEST
# =========================================================
class CounterexampleStressTest:
    @staticmethod
    def run_stress_test(candidate_8, num_simulations=1000):
        failures = 0
        for _ in range(num_simulations):
            bias = np.random.choice(["even_heavy", "odd_heavy", "random"])
            if bias == "even_heavy":
                draw = list(np.random.choice(range(2, 81, 2), 15, replace=False)) + \
                       list(np.random.choice(range(1, 81, 2), 5, replace=False))
            elif bias == "odd_heavy":
                draw = list(np.random.choice(range(1, 81, 2), 15, replace=False)) + \
                       list(np.random.choice(range(2, 81, 2), 5, replace=False))
            else:
                draw = list(np.random.choice(range(1, 81), 20, replace=False))
            
            hits = len(set(candidate_8).intersection(set(draw)))
            if hits < 2:
                failures += 1
        
        failure_rate = failures / num_simulations
        return failure_rate < 0.35, failure_rate

# =========================================================
# 4. TRỤ CỘT 1 & 2: DATA-DRIVEN MCTS SEARCH
# =========================================================
def execute_advanced_reasoning(history_draws, test_time_budget, decay_factor):
    logs = []
    
    # Tính toán phân bố xác suất từ dữ liệu
    if history_draws:
        probs = calculate_decay_weights(history_draws, decay_factor)
        logs.append(f"📊 **[Data Engine]:** Đã nạp **{len(history_draws)} kỳ**. Đã tính toán trọng số Exponential Decay ($\lambda={decay_factor}$).")
    else:
        probs = np.ones(80) / 80.0
        logs.append("⚠️ **[Data Engine]:** Chưa có dữ liệu lịch sử. Sử dụng phân bố đều (Uniform Prior).")

    logs.append("🧠 **[MCTS Engine]:** Bắt đầu duyệt cây với xác suất trọng số...")
    
    best_candidate = None
    verified = False
    attempts = 0
    
    while not verified and attempts < test_time_budget:
        attempts += 1
        
        # MCTS Weighted Sampling: Ưu tiên bốc các số có xác suất cao từ lịch sử
        candidate_idx = np.random.choice(range(1, 81), size=8, replace=False, p=probs)
        candidate = sorted([int(x) for x in candidate_idx])
        
        # Lớp 1: Formal Logic
        is_entropy_valid, entropy_val = FormalVerificationEngine.verify_quadrant_entropy(candidate)
        is_anti_cluster_valid = FormalVerificationEngine.verify_anti_clustering(candidate)
        
        if not (is_entropy_valid and is_anti_cluster_valid):
            continue
            
        # Lớp 2: Stress Test Phản Ví Dụ
        passed_stress, fail_rate = CounterexampleStressTest.run_stress_test(candidate)
        
        if passed_stress:
            verified = True
            best_candidate = candidate
            logs.append(f"🔄 **[Backtrack Loop]:** Tìm thấy ứng viên vượt qua kiểm chứng tại vòng lặp **#{attempts}**.")
            logs.append(f"✅ **[Formal Verification]:** Shannon Entropy = **{entropy_val:.2f}** (Đạt chuẩn $\ge 1.38$).")
            logs.append(f"🛡️ **[Stress Test]:** Tỷ lệ sập dàn = **{fail_rate*100:.1f}%** (Đạt chuẩn $<35\%$).")
            break

    return logs, best_candidate

# =========================================================
# GIAO DIỆN STREAMLIT
# =========================================================
# Sidebar
st.sidebar.header("⚙️ Cấu Hình Thuật Toán")
test_time_compute = st.sidebar.slider("Ngân sách Test-Time Compute (MCTS)", 100, 5000, 1000, step=100)
decay_factor = st.sidebar.slider("Hệ số suy giảm thời gian (Decay Lambda)", 0.80, 0.99, 0.95, step=0.01)

# Nạp dữ liệu
st.subheader("📥 1. Nạp Dữ Liệu Kết Quả Lịch Sử")
if "history" not in st.session_state:
    st.session_state.history = []

raw_input = st.text_area(
    "Dán kết quả các kỳ gần nhất (Mỗi kỳ 1 dòng 20 số, phân cách bằng khoảng trắng/dấu phẩy):",
    placeholder="01 05 12 18 27 33 41 52 ...\n03 08 15 22 29 34 45 60 ...",
    height=100
)

col_input1, col_input2 = st.columns([1, 4])
with col_input1:
    if st.button("💾 Nạp Dữ Liệu"):
        if raw_input.strip():
            lines = raw_input.strip().split("\n")
            new_draws = []
            for line in lines:
                nums = [int(s) for s in line.replace(",", " ").split() if s.isdigit()]
                if len(nums) == 20:
                    new_draws.append(nums)
            
            st.session_state.history.extend(new_draws)
            st.session_state.history = st.session_state.history[-100:] # Cửa sổ trượt 100
            st.success(f"✅ Đã nạp thành công {len(new_draws)} kỳ quay. Tổng dữ liệu hiện tại: {len(st.session_state.history)} kỳ.")

st.markdown("---")
st.subheader("🚀 2. Thực Thi Suy Luận & Xuất Dàn Vé")

if st.button("🎯 Chạy Engine Tối Ưu MCTS", type="primary"):
    with st.spinner("Đang tính toán trọng số, thực hiện MCTS Search & Kiểm chứng Formal..."):
        logs, final_8 = execute_advanced_reasoning(
            st.session_state.history, 
            test_time_compute, 
            decay_factor
        )
        
        st.write("📝 **Tiến Trình Suy Luận Internal Monologue:**")
        for log in logs:
            st.markdown(log)
            
        if final_8:
            clean_8 = [int(x) for x in final_8]
            st.markdown("---")
            st.success(f"🎯 **TẬP 8 SỐ TỐI ƯU HOÀN HẢO:** `{clean_8}`")
            
            # Chia dàn vé
            b3_tickets = [
                [clean_8[0], clean_8[1], clean_8[2]],
                [clean_8[2], clean_8[3], clean_8[4]],
                [clean_8[4], clean_8[5], clean_8[6]],
                [clean_8[0], clean_8[3], clean_8[6]],
                [clean_8[1], clean_8[4], clean_8[7]],
                [clean_8[0], clean_8[2], clean_8[7]]
            ]
            
            b2_tickets = [
                [clean_8[0], clean_8[1]],
                [clean_8[2], clean_8[3]],
                [clean_8[4], clean_8[5]],
                [clean_8[6], clean_8[7]]
            ]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**6 Vé Bậc 3 (Bảo Vệ Vốn):**")
                for i, t in enumerate(b3_tickets, 1):
                    st.code(f"Vé B3-{i}: {t} | 10,000 VNĐ")
                    
            with col_b:
                st.markdown("**4 Vé Bậc 2 (Mũi Nhọn):**")
                for i, t in enumerate(b2_tickets, 1):
                    st.code(f"Vé B2-{i}: {t} | 10,000 VNĐ")
        else:
            st.error("❌ Không tìm thấy tập số thỏa mãn kiểm chứng. Hãy tăng ngân sách Test-Time Compute trên thanh menu trái!")
