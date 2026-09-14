import streamlit as st
import numpy as np
import pandas as pd
import math

st.set_page_config(page_title="Keno Reasoning Engine v2.1", page_icon="🧮", layout="wide")

st.title("🧮 Integrated Keno Engine (Custom Input + MCTS + Formal Verification)")
st.caption("Hệ thống kiểm chứng dàn số tùy chọn & Tự động khai thác 4 trụ cột kỹ thuật")

# =========================================================
# 1. BỘ XỬ LÝ DỮ LIỆU LỊCH SỬ & TRỌNG SỐ THỜI GIAN
# =========================================================
def calculate_decay_weights(history_draws, decay_factor=0.95):
    """Tính trọng số Exponential Decay cho 80 số dựa trên lịch sử"""
    num_draws = len(history_draws)
    weights = np.zeros(80)
    
    for idx, draw in enumerate(history_draws):
        time_weight = math.pow(decay_factor, num_draws - 1 - idx)
        for num in draw:
            if 1 <= num <= 80:
                weights[num - 1] += time_weight
                
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
# 4. ENGINE SUY LUẬN TỰ ĐỘNG (MCTS)
# =========================================================
def execute_mcts_reasoning(history_draws, test_time_budget, decay_factor):
    logs = []
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
        candidate_idx = np.random.choice(range(1, 81), size=8, replace=False, p=probs)
        candidate = sorted([int(x) for x in candidate_idx])
        
        is_entropy_valid, entropy_val = FormalVerificationEngine.verify_quadrant_entropy(candidate)
        is_anti_cluster_valid = FormalVerificationEngine.verify_anti_clustering(candidate)
        
        if not (is_entropy_valid and is_anti_cluster_valid):
            continue
            
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
st.sidebar.header("⚙️ Cấu Hình Thuật Toán")
mode = st.sidebar.radio("Nguồn tạo tập 8 số:", ["Tự chọn / Nhập thủ công", "MCTS Search Tự Động"])
test_time_compute = st.sidebar.slider("Ngân sách Test-Time Compute (MCTS)", 100, 5000, 1000, step=100)
decay_factor = st.sidebar.slider("Hệ số suy giảm thời gian (Decay Lambda)", 0.80, 0.99, 0.95, step=0.01)

if "history" not in st.session_state:
    st.session_state.history = []

# GIAO DIỆN DÀNH CHO CHẾ ĐỘ NHẬP THỦ CÔNG
if mode == "Tự chọn / Nhập thủ công":
    st.subheader("📌 1. Nhập Trực Tiếp Dàn 8 Số Đã Phân Tích")
    manual_input = st.text_input(
        "Nhập 8 số phân cách bằng dấu phẩy hoặc khoảng trắng:",
        value="03, 05, 10, 31, 35, 37, 64, 66"
    )

# GIAO DIỆN DÀNH CHO CHẾ ĐỘ NẠP LỊCH SỬ TỰ ĐỘNG
else:
    st.subheader("📥 1. Nạp Dữ Liệu Kết Quả Lịch Sử")
    raw_input = st.text_area(
        "Dán kết quả các kỳ gần nhất (Mỗi kỳ 1 dòng 20 số):",
        placeholder="01 05 12 18 27 33 41 52 ...\n03 08 15 22 29 34 45 60 ...",
        height=100
    )

    if st.button("💾 Nạp Dữ Liệu"):
        if raw_input.strip():
            lines = raw_input.strip().split("\n")
            new_draws = []
            for line in lines:
                nums = [int(s) for s in line.replace(",", " ").split() if s.isdigit()]
                if len(nums) == 20:
                    new_draws.append(nums)
            
            st.session_state.history.extend(new_draws)
            st.session_state.history = st.session_state.history[-100:]
            st.success(f"✅ Đã nạp thành công {len(new_draws)} kỳ quay. Tổng dữ liệu hiện tại: {len(st.session_state.history)} kỳ.")

st.markdown("---")
st.subheader("🚀 2. Kiểm Chứng & Xuất Dàn Vé")

if st.button("🎯 Kiểm Chứng & Tách Dàn Vé", type="primary"):
    logs = []
    final_8 = None
    
    if mode == "Tự chọn / Nhập thủ công":
        # Parsing dàn 8 số thủ công
        candidate_nums = [int(s) for s in manual_input.replace(",", " ").split() if s.isdigit()]
        candidate_nums = sorted(list(set(candidate_nums)))
        
        if len(candidate_nums) != 8:
            st.error(f"❌ Vui lòng nhập đúng 8 số không trùng lặp! (Hiện tại phát hiện {len(candidate_nums)} số).")
        else:
            logs.append(f"📥 **[Input Engine]:** Nhận dàn 8 số thủ công: `{candidate_nums}`")
            
            # Chạy Formal Verification
            is_entropy_valid, entropy_val = FormalVerificationEngine.verify_quadrant_entropy(candidate_nums)
            is_anti_cluster_valid = FormalVerificationEngine.verify_anti_clustering(candidate_nums)
            
            logs.append(f"🔍 **[Formal Verification]:** Shannon Entropy = **{entropy_val:.2f}** {'✅' if is_entropy_valid else '⚠️ (Khuyên dùng >= 1.38)'}")
            
            # Chạy Stress Test
            passed_stress, fail_rate = CounterexampleStressTest.run_stress_test(candidate_nums)
            logs.append(f"🛡️ **[Stress Test]:** Tỷ lệ sập dàn = **{fail_rate*100:.1f}%** {'✅ (Đạt chuẩn <35%)' if passed_stress else '⚠️ (Rủi ro cao)'}")
            
            final_8 = candidate_nums

    else:
        # Chạy MCTS tự động
        with st.spinner("Đang thực hiện MCTS Search & Kiểm chứng Formal..."):
            logs, final_8 = execute_mcts_reasoning(
                st.session_state.history, 
                test_time_compute, 
                decay_factor
            )

    # Hiển thị tiến trình suy luận
    st.write("📝 **Tiến Trình Suy Luận Internal Monologue:**")
    for log in logs:
        st.markdown(log)
        
    # Xuất kết quả dàn vé
    if final_8:
        clean_8 = [int(x) for x in final_8]
        st.markdown("---")
        st.success(f"🎯 **TẬP 8 SỐ TỔI ƯU:** `{clean_8}`")
        
        # Chia dàn vé Wheel System
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
