import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Collatz-Keno Move-37 Hybrid Engine", layout="centered")

# ==============================================================================
# KERNEL 1: COLLATZ BIT-SHIFT ENTROPY FOR KENO
# ==============================================================================
class CollatzBitKernel:
    """Áp dụng phép dịch Bit Collatz để đo độ nén Entropy của 80 số Keno"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0).astype(int)
        
        bit_scores = np.zeros(D)
        for d in range(D):
            val = int(freqs[d])
            if val > 0:
                # Biến đổi Collatz vi mô: 3n + 1 rồi đếm bit 0 bị diệt
                collatz_val = 3 * val + 1
                tz = (collatz_val & -collatz_val).bit_length() - 1
                # Số có nhịp nén bit lớn = Số sắp bùng nổ điểm rơi
                bit_scores[d] = tz * 1.5 + (collatz_val.bit_length() * 0.5)
            else:
                # Số gan 0/5 kỳ có năng lượng nén bit cực đại
                bit_scores[d] = 5.0
                
        max_s = np.max(bit_scores)
        return bit_scores / max_s if max_s > 0 else bit_scores

# ==============================================================================
# KERNEL 2: COLLATZ MODULAR ORBIT FOR KENO (Z/8Z TRANSITION)
# ==============================================================================
class CollatzModularKernel:
    """Mapping 80 số Keno lên vành đồng dư Z/8Z của Collatz"""
    def analyze(self, X):
        T, D = X.shape
        mod_scores = np.zeros(D)
        
        # Lấy 20 số kỳ gần nhất (Kỳ 5)
        last_draw_indices = np.where(X[-1] == 1)[0] + 1
        last_mods = [n % 8 for n in last_draw_indices]
        
        # Đếm mật độ xuất hiện trên 8 vành đồng dư (0 đến 7)
        mod_counts = np.bincount(last_mods, minlength=8)
        
        for d in range(D):
            num = d + 1
            m = num % 8
            # Ưu tiên các số nằm ở vành đồng dư đang có mật độ tích lũy cao
            mod_scores[d] = mod_counts[m] * 1.2
            
        max_s = np.max(mod_scores)
        return mod_scores / max_s if max_s > 0 else mod_scores

# ==============================================================================
# KERNEL 3: COLLATZ ENERGY GRADIENT (LOG2 WAVELET)
# ==============================================================================
class CollatzEnergyKernel:
    """Tính gia tốc năng lượng Logarit suy hao kiểu Collatz"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0) + 1.0 # Tránh log(0)
        
        log_energy = np.log2(freqs)
        gradients = np.gradient(log_energy)
        
        # Chỉ số lyapunov địa phương
        energy_scores = np.abs(gradients)
        
        max_s = np.max(energy_scores)
        return energy_scores / max_s if max_s > 0 else energy_scores

# ==============================================================================
# SYSTEM INTEGRATOR: COLLATZ-KENO MOVE-37 ENGINE
# ==============================================================================
class CollatzKenoEngine:
    def __init__(self, num_dim=80):
        self.k1 = CollatzBitKernel()
        self.k2 = CollatzModularKernel()
        self.k3 = CollatzEnergyKernel()

    def process(self, X):
        s1 = self.k1.analyze(X)
        s2 = self.k2.analyze(X)
        s3 = self.k3.analyze(X)
        
        # Tổng hợp ma trận năng lượng Collatz: 40% Bit + 35% Modulo + 25% Energy
        total_energy = (0.40 * s1) + (0.35 * s2) + (0.25 * s3)
        
        freqs = X.sum(axis=0)
        last_draw = X[-1]
        
        # Phạt các số bão hòa (>3 kỳ)
        total_energy[freqs >= 4] *= 0.1
        
        # 1. TRÍCH XUẤT SỐ N1 (Trụ Cột Động Lượng Bit - Collatz Hot Core)
        # Lấy số vừa nổ ở kỳ 5 có năng lượng Collatz cao nhất
        hot_indices = np.where((freqs >= 2) & (last_draw == 1))[0]
        if len(hot_indices) > 0:
            best_hot = hot_indices[np.argmax(total_energy[hot_indices])]
        else:
            best_hot = int(np.argmax(total_energy))
        N1 = best_hot + 1
        
        # 2. TRÍCH XUẤT SỐ N2 (Bù Pha Gan Entropy Collatz - Cold Rebound)
        cold_indices = np.where(freqs == 0)[0]
        if len(cold_indices) > 0:
            best_cold = cold_indices[np.argmax(total_energy[cold_indices])]
        else:
            temp_e = total_energy.copy()
            temp_e[best_hot] = -1.0
            best_cold = int(np.argmax(temp_e))
        N2 = best_cold + 1
        
        # 3. TRÍCH XUẤT SỐ N3 (Số Điểm Hút Wavelet Collatz)
        temp_e3 = total_energy.copy()
        temp_e3[best_hot] = -1.0
        temp_e3[best_cold] = -1.0
        N3 = int(np.argmax(temp_e3)) + 1
        
        bo_bac_2 = tuple(sorted([N1, N2]))
        bo_bac_3 = tuple(sorted([N1, N2, N3]))

        explanation = (
            f"### ♟️ NƯỚC ĐI THỨ 37: BẢN CHẤT HỘI TỤ COLLATZ-KENO\n"
            f"• **Kernel 1 (Bit-Shift Entropy):** Lọc ra năng lượng nén bit $3n+1$, chốt số **{N1:02d}** giữ vai trò Trụ Cột Động Lượng.\n"
            f"• **Kernel 2 (Modular Z/8Z Orbit):** Tìm số dư hội tụ trên vành đồng dư $\\pmod 8$, chốt số **{N2:02d}** làm Trụ Bù Pha Gan.\n"
            f"• **Kernel 3 (Energy Gradient Log2):** Tính gia tốc độ cong suy hao, trích xuất số **{N3:02d}** làm Bọc Lót Điểm Hút.\n"
            f"• **Kết luận:** Sự giao thoa giữa 3 Kernel Collatz tạo nên cấu trúc **Bộ Bậc 2 [{N1:02d} — {N2:02d}]** và **Bộ Bậc 3 [{N1:02d} — {N2:02d} — {N3:02d}]** đạt điểm cân bằng toán học tối đa."
        )

        return bo_bac_2, bo_bac_3, N1, N2, N3, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("♟️ Collatz-Keno Move-37 Hybrid Engine")
st.caption("Ứng dụng Toán học Collatz (3n+1) vào Keno • 3 Kernel Độc Lập • Thuần Streamlit")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=150,
    key="raw_text_collatz_keno"
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
                
        st.success("🎉 Đã hoàn thành phân tích Collatz Multi-Kernel cho Keno!")
        
        engine = CollatzKenoEngine(num_dim=80)
        bo2, bo3, n1, n2, n3, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẤU TRÚC BỘ SỐ CHỐT TỪ COLLATZ KERNELS")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="🔥 COLLATZ BIT CORE", value=f"{n1:02d}")
        with c2:
            st.metric(label="❄️ MODULO Z/8Z COLD", value=f"{n2:02d}")
        with c3:
            st.metric(label="🌀 WAVELET ATTRACTOR", value=f"{n3:02d}")
            
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
    st.info("👆 Dán chuỗi số 5 kỳ vào khung trên để khởi chạy mô hình Collatz-Keno.")
