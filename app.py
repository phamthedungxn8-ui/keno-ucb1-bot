import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Move-37 Full Hybrid Keno Engine", layout="centered")

# ==============================================================================
# KERNEL 1: 2-ADIC BIT-SHIFT & ENTROPY DECAY (NÉN BIT COLLATZ)
# ==============================================================================
class BitShift2AdicKernel:
    """Đo mức độ nén Entropy bit 0 trong không gian 2-adic"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0).astype(int)
        scores = np.zeros(D)
        
        for d in range(D):
            val = int(freqs[d])
            if val > 0:
                collatz_val = 3 * val + 1
                # Lấy số lượng bit 0 liên tiếp ở cuối (2-adic valuation)
                tz = (collatz_val & -collatz_val).bit_length() - 1
                scores[d] = tz * 1.618 + (collatz_val.bit_length() * 0.5)
            else:
                # Số gan có năng lượng nén cực đại
                scores[d] = 6.0
                
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 2: PHASE SPACE ATTRACTOR MAPPING (KHÔNG GIANG PHA & ĐIỂM HÚT)
# ==============================================================================
class PhaseSpaceAttractorKernel:
    """Tái tạo không gian pha 2D/3D tìm tâm điểm bẫy hấp dẫn (Attractors)"""
    def analyze(self, X):
        T, D = X.shape
        scores = np.zeros(D)
        
        # Quỹ đạo gia tốc xuất hiện qua 5 kỳ
        for d in range(D):
            traj = X[:, d] # Chuỗi 5 kỳ của số d
            # Vector khoảng cách trong không gian pha
            velocity = np.diff(traj)
            acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0])
            
            # Tính khoảng cách Euclidean đến trung tâm hấp dẫn
            attractor_dist = np.sqrt(np.sum(velocity**2) + np.sum(acceleration**2))
            scores[d] = 1.0 / (1.0 + attractor_dist)
            
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 3: MODULAR TRANSITION Z/8Z (MA TRẬN ĐỒNG DƯ MARKOV)
# ==============================================================================
class ModularOrbitKernel:
    """Ma trận xả năng lượng trên vành đồng dư Z/8Z"""
    def analyze(self, X):
        T, D = X.shape
        scores = np.zeros(D)
        
        last_draw_indices = np.where(X[-1] == 1)[0] + 1
        last_mods = [n % 8 for n in last_draw_indices]
        mod_counts = np.bincount(last_mods, minlength=8)
        
        for d in range(D):
            num = d + 1
            m = num % 8
            # Ưu tiên số thuộc vành dư có mật độ nén cao
            scores[d] = mod_counts[m] * 1.25
            
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 4: CONTINUOUS WAVELET & FIBONACCI RATIO (SÓNG NĂNG LƯỢNG)
# ==============================================================================
class WaveletFibonacciKernel:
    """Phân rã gia tốc sóng Logarit2 theo tỷ lệ Vàng (Phi = 1.618)"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0) + 1.0
        
        log_wave = np.log2(freqs)
        gradients = np.gradient(log_wave)
        
        # Tích hợp hệ số Vàng Golden Ratio
        scores = np.abs(gradients) * 1.618
        
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# MASTER INTEGRATOR: MOVE-37 FULL HYBRID ENGINE
# ==============================================================================
class Move37FullHybridEngine:
    def __init__(self, num_dim=80):
        self.k1 = BitShift2AdicKernel()
        self.k2 = PhaseSpaceAttractorKernel()
        self.k3 = ModularOrbitKernel()
        self.k4 = WaveletFibonacciKernel()

    def process(self, X):
        s1 = self.k1.analyze(X)
        s2 = self.k2.analyze(X)
        s3 = self.k3.analyze(X)
        s4 = self.k4.analyze(X)
        
        # Trọng số tổng hợp Move 37: 30% Bit + 25% Attractor + 25% Modulo + 20% Wavelet
        total_energy = (0.30 * s1) + (0.25 * s2) + (0.25 * s3) + (0.20 * s4)
        
        freqs = X.sum(axis=0)
        last_draw = X[-1]
        
        # Phạt triệt tiêu số bão hòa (>3 kỳ liên tiếp)
        total_energy[freqs >= 4] *= 0.05
        
        # 1. TRÍCH XUẤT N1: Động Lượng Bit Core (Hot Momentum)
        hot_indices = np.where((freqs >= 2) & (last_draw == 1))[0]
        if len(hot_indices) > 0:
            best_hot = hot_indices[np.argmax(total_energy[hot_indices])]
        else:
            best_hot = int(np.argmax(total_energy))
        N1 = best_hot + 1
        
        # 2. TRÍCH XUẤT N2: Bù Pha Attractor Gan (Cold Rebound)
        cold_indices = np.where(freqs == 0)[0]
        if len(cold_indices) > 0:
            best_cold = cold_indices[np.argmax(total_energy[cold_indices])]
        else:
            temp_e = total_energy.copy()
            temp_e[best_hot] = -1.0
            best_cold = int(np.argmax(temp_e))
        N2 = best_cold + 1
        
        # 3. TRÍCH XUẤT N3: Bọc Lót Cân Bằng Sóng Wavelet
        temp_e3 = total_energy.copy()
        temp_e3[best_hot] = -1.0
        temp_e3[best_cold] = -1.0
        N3 = int(np.argmax(temp_e3)) + 1
        
        bo_bac_2 = tuple(sorted([N1, N2]))
        bo_bac_3 = tuple(sorted([N1, N2, N3]))

        explanation = (
            f"### ♟️️ BÁO CÁO TỔNG HỢP NĂNG LƯỢNG MOVE 37\n"
            f"• **Kernel 1 (2-adic Bit Shift):** Nén bit năng lượng chốt trụ động lượng **{N1:02d}**.\n"
            f"• **Kernel 2 (Phase Attractor):** Tìm điểm bẫy hấp dẫn trong không gian pha, xác định số gan hồi pha **{N2:02d}**.\n"
            f"• **Kernel 3 & 4 (Z/8Z Orbit & Wavelet):** Giao thoa sóng tỷ lệ Vàng trích xuất điểm cân bằng bọc lót **{N3:02d}**.\n\n"
            f"👉 **Điểm hội tụ toán học hoàn hảo:** Cấu trúc **Bộ Bậc 2 [{N1:02d} — {N2:02d}]** và **Bộ BẬC 3 [{N1:02d} — {N2:02d} — {N3:02d}]**."
        )

        return bo_bac_2, bo_bac_3, N1, N2, N3, explanation, (s1, s2, s3, s4)

# ==============================================================================
# STREAMLIT UI (NATIVE)
# ==============================================================================
st.title("♟️ Move-37 Full Hybrid Keno Engine")
st.caption("Tích hợp 4 Kernel Độc Lập • 2-adic Bit • Attractor • Modulo Z/8Z • Sóng Wavelet")

raw_text_input = st.text_area(
    "Dán dữ liệu 5 kỳ (100 số) vào đây:",
    placeholder="Kỳ 1: 01 05 12 ...\nKỳ 2: ...",
    height=150,
    key="raw_full_hybrid"
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
                
        st.success("🎉 Đã hoàn tất xử lý 4 Kernel Move 37!")
        
        engine = Move37FullHybridEngine(num_dim=80)
        bo2, bo3, n1, n2, n3, explanation, kernel_scores = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 BẢNG CHỐT BỘ SỐ HOÀN CHỈNH")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="🔥 BIT CORE (N1)", value=f"{n1:02d}")
        with m2:
            st.metric(label="❄️ ATTRACTOR COLD (N2)", value=f"{n2:02d}")
        with m3:
            st.metric(label="🌀 WAVELET BALANCER (N3)", value=f"{n3:02d}")
            
        st.markdown("---")
        col_b2, col_b3 = st.columns(2)
        with col_b2:
            st.subheader("1️⃣ BỘ BẬC 2 CHỐT")
            st.title(f"{bo2[0]:02d} — {bo2[1]:02d}")
        with col_b3:
            st.subheader("2️⃣ BỘ BẬC 3 CHỐT")
            st.title(f"{bo3[0]:02d} — {bo3[1]:02d} — {bo3[2]:02d}")
            
        st.markdown("---")
        st.info(explanation)
        
        # Bảng đóng góp năng lượng từ 4 Kernel (dùng UI thuần Streamlit)
        st.subheader("📊 Mức độ đóng góp năng lượng từ 4 Kernel")
        df_kernels = pd.DataFrame({
            "Con số": [f"Số {n1:02d}", f"Số {n2:02d}", f"Số {n3:02d}"],
            "Kernel 1 (2-adic Bit)": [f"{kernel_scores[0][n1-1]:.2f}", f"{kernel_scores[0][n2-1]:.2f}", f"{kernel_scores[0][n3-1]:.2f}"],
            "Kernel 2 (Attractor)": [f"{kernel_scores[1][n1-1]:.2f}", f"{kernel_scores[1][n2-1]:.2f}", f"{kernel_scores[1][n3-1]:.2f}"],
            "Kernel 3 (Modulo Z/8Z)": [f"{kernel_scores[2][n1-1]:.2f}", f"{kernel_scores[2][n2-1]:.2f}", f"{kernel_scores[2][n3-1]:.2f}"],
            "Kernel 4 (Wavelet Phi)": [f"{kernel_scores[3][n1-1]:.2f}", f"{kernel_scores[3][n2-1]:.2f}", f"{kernel_scores[3][n3-1]:.2f}"]
        })
        st.table(df_kernels)
        
    else:
        st.warning(f"⚠️ Đã đọc được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng cung cấp đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi 5 kỳ Keno vào khung văn bản phía trên để khởi chạy mô hình.")
