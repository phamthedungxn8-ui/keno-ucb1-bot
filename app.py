import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Collatz Multi-Kernel Engine v2", layout="centered")

# ==============================================================================
# KERNEL 1: 2-ADIC COLLATZ & EXHAUSTION FILTER
# ==============================================================================
class Collatz2AdicKernelV2:
    """Tăng cường giả thuyết Collatz 3n+1 kết hợp Bộ lọc Bão hòa Động lượng"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0).astype(int)
        scores = np.zeros(D)
        
        for d in range(D):
            val = int(freqs[d])
            if val > 0:
                # Phép biến đổi Collatz 3n + 1
                collatz_val = 3 * val + 1
                # Đếm số bit 0 ở cuối (2-adic valuation)
                tz = (collatz_val & -collatz_val).bit_length() - 1
                scores[d] = tz * 1.618 + (collatz_val.bit_length() * 0.5)
            else:
                # Giảm ưu tiên số gan tuyệt đối để tránh bẫy gan kéo dài
                scores[d] = 2.0 
                
        # BỘ LỌC TRIỆT TIÊU BÃO HÒA (Phạt nặng nếu về 2 kỳ liên tiếp như số 24)
        for d in range(D):
            if T >= 2 and X[-1, d] == 1 and X[-2, d] == 1:
                scores[d] *= 0.15 
                
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 2: ATTRACTOR KHÔNG GIAN PHA (PHASE SPACE ATTRACTOR)
# ==============================================================================
class PhaseSpaceAttractorKernelV2:
    """Đo độ lệch đạo hàm bậc 1 & 2 trên quỹ đạo không gian pha"""
    def analyze(self, X):
        T, D = X.shape
        scores = np.zeros(D)
        
        for d in range(D):
            traj = X[:, d]
            velocity = np.diff(traj)
            acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0])
            
            # Tính khoảng cách Euclidean đến tâm hút Attractor
            attractor_dist = np.sqrt(np.mean(velocity**2) + np.mean(acceleration**2))
            scores[d] = 1.0 / (1.0 + attractor_dist)
            
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 3: MODULAR TRANSITION Z/8Z
# ==============================================================================
class ModularOrbitKernelV2:
    """Ma trận đồng dư Z/8Z giải phóng năng lượng nhịp"""
    def analyze(self, X):
        T, D = X.shape
        scores = np.zeros(D)
        
        last_draw_indices = np.where(X[-1] == 1)[0] + 1
        last_mods = [n % 8 for n in last_draw_indices]
        mod_counts = np.bincount(last_mods, minlength=8)
        
        for d in range(D):
            num = d + 1
            m = num % 8
            scores[d] = mod_counts[m] * 1.12
            
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 4: WAVELET LOGARITHMIC & FIBONACCI HARMONICS
# ==============================================================================
class WaveletFibonacciKernelV2:
    """Chuyển đổi sóng Logarit và giao thoa Tỷ lệ Vàng Phi"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0) + 1.0
        
        log_wave = np.log2(freqs)
        gradients = np.gradient(log_wave)
        
        # Gia tải tần số điều hòa Golden Ratio (1.618)
        scores = np.abs(gradients) * 1.618
        
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# INTEGRATOR: ADAPTIVE COLLATZ ENGINE V2
# ==============================================================================
class CollatzEngineV2:
    def __init__(self):
        self.k1 = Collatz2AdicKernelV2()
        self.k2 = PhaseSpaceAttractorKernelV2()
        self.k3 = ModularOrbitKernelV2()
        self.k4 = WaveletFibonacciKernelV2()

    def process(self, X):
        T, D = X.shape
        s1 = self.k1.analyze(X)
        s2 = self.k2.analyze(X)
        s3 = self.k3.analyze(X)
        s4 = self.k4.analyze(X)
        
        # TÍNH TRỌNG SỐ THÍCH ỨNG ĐỘNG (ADAPTIVE WEIGHTS)
        recent_density = X[-3:].sum(axis=1) if T >= 3 else X.sum(axis=1)
        volatility = np.std(recent_density)
        
        if volatility > 1.2:
            # Chuỗi biến động mạnh -> Tăng ưu tiên Sóng Wavelet & Đồng dư
            w1, w2, w3, w4 = 0.15, 0.25, 0.25, 0.35
            mode_label = "Chế độ Sóng Thích ứng (Ưu tiên Wavelet & Modulo)"
        else:
            # Chuỗi ổn định -> Cân bằng 4 Kernel
            w1, w2, w3, w4 = 0.20, 0.30, 0.25, 0.25
            mode_label = "Chế độ Cân bằng Không gian Pha"
            
        total_energy = (w1 * s1) + (w2 * s2) + (w3 * s3) + (w4 * s4)
        
        freqs = X.sum(axis=0)
        last_draw = X[-1]
        
        # TRÍCH XUẤT N1: Động lượng nhịp điệu (Loại bỏ các số bão hòa như 24)
        valid_n1 = []
        for d in range(D):
            # Điều kiện: Xuất hiện ở Kỳ 5 nhưng KHÔNG xuất hiện ở Kỳ 4 (Tránh bẫy cháy 2 kỳ)
            if last_draw[d] == 1 and (T < 2 or X[-2, d] == 0) and freqs[d] >= 2:
                valid_n1.append(d)
                
        if len(valid_n1) > 0:
            best_n1 = valid_n1[np.argmax(total_energy[valid_n1])]
        else:
            best_n1 = int(np.argmax(total_energy))
        N1 = best_n1 + 1
        
        # TRÍCH XUẤT N2: Sóng Wavelet Phi Balancer (Nhịp giống con số 08)
        temp_e2 = total_energy.copy()
        temp_e2[best_n1] = -1.0
        best_n2 = int(np.argmax(temp_e2))
        N2 = best_n2 + 1
        
        # TRÍCH XUẤT N3: Hồi pha Attractor Trung tính (Tránh bẫy số gan dài như 80)
        temp_e3 = total_energy.copy()
        temp_e3[best_n1] = -1.0
        temp_e3[best_n2] = -1.0
        best_n3 = int(np.argmax(temp_e3))
        N3 = best_n3 + 1
        
        bo_bac_2 = tuple(sorted([N1, N2]))
        bo_bac_3 = tuple(sorted([N1, N2, N3]))

        return bo_bac_2, bo_bac_3, N1, N2, N3, mode_label, (s1, s2, s3, s4), (w1, w2, w3, w4)

# ==============================================================================
# STREAMLIT USER INTERFACE
# ==============================================================================
st.title("♟️ Collatz Multi-Kernel Engine v2")
st.caption("Mô hình tái tạo nâng cấp: Lọc Bão hòa Động lượng • Thích ứng Trọng số Động")

raw_input = st.text_area(
    "Dán dữ liệu từ 5 đến 10 kỳ vào đây:",
    placeholder="Kỳ 1: 01 05 12 ...\nKỳ 2: ...",
    height=180
)

if raw_input.strip():
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_input.strip(), flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        kies_to_use = min(total_kies, 10)
        used_numbers = all_numbers[-kies_to_use * 20:]
        
        matrix = np.zeros((kies_to_use, 80), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0
                
        engine = CollatzEngineV2()
        bo2, bo3, n1, n2, n3, mode, scores, weights = engine.process(matrix)
        
        st.success(f"⚡ Phân tích thành công {kies_to_use} kỳ dữ liệu bằng Engine v2!")
        st.info(f"📌 **Cấu hình thuật toán:** {mode}")
        
        st.markdown("---")
        st.subheader("🎯 TỔNG HỢP BỘ SỐ CHỐT MỚI (V2)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("N1 (Rhythmic Core)", f"{n1:02d}")
        with c2:
            st.metric("N2 (Wavelet Balancer)", f"{n2:02d}")
        with c3:
            st.metric("N3 (Neutral Attractor)", f"{n3:02d}")
            
        col2, col3 = st.columns(2)
        with col2:
            st.subheader("BỘ BẬC 2 CHỐT")
            st.title(f"{bo2[0]:02d} — {bo2[1]:02d}")
        with col3:
            st.subheader("BỘ BẬC 3 CHỐT")
            st.title(f"{bo3[0]:02d} — {bo3[1]:02d} — {bo3[2]:02d}")
            
        st.markdown("---")
        st.subheader("📊 Mức Năng Lượng Đã Điều Chỉnh Của Các Kernel")
        df_res = pd.DataFrame({
            "Con số": [f"Số {n1:02d}", f"Số {n2:02d}", f"Số {n3:02d}"],
            f"K1 (Collatz 2-adic) [{weights[0]*100:.0f}%]": [f"{scores[0][n1-1]:.2f}", f"{scores[0][n2-1]:.2f}", f"{scores[0][n3-1]:.2f}"],
            f"K2 (Attractor) [{weights[1]*100:.0f}%]": [f"{scores[1][n1-1]:.2f}", f"{scores[1][n2-1]:.2f}", f"{scores[1][n3-1]:.2f}"],
            f"K3 (Modulo Z/8Z) [{weights[2]*100:.0f}%]": [f"{scores[2][n1-1]:.2f}", f"{scores[2][n2-1]:.2f}", f"{scores[2][n3-1]:.2f}"],
            f"K4 (Wavelet Phi) [{weights[3]*100:.0f}%]": [f"{scores[3][n1-1]:.2f}", f"{scores[3][n2-1]:.2f}", f"{scores[3][n3-1]:.2f}"]
        })
        st.table(df_res)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hiện tại đọc được {total_kies} kỳ.")3 (Modulo Z/8Z) [{weights[2]*100:.0f}%]": [f"{kernel_scores[2][n1-1]:.2f}", f"{kernel_scores[2][n2-1]:.2f}", f"{kernel_scores[2][n3-1]:.2f}"],
            f"Kernel 4 (Wavelet Phi) [{weights[3]*100:.0f}%]": [f"{kernel_scores[3][n1-1]:.2f}", f"{kernel_scores[3][n2-1]:.2f}", f"{kernel_scores[3][n3-1]:.2f}"]
        })
        st.table(df_kernels)
        
    else:
        st.warning(f"⚠️ Đã đọc được {len(all_numbers)} số ({total_kies}/5 kỳ tối thiểu). Vui lòng cung cấp từ 5 đến 10 kỳ!")
else:
    st.info("👆 Dán chuỗi 5–10 kỳ Keno vào khung văn bản phía trên để khởi chạy mô hình v2.")
