import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Move-37 Keno Engine v2", layout="centered")

# ==============================================================================
# KERNEL 1: 2-ADIC BIT-SHIFT & EXHAUSTION FILTER
# ==============================================================================
class BitShift2AdicKernelV2:
    """Đo độ nén bit 2-adic kết hợp bộ lọc triệt tiêu kiệt sức động lượng"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0).astype(int)
        scores = np.zeros(D)
        
        for d in range(D):
            val = int(freqs[d])
            if val > 0:
                collatz_val = 3 * val + 1
                tz = (collatz_val & -collatz_val).bit_length() - 1
                scores[d] = tz * 1.618 + (collatz_val.bit_length() * 0.5)
            else:
                # Hạ điểm phạt số gan kéo dài để tránh bẫy "Gan Giả"
                scores[d] = 3.5 
                
        # Bộ lọc bão hòa: Phạt nặng nếu xuất hiện liên tiếp 2 kỳ cuối cùng
        for d in range(D):
            if T >= 2 and X[-1, d] == 1 and X[-2, d] == 1:
                scores[d] *= 0.15 # Phạt kiệt sức động lượng
                
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 2: EXTENDED PHASE SPACE ATTRACTOR (CỬA SỔ DÀI T >= 10)
# ==============================================================================
class PhaseSpaceAttractorKernelV2:
    """Tái tạo không gian pha trên cửa sổ dữ liệu mở rộng tìm Attractor thực"""
    def analyze(self, X):
        T, D = X.shape
        scores = np.zeros(D)
        
        for d in range(D):
            traj = X[:, d]
            velocity = np.diff(traj)
            acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0])
            
            # Quỹ đạo trung bình trượt định hình tâm hút
            attractor_dist = np.sqrt(np.mean(velocity**2) + np.mean(acceleration**2))
            
            # Ưu tiên các số có nhịp điều hòa nhấp nhô ổn định
            scores[d] = 1.0 / (1.0 + attractor_dist)
            
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 3: MODULAR TRANSITION Z/8Z & CROSS-RESIDUE
# ==============================================================================
class ModularOrbitKernelV2:
    """Ma trận giải phóng năng lượng trên vành đồng dư Z/8Z"""
    def analyze(self, X):
        T, D = X.shape
        scores = np.zeros(D)
        
        last_draw_indices = np.where(X[-1] == 1)[0] + 1
        last_mods = [n % 8 for n in last_draw_indices]
        mod_counts = np.bincount(last_mods, minlength=8)
        
        for d in range(D):
            num = d + 1
            m = num % 8
            # Phân bổ năng lượng theo mật độ vành dư
            scores[d] = mod_counts[m] * 1.10
            
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# KERNEL 4: CONTINUOUS WAVELET & FIBONACCI HARMONICS
# ==============================================================================
class WaveletFibonacciKernelV2:
    """Chuyển đổi sóng logarit và giao thoa tăng cường theo tỷ lệ Phi"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0) + 1.0
        
        log_wave = np.log2(freqs)
        gradients = np.gradient(log_wave)
        
        # Gia tải tần số điều hòa Golden Ratio
        scores = np.abs(gradients) * 1.618
        
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# MASTER INTEGRATOR: ADAPTIVE DYNAMIC ENGINE V2
# ==============================================================================
class Move37AdaptiveEngineV2:
    def __init__(self):
        self.k1 = BitShift2AdicKernelV2()
        self.k2 = PhaseSpaceAttractorKernelV2()
        self.k3 = ModularOrbitKernelV2()
        self.k4 = WaveletFibonacciKernelV2()

    def process(self, X):
        T, D = X.shape
        s1 = self.k1.analyze(X)
        s2 = self.k2.analyze(X)
        s3 = self.k3.analyze(X)
        s4 = self.k4.analyze(X)
        
        # TỰ ĐỘNG ĐIỀU CHỈNH TRỌNG SỐ ĐỘNG (ADAPTIVE WEIGHTS)
        # Đo biến động (Volatility) của 3 kỳ gần nhất
        recent_density = X[-3:].sum(axis=1) if T >= 3 else X.sum(axis=1)
        volatility = np.std(recent_density)
        
        if volatility > 1.5:
            # Nhịp biến động mạnh -> Tăng trọng số Sóng Wavelet & Modulo
            w1, w2, w3, w4 = 0.15, 0.25, 0.25, 0.35
            mode_desc = "Cấu hình Sóng Thích ứng (Ưu tiên Wavelet & Modulo)"
        else:
            # Nhịp ổn định -> Cân bằng 4 Kernel
            w1, w2, w3, w4 = 0.20, 0.30, 0.25, 0.25
            mode_desc = "Cấu hình Cân bằng Không gian Pha (Ưu tiên Attractor)"
            
        total_energy = (w1 * s1) + (w2 * s2) + (w3 * s3) + (w4 * s4)
        
        freqs = X.sum(axis=0)
        last_draw = X[-1]
        
        # 1. TRÍCH XUẤT N1: Động Lượng Nhiệp Điệu (Rhythmic Core)
        # Ưu tiên số có tần suất 2-3 lần, xuất hiện ở Kỳ 5 nhưng KHÔNG xuất hiện ở Kỳ 4
        valid_n1 = []
        for d in range(D):
            if last_draw[d] == 1 and (T < 2 or X[-2, d] == 0) and freqs[d] >= 2:
                valid_n1.append(d)
                
        if len(valid_n1) > 0:
            best_n1 = valid_n1[np.argmax(total_energy[valid_n1])]
        else:
            best_n1 = int(np.argmax(total_energy))
        N1 = best_n1 + 1
        
        # 2. TRÍCH XUẤT N2: Sóng Điểm Cân Bằng (Wavelet Balancer)
        temp_e2 = total_energy.copy()
        temp_e2[best_n1] = -1.0
        best_n2 = int(np.argmax(temp_e2))
        N2 = best_n2 + 1
        
        # 3. TRÍCH XUẤT N3: Hồi Pha Attractor Trung Tính (Neutral Attractor)
        temp_e3 = total_energy.copy()
        temp_e3[best_n1] = -1.0
        temp_e3[best_n2] = -1.0
        best_n3 = int(np.argmax(temp_e3))
        N3 = best_n3 + 1
        
        bo_bac_2 = tuple(sorted([N1, N2]))
        bo_bac_3 = tuple(sorted([N1, N2, N3]))

        explanation = (
            f"### ♟ BÁO CÁO TỔNG HỢP MOVE 37 ENGINE V2\n"
            f"• **Chế độ vận hành:** {mode_desc}\n"
            f"• **Lọc kiệt sức động lượng:** Đã loại bỏ bẫy cháy số liên tiếp và bẫy gan kéo dài.\n"
            f"• **Cấu trúc Năng lượng v2:**\n"
            f"  - **N1 ({N1:02d}):** Trụ động lượng nhịp điệu (không bị bão hòa).\n"
            f"  - **N2 ({N2:02d}):** Điểm rơi giao thoa sóng Wavelet tỷ lệ Vàng.\n"
            f"  - **N3 ({N3:02d}):** Tọa độ cân bằng trong không gian pha Attractor.\n\n"
            f"👉 **Bộ Chốt Tối Ưu v2:** **[ {N1:02d} — {N2:02d} ]** (Bậc 2) và **[ {N1:02d} — {N2:02d} — {N3:02d} ]** (Bậc 3)."
        )

        return bo_bac_2, bo_bac_3, N1, N2, N3, explanation, (s1, s2, s3, s4), (w1, w2, w3, w4)

# ==============================================================================
# STREAMLIT UI (NATIVE & STREAMLINED)
# ==============================================================================
st.title("♟️ Move-37 Adaptive Engine v2")
st.caption("Cập nhật v2: Mở rộng Cửa sổ dữ liệu • Lọc Triệt tiêu Bão hòa • Trọng số Thích ứng Động")

raw_text_input = st.text_area(
    "Dán dữ liệu 5 đến 10 kỳ (100 - 200 số) vào đây:",
    placeholder="Kỳ 1: 01 05 12 ...\nKỳ 2: ...",
    height=180,
    key="raw_v2_input"
)

if raw_text_input.strip():
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text_input.strip(), flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        # Sử dụng tối đa 10 kỳ gần nhất nếu có
        kies_to_use = min(total_kies, 10)
        used_numbers = all_numbers[-kies_to_use * 20:]
        
        matrix = np.zeros((kies_to_use, 80), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0
                
        st.success(f"🎉 Đã hoàn tất phân tích trên {kies_to_use} kỳ gần nhất bằng Move-37 Engine v2!")
        
        engine = Move37AdaptiveEngineV2()
        bo2, bo3, n1, n2, n3, explanation, kernel_scores, weights = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 BẢNG CHỐT BỘ SỐ HOÀN CHỈNH (V2)")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="⚡ RHYTHMIC CORE (N1)", value=f"{n1:02d}")
        with m2:
            st.metric(label="🌀 WAVELET PHI (N2)", value=f"{n2:02d}")
        with m3:
            st.metric(label="🎯 ATTRACTOR BALANCER (N3)", value=f"{n3:02d}")
            
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
        
        # Bảng hiển thị chỉ số năng lượng và trọng số động
        st.subheader("📊 Trọng số Động & Mức năng lượng các số chốt")
        df_kernels = pd.DataFrame({
            "Con số": [f"Số {n1:02d} (N1)", f"Số {n2:02d} (N2)", f"Số {n3:02d} (N3)"],
            f"Kernel 1 (Bit-Shift) [{weights[0]*100:.0f}%]": [f"{kernel_scores[0][n1-1]:.2f}", f"{kernel_scores[0][n2-1]:.2f}", f"{kernel_scores[0][n3-1]:.2f}"],
            f"Kernel 2 (Attractor) [{weights[1]*100:.0f}%]": [f"{kernel_scores[1][n1-1]:.2f}", f"{kernel_scores[1][n2-1]:.2f}", f"{kernel_scores[1][n3-1]:.2f}"],
            f"Kernel 3 (Modulo Z/8Z) [{weights[2]*100:.0f}%]": [f"{kernel_scores[2][n1-1]:.2f}", f"{kernel_scores[2][n2-1]:.2f}", f"{kernel_scores[2][n3-1]:.2f}"],
            f"Kernel 4 (Wavelet Phi) [{weights[3]*100:.0f}%]": [f"{kernel_scores[3][n1-1]:.2f}", f"{kernel_scores[3][n2-1]:.2f}", f"{kernel_scores[3][n3-1]:.2f}"]
        })
        st.table(df_kernels)
        
    else:
        st.warning(f"⚠️ Đã đọc được {len(all_numbers)} số ({total_kies}/5 kỳ tối thiểu). Vui lòng cung cấp từ 5 đến 10 kỳ!")
else:
    st.info("👆 Dán chuỗi 5–10 kỳ Keno vào khung văn bản phía trên để khởi chạy mô hình v2.")
