import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Move-37 Micro-Chunk Engine v3", layout="centered")

# ==============================================================================
# 1. KERNEL SHANNON ENTROPY (MICRO-WINDOW DATA CHUNKING)
# ==============================================================================
class MicroWindowEntropyKernel:
    """Chia nhỏ dữ liệu thành các vi cửa sổ (k=3) và tính độ biến thiên Entropy"""
    def analyze(self, X):
        T, D = X.shape
        if T < 3:
            return np.ones(D) / D
            
        entropies = np.zeros(D)
        # Quét trượt qua từng vi cửa sổ 3 kỳ
        for t in range(T - 2):
            window = X[t : t + 3] # Cửa sổ trượt 3 kỳ
            p = window.sum(axis=0) / 3.0 # Xác suất xuất hiện vi mô
            for d in range(D):
                prob = p[d]
                if 0 < prob < 1:
                    # Shannon Entropy: H(X) = -p*log2(p) - (1-p)*log2(1-p)
                    entropies[d] += -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
                    
        max_e = np.max(entropies)
        return entropies / max_e if max_e > 0 else entropies

# ==============================================================================
# 2. KERNEL COLLATZ-MARKOV TRANSITION (MA TRẬN CHUYỂN TRẠNG THÁI)
# ==============================================================================
class CollatzMarkovKernel:
    """Mã hóa tần suất theo Collatz 3n+1 và dựng ma trận chuyển Markov"""
    def analyze(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0).astype(int)
        scores = np.zeros(D)
        
        # Ma trận chuyển trạng thái Markov 2x2 cho từng con số
        for d in range(D):
            # Tính toán xích Markov từ chuỗi xuất hiện (0/1)
            history = X[:, d]
            transitions = np.zeros((2, 2))
            for i in range(len(history) - 1):
                from_state = int(history[i])
                to_state = int(history[i+1])
                transitions[from_state, to_state] += 1
                
            # Chuẩn hóa xác suất chuyển
            row_sums = transitions.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            prob_matrix = transitions / row_sums
            
            # Khai thác điểm rơi Collatz kết hợp xác suất chuyển 0 -> 1
            v = freqs[d]
            collatz_factor = (3 * v + 1) % 8
            move37_score = prob_matrix[int(history[-1]), 1] * (collatz_factor + 1)
            scores[d] = move37_score
            
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# 3. KERNEL WAVELET SINGULARITY (ĐIỂM ĐỨT GÃY SÓNG)
# ==============================================================================
class WaveletSingularityKernel:
    """Bắt các điểm đứt gãy phi tuyến tính trong chuỗi thời gian vi mô"""
    def analyze(self, X):
        T, D = X.shape
        scores = np.zeros(D)
        for d in range(D):
            signal = X[:, d]
            # Sử dụng đạo hàm bậc 2 để tìm điểm uốn (Inflection Points)
            diff2 = np.diff(signal, n=2) if len(signal) >= 3 else np.array([0])
            scores[d] = np.sum(np.abs(diff2)) + 1.0 / (np.std(signal) + 1.0)
            
        max_s = np.max(scores)
        return scores / max_s if max_s > 0 else scores

# ==============================================================================
# MASTER ENGINE: MOVE-37 INTEGRATOR V3
# ==============================================================================
class Move37EngineV3:
    def __init__(self):
        self.k_entropy = MicroWindowEntropyKernel()
        self.k_markov = CollatzMarkovKernel()
        self.k_wavelet = WaveletSingularityKernel()

    def process(self, X):
        T, D = X.shape
        
        # 1. Phân tích trên các tầng Kernel vi mô
        s_entropy = self.k_entropy.analyze(X)
        s_markov = self.k_markov.analyze(X)
        s_wavelet = self.k_wavelet.analyze(X)
        
        # 2. Tổng hợp Năng lượng Move 37 (Phi tuyến tính)
        # Sử dụng Tích hình học (Geometric Mean) thay cho Cộng trọng số tuyến tính
        # Tích hình học giúp lọc bỏ các số chỉ nhỉnh hơn ở 1 Kernel nhưng kém ở các Kernel khác
        combined_energy = (s_entropy * s_markov * s_wavelet) ** (1.0 / 3.0)
        
        # 3. Lọc triệt tiêu kiệt sức & Bẫy bão hòa vi mô
        freqs = X.sum(axis=0)
        for d in range(D):
            # Nếu nổ 2 kỳ liên tiếp gần nhất -> Phạt nặng
            if T >= 2 and X[-1, d] == 1 and X[-2, d] == 1:
                combined_energy[d] *= 0.05
            # Nếu là số Gan kéo dài (> 5 kỳ chưa về) -> Phạt bẫy gan
            if freqs[d] == 0:
                combined_energy[d] *= 0.20

        # 4. Trích xuất Bộ số Move 37 (Lựa chọn theo ngưỡng Bất định tối ưu)
        sorted_indices = np.argsort(combined_energy)[::-1]
        
        N1 = sorted_indices[0] + 1 # Điểm nút Năng lượng Move 37 cao nhất
        N2 = sorted_indices[1] + 1 # Điểm nút Giao thoa Markov
        N3 = sorted_indices[2] + 1 # Điểm nút Cân bằng Entropy

        bo_bac_2 = tuple(sorted([N1, N2]))
        bo_bac_3 = tuple(sorted([N1, N2, N3]))

        return bo_bac_2, bo_bac_3, N1, N2, N3, (s_entropy, s_markov, s_wavelet), combined_energy

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("♟️ Move-37 Micro-Chunk Engine v3")
st.caption("Khải phóng tư duy Move 37: Chia nhỏ Vi cửa sổ • Xích Markov Collatz • Entropy Phi tuyến")

raw_input = st.text_area(
    "Dán dữ liệu từ 5 đến 10 kỳ Keno vào đây:",
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
                
        engine = Move37EngineV3()
        bo2, bo3, n1, n2, n3, sub_scores, combined = engine.process(matrix)
        
        st.success(f"⚡ Đã xử lý {kies_to_use} kỳ bằng thuật toán Chia nhỏ Vi cửa sổ (Micro-Windowing)!")
        
        st.markdown("---")
        st.subheader("🎯 TỔNG HỢP BỘ SỐ BẤT ĐỊNH MOVE 37 (V3)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("MOVE 37 CORE (N1)", f"{n1:02d}")
        with c2:
            st.metric("MARKOV NODE (N2)", f"{n2:02d}")
        with c3:
            st.metric("ENTROPY BALANCER (N3)", f"{n3:02d}")
            
        col2, col3 = st.columns(2)
        with col2:
            st.subheader("BỘ BẬC 2 CHỐT")
            st.title(f"{bo2[0]:02d} — {bo2[1]:02d}")
        with col3:
            st.subheader("BỘ BẬC 3 CHỐT")
            st.title(f"{bo3[0]:02d} — {bo3[1]:02d} — {bo3[2]:02d}")
            
        st.markdown("---")
        st.subheader("📊 Mức Năng Lượng Vi Mô 3 Tầng")
        df_res = pd.DataFrame({
            "Con số": [f"Số {n1:02d}", f"Số {n2:02d}", f"Số {n3:02d}"],
            "Shannon Entropy (Micro-Window)": [f"{sub_scores[0][n1-1]:.3f}", f"{sub_scores[0][n2-1]:.3f}", f"{sub_scores[0][n3-1]:.3f}"],
            "Collatz-Markov State": [f"{sub_scores[1][n1-1]:.3f}", f"{sub_scores[1][n2-1]:.3f}", f"{sub_scores[1][n3-1]:.3f}"],
            "Wavelet Singularity": [f"{sub_scores[2][n1-1]:.3f}", f"{sub_scores[2][n2-1]:.3f}", f"{sub_scores[2][n3-1]:.3f}"],
            "Tích Năng Lượng Geometric Mean": [f"{combined[n1-1]:.3f}", f"{combined[n2-1]:.3f}", f"{combined[n3-1]:.3f}"]
        })
        st.table(df_res)
    else:
        msg_err = "Cần tối thiểu 5 kỳ dữ liệu (100 số). Hiện tại đọc được " + str(total_kies) + " kỳ."
        st.warning(msg_err)
else:
    st.info("Dán chuỗi 5–10 kỳ Keno vào khung văn bản phía trên để khởi chạy mô hình Move 37 v3.")
