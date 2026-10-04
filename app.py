import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS Quantum-Classical Engine v1.1", layout="centered")

# ==============================================================================
# PHƯƠNG PHÁP CẢI TIẾN: MDM-IDS v1.1 (OPTIMIZED DENSITY MATRIX & DISSIPATION)
# ==============================================================================
class MDM_IDSEngineV1_1:
    def __init__(self, dim=80):
        self.D = dim

    def build_hamiltonian(self, X):
        """Dựng toán tử Hamilton H chứa năng lượng Collatz & Wavelet"""
        T, D = X.shape
        freqs = X.sum(axis=0).astype(float)
        
        # 1. Năng lượng tự do Collatz 2-adic trên đường chéo
        H_diag = np.zeros(D)
        for d in range(D):
            v = int(freqs[d])
            if v > 0:
                collatz_v = 3 * v + 1
                tz = (collatz_v & -collatz_v).bit_length() - 1
                H_diag[d] = tz * 1.618 + np.log2(v + 1)
            else:
                H_diag[d] = 0.5 # Mức năng lượng chấn động nền cho số chưa xuất hiện
                
        # 2. Tương tác vành đồng dư Modulo Z/8Z trên các phần tử ngoài đường chéo
        H_interaction = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                if (i + 1) % 8 == (j + 1) % 8:
                    H_interaction[i, j] = 0.25
                    H_interaction[j, i] = 0.25
                    
        H = np.diag(H_diag) + H_interaction
        return H

    def build_density_matrix(self, X):
        """Dựng ma trận mật độ tương quan không gian rho"""
        T, D = X.shape
        rho = np.zeros((D, D))
        for t in range(T):
            v = X[t].reshape(-1, 1)
            rho += np.dot(v, v.T)
        rho /= T
        return rho

    def svd_denoise(self, matrix, keep_ratio=0.25):
        """Lọc nhiễu trắng ngẫu nhiên bằng phân rã SVD"""
        U, S, Vt = np.linalg.svd(matrix)
        k = max(1, int(len(S) * keep_ratio))
        S_filtered = np.zeros_like(S)
        S_filtered[:k] = S[:k]
        return np.dot(U, np.dot(np.diag(S_filtered), Vt))

    def process(self, X):
        T, D = X.shape
        
        # 1. Dựng toán tử Hamilton H và Ma trận Mật độ Rho
        H = self.build_hamiltonian(X)
        rho_raw = self.build_density_matrix(X)
        
        # 2. Tối ưu hóa SVD Lọc nhiễu
        rho = self.svd_denoise(rho_raw, keep_ratio=0.25)
        
        # 3. Tiến hóa Von Neumann: [H, rho] = H*rho - rho*H
        comm = np.dot(H, rho) - np.dot(rho, H)
        evolution_energy = np.abs(np.diag(comm)) + np.diag(rho)
        
        # 4. BỘ TIÊU TÁN ENTROPY NÂNG CAO (SIẾT CHẶT LOẠI BỎ BẪY MẬT ĐỘ)
        freqs = X.sum(axis=0)
        for d in range(D):
            # A. Phạt nặng số xuất hiện quá dày (Mật độ >= 3 lần trong 5 kỳ - như số 20)
            if freqs[d] >= 3:
                evolution_energy[d] *= 0.01

            # B. Phạt bão hòa nổ 2 kỳ liên tiếp gần nhất
            if T >= 2 and X[-1, d] == 1 and X[-2, d] == 1:
                evolution_energy[d] *= 0.02

            # C. Phạt số nổ cách 1 kỳ (như số 73 xuất hiện kỳ 3 và kỳ 5)
            if T >= 3 and X[-1, d] == 1 and X[-3, d] == 1:
                evolution_energy[d] *= 0.05

            # D. Phạt bẫy gan tuyệt đối (0 lần xuất hiện trong tập cửa sổ)
            if freqs[d] == 0:
                evolution_energy[d] *= 0.15

        # 5. TRÍCH XUẤT NÚT NĂNG LƯỢNG ĐỘC LẬP KHÔNG GIAN TRIỆT ĐỂ
        sorted_indices = np.argsort(evolution_energy)[::-1]
        
        # N1: Nút năng lượng tiến hóa cực đại đạt chuẩn
        best_n1 = sorted_indices[0]
        
        # N2: Nút có tương quan không gian độc lập nhất với N1
        best_n2 = sorted_indices[1]
        for idx in sorted_indices[1:]:
            if rho[best_n1, idx] < np.percentile(rho[best_n1], 40): # Siết ngưỡng độc lập < 40%
                best_n2 = idx
                break
                
        # N3: Nút cân bằng entropy có tương quan thấp với cả N1 và N2
        best_n3 = sorted_indices[2]
        for idx in sorted_indices[2:]:
            if idx != best_n1 and idx != best_n2:
                cond1 = rho[best_n1, idx] < np.percentile(rho[best_n1], 40)
                cond2 = rho[best_n2, idx] < np.percentile(rho[best_n2], 40)
                if cond1 and cond2:
                    best_n3 = idx
                    break

        N1 = best_n1 + 1
        N2 = best_n2 + 1
        N3 = best_n3 + 1

        bo_bac_2 = tuple(sorted([N1, N2]))
        bo_bac_3 = tuple(sorted([N1, N2, N3]))

        return bo_bac_2, bo_bac_3, N1, N2, N3, evolution_energy

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("🌌 MDM-IDS Quantum-Classical Engine v1.1")
st.caption("Phương pháp tối ưu: Tiêu tán Mật độ Đa kỳ • Lọc SVD • Khai thác Nút Độc lập Không gian")

raw_input = st.text_area(
    "Dán dữ liệu 5 đến 10 kỳ Keno vào đây:",
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
                
        engine = MDM_IDSEngineV1_1()
        bo2, bo3, n1, n2, n3, energy_spectrum = engine.process(matrix)
        
        st.success(f"⚡ Đã xử lý {kies_to_use} kỳ bằng thuật toán MDM-IDS v1.1 tối ưu!")
        
        st.markdown("---")
        st.subheader("🎯 TỔNG HỢP BỘ SỐ CHỐT NĂNG LƯỢNG MỚI")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("HAMILTON CORE (N1)", f"{n1:02d}")
        with c2:
            st.metric("DECOHERENCE BALANCER (N2)", f"{n2:02d}")
        with c3:
            st.metric("SPATIAL ATTRACTOR (N3)", f"{n3:02d}")
            
        col2, col3 = st.columns(2)
        with col2:
            st.subheader("BỘ BẬC 2 CHỐT")
            st.title(f"{bo2[0]:02d} — {bo2[1]:02d}")
        with col3:
            st.subheader("BỘ BẬC 3 CHỐT")
            st.title(f"{bo3[0]:02d} — {bo3[1]:02d} — {bo3[2]:02d}")
            
        st.markdown("---")
        st.subheader("📊 Mức Năng Lượng Tiến Hóa Sau Khi Triệt Tiêu Nhiễu Bão Hòa")
        df_res = pd.DataFrame({
            "Con số": [f"Số {n1:02d} (N1)", f"Số {n2:02d} (N2)", f"Số {n3:02d} (N3)"],
            "Điểm Năng Lượng Chốt": [
                f"{energy_spectrum[n1-1]:.5f}", 
                f"{energy_spectrum[n2-1]:.5f}", 
                f"{energy_spectrum[n3-1]:.5f}"
            ]
        })
        st.table(df_res)
    else:
        msg_err = "Cần tối thiểu 5 kỳ dữ liệu (100 số). Hiện tại đọc được " + str(total_kies) + " kỳ."
        st.warning(msg_err)
else:
    st.info("Dán chuỗi 5–10 kỳ Keno vào khung văn bản phía trên để khởi chạy mô hình MDM-IDS v1.1.")
