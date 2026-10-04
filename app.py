import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS Quantum-Classical Engine", layout="centered")

# ==============================================================================
# PHƯƠNG PHÁP MỚI: MULTILAYER DENSITY MATRIX & INFORMATION DYNAMICS (MDM-IDS)
# ==============================================================================
class MDM_IDSEngine:
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

    def svd_denoise(self, matrix, keep_ratio=0.3):
        """Lọc nhiễu trắng ngẫu nhiên bằng phân rã SVD"""
        U, S, Vt = np.linalg.svd(matrix)
        k = max(1, int(len(S) * keep_ratio))
        S_filtered = np.zeros_like(S)
        S_filtered[:k] = S[:k]
        return np.dot(U, np.dot(np.diag(S_filtered), Vt))

    def process(self, X):
        T, D = X.shape
        
        # Dựng H và Rho
        H = self.build_hamiltonian(X)
        rho_raw = self.build_density_matrix(X)
        
        # Tối ưu hóa SVD Lọc nhiễu
        rho = self.svd_denoise(rho_raw, keep_ratio=0.25)
        
        # Tiến hóa Von Neumann: [H, rho] = H*rho - rho*H
        comm = np.dot(H, rho) - np.dot(rho, H)
        evolution_energy = np.abs(np.diag(comm)) + np.diag(rho)
        
        # Bộ tiêu tán Entropy (Dissipation Operator): Phạt số bão hòa & Bẫy gan
        freqs = X.sum(axis=0)
        for d in range(D):
            # Phạt bão hòa (về 2 kỳ liên tiếp)
            if T >= 2 and X[-1, d] == 1 and X[-2, d] == 1:
                evolution_energy[d] *= 0.02
            # Phạt số gan tuyệt đối (> 5 kỳ chưa về)
            if freqs[d] == 0:
                evolution_energy[d] *= 0.15
                
        # Trích xuất 3 Nút Năng Lượng Đa Tầng
        sorted_indices = np.argsort(evolution_energy)[::-1]
        
        # N1: Nút năng lượng tiến hóa cực đại
        best_n1 = sorted_indices[0]
        
        # N2: Nút năng lượng cao thứ 2 có tương quan không gian nhỏ nhất với N1
        best_n2 = sorted_indices[1]
        for idx in sorted_indices[1:]:
            if rho[best_n1, idx] < np.median(rho[best_n1]): # Chọn nút độc lập không gian
                best_n2 = idx
                break
                
        # N3: Nút cân bằng entropy
        best_n3 = sorted_indices[2]
        for idx in sorted_indices[2:]:
            if idx != best_n1 and idx != best_n2:
                if rho[best_n1, idx] < np.median(rho[best_n1]) and rho[best_n2, idx] < np.median(rho[best_n2]):
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
st.title("🌌 MDM-IDS Quantum-Classical Engine")
st.caption("Phương pháp mới: Ma trận Mật độ Tương quan Không gian • Tiến hóa Von Neumann • Lọc Denoise SVD")

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
                
        engine = MDM_IDSEngine()
        bo2, bo3, n1, n2, n3, energy_spectrum = engine.process(matrix)
        
        st.success(f"⚡ Đã thực thi phương pháp MDM-IDS với thuật toán Lọc SVD trên {kies_to_use} kỳ!")
        
        st.markdown("---")
        st.subheader("🎯 TỔNG HỢP BỘ SỐ CHỐT MDM-IDS")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("HAMILTON NODE (N1)", f"{n1:02d}")
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
        st.subheader("📊 Năng lượng Tiến hóa Lượng tử Cổ điển")
        df_res = pd.DataFrame({
            "Con số": [f"Số {n1:02d} (N1)", f"Số {n2:02d} (N2)", f"Số {n3:02d} (N3)"],
            "Điểm Năng lượng Tiến hóa (Von Neumann Energy)": [
                f"{energy_spectrum[n1-1]:.4f}", 
                f"{energy_spectrum[n2-1]:.4f}", 
                f"{energy_spectrum[n3-1]:.4f}"
            ]
        })
        st.table(df_res)
    else:
        msg_err = "Cần tối thiểu 5 kỳ dữ liệu (100 số). Hiện tại đọc được " + str(total_kies) + " kỳ."
        st.warning(msg_err)
else:
    st.info("Dán chuỗi 5–10 kỳ Keno vào khung văn bản phía trên để khởi chạy mô hình MDM-IDS.")
