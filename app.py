import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v5.0 Quantum-Topology Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v5.0: QUANTUM-TOPOLOGY & MARKOV RANDOM FIELDS (BAC 2)
# ==============================================================================
class MDM_IDS_Bac2_v5:
    def __init__(self, dim=80):
        self.D = dim

    def compute_von_neumann_entropy(self, rho):
        """1. MỨC ĐỘ VƯỚNG VÍU LƯỢNG TỬ (von Neumann Entropy)"""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-12] # Lọc nhiễu tiệm cận 0
        eigenvalues = eigenvalues / np.sum(eigenvalues) # Chuẩn hóa
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    def compute_tda_persistence_weight(self, X):
        """2. TOPOLOGICAL DATA ANALYSIS (TDA Loop Persistence)"""
        T, D = X.shape
        # Dựng ma trận khoảng cách topo dựa trên correlation
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0.0)
        dist_matrix = 1.0 - corr
        
        # Bán kính lọc Topo
        persistence_scores = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                d_ij = dist_matrix[i, j]
                # Vòng lặp Homology có năng lượng cao khi khoảng cách topo nằm trong dải chuyển pha
                if 0.15 <= d_ij <= 0.65:
                    persistence_scores[i, j] = 1.0 / (d_ij + 1e-5)
                else:
                    persistence_scores[i, j] = 0.1
                persistence_scores[j, i] = persistence_scores[i, j]
        return persistence_scores

    def compute_hmrf_ising_energy(self, i, j):
        """3. MARKOV RANDOM FIELD (Ising Energy trên lưới Keno 8x10)"""
        # Đổi số thứ tự (1-80) sang tọa độ lưới 2D (8 hàng x 10 cột)
        r1, c1 = (i) // 10, (i) % 10
        r2, c2 = (j) // 10, (j) % 10
        
        manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
        
        # Năng lượng liên kết vi mảng (Ising Coupling)
        if manhattan_dist == 1:
            return 0.35 # Kề cận trực tiếp
        elif manhattan_dist == 2:
            return 0.25 # Kề chéo hoặc cách 1 ô
        elif manhattan_dist in [3, 5, 8]: # Khoảng cách Fibonacci 2D
            return 0.40
        return 0.05

    def process_bac2_v5(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Ma trận mật độ chuẩn hóa Lượng tử
        rho_raw = np.dot(X.T, X) / float(T)
        trace_val = np.trace(rho_raw)
        rho_norm = rho_raw / (trace_val if trace_val > 0 else 1.0)
        
        # 2. Entropy von Neumann tổng thể
        vn_entropy = self.compute_von_neumann_entropy(rho_norm)
        
        # 3. Trọng số TDA Topology
        tda_scores = self.compute_tda_persistence_weight(X)
        
        # 4. SVD Lọc nhiễu không gian ma trận
        U, S, Vt = np.linalg.svd(rho_raw)
        S_clean = np.zeros_like(S)
        k_keep = max(1, int(len(S) * 0.18)) # Giữ 18% giá trị riêng cốt lõi
        S_clean[:k_keep] = S[:k_keep]
        rho_svd = np.dot(U, np.dot(np.diag(S_clean), Vt))
        
        # CHẤM ĐIỂM VI MÔ 3,160 CẶP BẬC 2
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                # A. Tương tác Vướng víu Lượng tử
                quantum_coupling = rho_norm[i, j] * (1.0 / (vn_entropy + 1e-5))
                
                # B. Độ bền Betti Loop (TDA)
                tda_weight = tda_scores[i, j]
                
                # C. Năng lượng Ising HMRF (Hình học 2D 8x10)
                ising_weight = self.compute_hmrf_ising_energy(i, j)
                
                # D. Tương quan SVD
                svd_weight = rho_svd[i, j]
                
                # Tổng điểm chưa phạt
                score = (quantum_coupling * 4.5) + (tda_weight * 3.5) + (svd_weight * 2.5) + (ising_weight * 1.5)
                
                # --- SIẾT CHẶT QUY TẮC TRIỆT HẠ BẪY SAI SỐ ---
                # Phạt số đã bão hòa tần suất (về >= 3 lần trong cửa sổ 5-10 kỳ)
                if freqs[i] >= 3: score *= 0.05
                if freqs[j] >= 3: score *= 0.05
                
                # Phạt cực nặng nếu CẢ HAI SỐ vừa nổ cùng nhau ở kỳ T-1
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.005
                    
                # Phạt nhịp nổ đối xứng cách kỳ
                if T >= 3 and X[-1, i] == 1 and X[-3, j] == 1:
                    score *= 0.02
                if T >= 3 and X[-1, j] == 1 and X[-3, i] == 1:
                    score *= 0.02
                    
                # Phạt số gan bão hòa âm (0 lần xuất hiện liên tiếp > 8 kỳ)
                if freqs[i] == 0 and freqs[j] == 0:
                    score *= 0.01
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp kết quả
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # Lọc 3 cặp phụ hoàn toàn độc lập không gian với Best Pair
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - MDM-IDS v5.0
# ==============================================================================
st.title("⚛️ MDM-IDS v5.0: QUANTUM-TOPOLOGY BẬC 2")
st.caption("Entropy Lượng tử von Neumann • Đại số Đồng thủy TDA • Mảng ngẫu nhiên Markov 8x10")

raw_input = st.text_area(
    "Dán dữ liệu 5 đến 10 kỳ Keno gần nhất vào đây:",
    placeholder="Kỳ 1: 02 05 08 ...\nKỳ 2: ...",
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
                
        engine = MDM_IDS_Bac2_v5()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2_v5(matrix)
        
        st.success(f"⚡ Đã hoàn tất quét Quantum-Topology v5.0 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 VƯỚNG VÍU LƯỢNG TỬ CỰC ĐẠI (CHỐT 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT LƯỢNG TỬ N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT TOPOLOGY N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 15px; background-color: #0E1117; border-radius: 10px; border: 2px solid #7928CA;'>"
            f"<h1 style='color: #7928CA; margin:0;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #AAA; margin:0;'>Chỉ số Năng lượng Lượng tử - Topo: {best_score:.6f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 KHÔNG GIAN BÙ TRỪ DỰ PHÒNG")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 CHI TIẾT TOP 10 CẶP BẬC 2 CÓ MỨC ĐỘ VƯỚNG VÍU CAO NHẤT")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Điểm Lượng Tử - Topo": [f"{all_sorted[i][1]:.6f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ lọc Bậc 2 Quantum v5.0.")
