import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v2.0 - Bậc 2 Dual-Core Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v2.0: TỐI ƯU HÓA KHÔNG GIAN BẬC 2 (DUAL-CORE PAIR)
# ==============================================================================
class MDM_IDS_Bac2_Engine:
    def __init__(self, dim=80):
        self.D = dim

    def build_duality_hamiltonian(self, X):
        """1. TRIẾT HỌC ÂM DƯƠNG & MA TRẬN HAMILTON (Động - Tĩnh)"""
        T, D = X.shape
        freqs = X.sum(axis=0).astype(float)
        
        # Năng lượng Tĩnh (Âm): Độ giãn cách nhịp chưa nổ (Nhiệt độ ngầm)
        # Năng lượng Động (Dương): Tần suất tích tụ gần nhất (Mật độ)
        static_energy = np.zeros(D)
        for d in range(D):
            # Tính kỳ gần nhất xuất hiện
            last_seen = np.where(X[:, d] == 1)[0]
            if len(last_seen) > 0:
                gap = (T - 1) - last_seen[-1]
            else:
                gap = T
            # Cân bằng Triết học: Nhịp nén vừa đủ (Gap từ 1 đến 3 kỳ) có năng lượng tích tụ cao nhất
            static_energy[d] = np.exp(-((gap - 2.0)**2) / 2.5) * 2.0 + 0.1
            
        # Ma trận tương tác Rối Lượng tử & Topo (Phase Entanglement)
        H_entangle = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                # Tương tácModulo 10 (Hàng đơn vị) & Modulo 8 (Tính đối xứng)
                mod10_match = ((i + 1) % 10 == (j + 1) % 10)
                mod8_match = ((i + 1) % 8 == (j + 1) % 8)
                
                # Khoảng cách Topo trên vành 80 số
                topo_dist = abs((i + 1) - (j + 1))
                
                weight = 0.0
                if mod10_match: weight += 0.35
                if mod8_match: weight += 0.25
                if 10 <= topo_dist <= 25: weight += 0.20 # Vùng khoảng cách vàng không gian
                
                H_entangle[i, j] = weight
                H_entangle[j, i] = weight
                
        H = np.diag(static_energy) + H_entangle
        return H, static_energy

    def build_cooccurrence_matrix(self, X):
        """2. MA TRẬN MẬT ĐỘ TƯƠNG QUAN ĐỒNG THỜI (Co-occurrence Density)"""
        T, D = X.shape
        co_matrix = np.zeros((D, D))
        for t in range(T):
            row = X[t]
            indices = np.where(row == 1)[0]
            for i in indices:
                for j in indices:
                    if i != j:
                        co_matrix[i, j] += 1.0
        return co_matrix / T

    def process_bac2(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Dựng Toán tử Hamilton Âm-Dương & Ma trận Tương quan Đồng thời
        H, static_energy = self.build_duality_hamiltonian(X)
        rho_co = self.build_cooccurrence_matrix(X)
        
        # 2. Phân rã SVD lọc nhiễu không gian ma trận tương quan Bậc 2
        U, S, Vt = np.linalg.svd(rho_co)
        S_filtered = np.zeros_like(S)
        # Giữ lại 30% thành phần tần số liên kết cao nhất
        k = max(1, int(len(S) * 0.30)) 
        S_filtered[:k] = S[:k]
        rho_filtered = np.dot(U, np.dot(np.diag(S_filtered), Vt))

        # 3. CHẤM ĐIỂM TẤT CẢ CÁC CẶP BẬC 2 (C3200 cặp trong 80 số)
        pair_scores = {}
        
        for i in range(D):
            for j in range(i + 1, D):
                # Tiêu chí A: Độ rối lượng tử từ ma trận Hamilton
                h_coupling = H[i, j]
                
                # Tiêu chí B: Điểm tương quan không gian đã lọc SVD
                co_score = rho_filtered[i, j]
                
                # Tiêu chí C: Cân bằng Âm Dương (Một nút nén + Một nút nhịp giao thoa)
                balance_score = static_energy[i] + static_energy[j]
                
                # Tổng điểm Bậc 2
                score = (co_score * 4.0) + (h_coupling * 2.5) + (balance_score * 1.5)
                
                # PHẠT BẮC BẪY MẬT ĐỘ (BẢO VỆ CẶP BẬC 2)
                # Phạt nếu cả 2 số đều đã xuất hiện >= 3 lần (bão hòa)
                if freqs[i] >= 3 and freqs[j] >= 3:
                    score *= 0.05
                # Phạt nếu cả 2 số chưa từng xuất hiện (gan cấm)
                if freqs[i] == 0 and freqs[j] == 0:
                    score *= 0.10
                # Phạt cặp vừa cùng về ở kỳ gần nhất
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.02
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp danh sách các cặp Bậc 2 tốt nhất
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Cặp Bậc 2 Tối ưu nhất (Top 1)
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # 3 Cặp Bậc 2 Dự phòng tốt nhất
        top_backup_pairs = [sorted_pairs[i][0] for i in range(1, 4)]
        
        return best_pair, best_score, top_backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - CHUYÊN BIỆT PHÂN TÍCH BẬC 2
# ==============================================================================
st.title("🎯 MDM-IDS v2.0: CHUYÊN PHÂN TÍCH BẬC 2")
st.caption("Ứng dụng Toán tử Hamilton Âm-Dương • Lọc Rối Lượng Tử SVD • Không Gian Betti Bậc 2")

raw_input = st.text_area(
    "Dán dữ liệu 5 đến 10 kỳ Keno vào đây:",
    placeholder="Kỳ 1: 01 03 06 ...\nKỳ 2: ...",
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
                
        engine = MDM_IDS_Bac2_Engine()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2(matrix)
        
        st.success(f"⚡ Đã quét toàn bộ 3,160 cặp Bậc 2 trên tập dữ liệu {kies_to_use} kỳ!")
        
        st.markdown("---")
        st.subheader("🔥 BỘ BẬC 2 TỐI ƯU NHẤT (CHỐT)")
        
        col_pair1, col_pair2 = st.columns(2)
        with col_pair1:
            st.metric("SỐ THỨ NHẤT (N1)", f"{best_pair[0]:02d}")
        with col_pair2:
            st.metric("SỐ THỨ HAI (N2)", f"{best_pair[1]:02d}")
            
        st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>", unsafe_allow_dict=True)
        st.caption(f"Trạng thái năng lượng rối không gian: {best_score:.4f}")

        st.markdown("---")
        st.subheader("🛡️ TOP 3 CẶP BẬC 2 DỰ PHÒNG (DƯƠNG BẢN BẤT ĐOẠN)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📋 BẢNG XẾP HẠNG TOP 10 CẶP BẬC 2 NĂNG LƯỢNG CAO NHẤT")
        
        df_top = pd.DataFrame({
            "Hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Số Bậc 2": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Điểm Tương Quan Không Gian": [f"{all_sorted[i][1]:.5f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hiện tại hệ thống nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán chuỗi dữ liệu Keno vào khung trên để tiến hành phân tích chiều sâu Bậc 2.")
