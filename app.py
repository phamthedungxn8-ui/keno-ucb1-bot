import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v4.0 Kuramoto-Percolation Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v4.0: KURAMOTO PHASE SYNCHRONIZATION & PERCOLATION
# ==============================================================================
class MDM_IDS_Bac2_v4:
    def __init__(self, dim=80):
        self.D = dim

    def calculate_kuramoto_phase(self, X):
        """1. ĐỒNG BỘ HÓA PHA KURAMOTO (Phát hiện cặp lệch pha/trùng pha)"""
        T, D = X.shape
        phases = np.zeros(D)
        for d in range(D):
            history = X[:, d]
            # Tính góc pha theta dựa trên khoảng cách giữa các lần nổ
            seen_indices = np.where(history == 1)[0]
            if len(seen_indices) > 1:
                gaps = np.diff(seen_indices)
                mean_gap = np.mean(gaps)
                last_seen = T - 1 - seen_indices[-1]
                phases[d] = (2 * np.pi * last_seen) / (mean_gap + 1e-5)
            else:
                phases[d] = np.pi # Pha trung tính
        return phases

    def build_percolation_network(self, X):
        """2. LÝ THUYẾT PERCOLATION (Mạng lưới điểm thấm)"""
        T, D = X.shape
        adj_matrix = np.dot(X.T, X) / float(T) # Ma trận kề liên kết
        
        # Lực thấm lan truyền (Percolation Centrality)
        percolation_scores = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                # Khả năng thấm qua nút trung gian k
                shared_neighbors = np.sum(adj_matrix[i, :] * adj_matrix[j, :])
                percolation_scores[i, j] = adj_matrix[i, j] + 0.5 * shared_neighbors
                percolation_scores[j, i] = percolation_scores[i, j]
                
        return percolation_scores

    def process_bac2_v4(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Tính toán Pha Kuramoto và Mạng lưới Thấm Percolation
        phases = self.calculate_kuramoto_phase(X)
        percolation = self.build_percolation_network(X)
        
        # 2. Dựng Ma trận Tương quan Đồng xuất hiện đã Lọc Nhiễu SVD
        rho_raw = np.dot(X.T, X) / float(T)
        U, S, Vt = np.linalg.svd(rho_raw)
        S_clean = np.zeros_like(S)
        k_keep = max(1, int(len(S) * 0.20)) # Giữ 20% trị riêng chính
        S_clean[:k_keep] = S[:k_keep]
        rho_svd = np.dot(U, np.dot(np.diag(S_clean), Vt))
        
        # 3. CHẤM ĐIỂM CHI TIẾT TẤT CẢ 3,160 CẶP BẬC 2
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                # A. Điểm Đồng bộ Pha Kuramoto: cos(theta_i - theta_j) gần 1 là đồng bộ cực đại
                phase_diff = abs(phases[i] - phases[j])
                sync_score = np.cos(phase_diff)
                
                # B. Điểm Thấm Mạng lưới Percolation
                perc_score = percolation[i, j]
                
                # C. Điểm Tương quan SVD
                svd_score = rho_svd[i, j]
                
                # Tổng điểm chưa phạt
                score = (perc_score * 4.0) + (sync_score * 3.0) + (svd_score * 2.5)
                
                # --- BỘ LỌC KHẮC PHỤC TRÚNG 1/2 (SIẾT CHẶT LOẠI BỎ SỐ LỆCH) ---
                # Phạt nếu 1 trong 2 số là số bị bão hòa (về >= 3 lần trong 5 kỳ)
                if freqs[i] >= 3 or freqs[j] >= 3:
                    score *= 0.08
                    
                # Phạt cực nặng nếu CẢ HAI SỐ vừa nổ ở kỳ gần nhất T-1
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.01
                    
                # Phạt nếu cặp này trùng với nhịp nổ cách 1 kỳ
                if T >= 3 and X[-1, i] == 1 and X[-3, j] == 1:
                    score *= 0.05
                if T >= 3 and X[-1, j] == 1 and X[-3, i] == 1:
                    score *= 0.05
                    
                # Phạt cặp số quá kề nhau (Gap = 1) mà không có lực thấm hỗ trợ
                if abs(i - j) == 1 and perc_score < 0.2:
                    score *= 0.20
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp danh sách cặp Bậc 2
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # Lọc 3 cặp phụ độc lập không gian triệt để với Best Pair
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - MDM-IDS v4.0
# ==============================================================================
st.title("🎯 MDM-IDS v4.0: BẬC 2 KURAMOTO & PERCOLATION")
st.caption("Đồng bộ hóa Pha Kuramoto • Mạng lưới Thấm Percolation • Lọc Không gian Riemann")

raw_input = st.text_area(
    "Dán dữ liệu 5 đến 10 kỳ Keno gần nhất vào đây:",
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
                
        engine = MDM_IDS_Bac2_v4()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2_v4(matrix)
        
        st.success(f"⚡ Đã hoàn tất quét Kuramoto & Percolation trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 ĐỒNG BỘ PHA TỐI ƯU (CHỐT 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT ĐỒNG BỘ N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT THẤM MẠNG N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 15px; background-color: #1E1E1E; border-radius: 10px; border: 2px solid #00FFAA;'>"
            f"<h1 style='color: #00FFAA; margin:0;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #AAA; margin:0;'>Mức độ đồng bộ pha Kuramoto: {best_score:.5f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 DỰ PHÒNG CHUẨN PHA")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 CHI TIẾT TOP 10 CẶP BẬC 2 ĐỒNG BỘ CAO NHẤT")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Điểm Lực Thấm & Đồng Bộ": [f"{all_sorted[i][1]:.6f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ lọc Bậc 2 v4.0.")
