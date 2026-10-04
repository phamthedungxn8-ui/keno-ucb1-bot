import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v12.0 Free-Field Topo Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v12.0: FREE-FIELD NON-EUCLIDEAN TOPO ENGINE
# ==============================================================================
class MDM_IDS_Bac2_v12_FreeField:
    def __init__(self, dim=80):
        self.D = dim

    def compute_free_mutual_information_matrix(self, X):
        """1. TRƯỜNG ENTROPY SHANNON TƯƠNG HỖ TỰ DO (Full 80x80 Matrix)"""
        T, D = X.shape
        mi_matrix = np.zeros((D, D))
        
        # Tính xác suất biên
        p1 = np.mean(X == 1, axis=0) + 1e-9
        p0 = np.mean(X == 0, axis=0) + 1e-9
        
        for i in range(D):
            for j in range(i + 1, D):
                x, y = X[:, i], X[:, j]
                p_11 = np.mean((x == 1) & (y == 1)) + 1e-9
                p_10 = np.mean((x == 1) & (y == 0)) + 1e-9
                p_01 = np.mean((x == 0) & (y == 1)) + 1e-9
                p_00 = np.mean((x == 0) & (y == 0)) + 1e-9
                
                mi = (p_11 * np.log2(p_11 / (p1[i] * p1[j])) +
                      p_10 * np.log2(p_10 / (p1[i] * p0[j])) +
                      p_01 * np.log2(p_01 / (p0[i] * p1[j])) +
                      p_00 * np.log2(p_00 / (p0[i] * p0[j])))
                
                val = max(0.0, mi)
                mi_matrix[i, j] = val
                mi_matrix[j, i] = val
                
        return mi_matrix

    def compute_riemannian_geodesic_field(self, X):
        """2. KHÔNG GIANG CONG RIEMANN PHI TUYẾN 4D"""
        T, D = X.shape
        norms = np.linalg.norm(X, axis=0) + 1e-5
        dot_products = np.dot(X.T, X)
        
        # Matrix Cosine similarity
        cos_sim = dot_products / np.outer(norms, norms)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # Khoảng cách Geodesic trên mặt phẳng cong
        geodesic_matrix = np.arccos(cos_sim)
        return geodesic_matrix

    def compute_tensor_mps_entanglement(self, X):
        """3. ĐỘ VƯỚNG VÍU MẠNG TENSOR MPS DÙNG SVD"""
        T, D = X.shape
        if T >= 4:
            split = T // 2
            X_past, X_recent = X[:split, :], X[split:, :]
            cov_past = np.dot(X_past.T, X_past) / float(split)
            cov_recent = np.dot(X_recent.T, X_recent) / float(T - split)
            
            # Sức căng vướng víu Tensor qua 2 phân đoạn thời gian
            mps_matrix = np.abs(cov_past * cov_recent)
        else:
            mps_matrix = np.dot(X.T, X) / float(T)
            
        return mps_matrix

    def process_bac2_v12(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Khởi chạy 3 Luồng Toán học Nâng cao
        mi_mat = self.compute_free_mutual_information_matrix(X)
        geo_mat = self.compute_riemannian_geodesic_field(X)
        mps_mat = self.compute_tensor_mps_entanglement(X)
        
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                # Factor A: Entropy Tin tức Shannon Tương hỗ
                f_mi = mi_mat[i, j]
                
                # Factor B: Mức độ tương đồng không gian cong Riemann Topo
                f_topo = np.exp(-geo_mat[i, j])
                
                # Factor C: Độ vướng víu Mạng Tensor MPS
                f_mps = mps_mat[i, j]
                
                # TÍCH CHÉO ĐA LUỒNG TỰ DO PHI TUYẾN
                score = (f_mi ** 1.8) * (f_topo ** 1.5) * (f_mps ** 1.2)
                
                # --- LỌC TRIỆT TIÊU NHIỄU & BẪY TRÚNG 1/2 ---
                # Phạt số bão hòa tần suất (>= 3 lần nổ trong cửa sổ)
                if freqs[i] >= 3: score *= 0.001
                if freqs[j] >= 3: score *= 0.001
                
                # Phạt nghẽn mạch (Cả 2 số cùng xuất hiện ở kỳ T-1)
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.0001
                    
                # Phạt lệch nhịp đối xứng (Kỳ T-1 và T-3)
                if T >= 3 and X[-1, i] == 1 and X[-3, j] == 1:
                    score *= 0.001
                if T >= 3 and X[-1, j] == 1 and X[-3, i] == 1:
                    score *= 0.001
                    
                # Phạt số gan bão hòa âm (0 lần xuất hiện)
                if freqs[i] == 0 and freqs[j] == 0:
                    score *= 0.001
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp danh sách tất cả 3.160 cặp Bậc 2
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # Khai thác 3 cặp phụ hoàn toàn độc lập về mặt không gian số
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - MDM-IDS v12.0
# ==============================================================================
st.title("🌐 MDM-IDS v12.0: FREE-FIELD TOPO ENGINE")
st.caption("Quét Tự Do 3.160 Cặp Bậc 2 • Shannon Mutual Information • Độ Cong Riemann 4D • Mạng Tensor MPS")

raw_input = st.text_area(
    "Dán dữ liệu 5 đến 10 kỳ Keno mới nhất vào đây:",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
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
                
        engine = MDM_IDS_Bac2_v12_FreeField()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2_v12(matrix)
        
        st.success(f"⚡ Đã quét xong trường tự do Topo v12.0 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 ĐỒNG THUẬN TOPO TỰ DO TỐI ƯU NHẤT (CHỐT 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 18px; background-color: #051911; border-radius: 12px; border: 2px solid #00FF88; box-shadow: 0 0 15px rgba(0, 255, 136, 0.4);'>"
            f"<h1 style='color: #00FF88; margin:0; font-size: 2.8rem;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:5px 0 0 0;'>Cường độ vướng víu trường tự do v12.0: {best_score:.8f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 DỰ PHÒNG TỰ DO CHUẨN TOPO")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 BẢNG TOP 10 CẶP BẬC 2 CÓ ĐỘ HỘI TỤ MẠNG TOPO CAO NHẤT")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2 Tự Do": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Điểm Tích Chéo v12.0": [f"{all_sorted[i][1]:.8f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ quét trường tự do v12.0.")
