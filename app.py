import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v6.0 Quantum Phase Singularity", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v6.0: ULTIMATE PHYSICAL LIMIT (QPS ENGINE)
# ==============================================================================
class MDM_IDS_Bac2_v6_Ultimate:
    def __init__(self, dim=80):
        self.D = dim

    def apply_rmt_marchenko_pastur(self, X):
        """1. LÝ THUYẾT MA TRẬN NGẪU NHIÊN WIGNER & RMT (Lọc sạch nhiễu nền)"""
        T, D = X.shape
        rho_raw = np.dot(X.T, X) / float(T)
        
        # Ngưỡng RMT Marchenko-Pastur Boundary
        q = D / float(T) if T > 0 else 1.0
        sigma2 = 1.0
        lambda_max = sigma2 * ((1.0 + np.sqrt(q)) ** 2)
        
        # Phân rã Trị riêng (Eigendecomposition)
        eigenvalues, eigenvectors = np.linalg.eigh(rho_raw)
        
        # Chỉ giữ lại các trị riêng vượt khỏi ngưỡng nhiễu ngẫu nhiên RMT
        clean_eigenvalues = np.where(eigenvalues > lambda_max, eigenvalues, 0.0)
        
        # Tái thiết lập ma trận mật độ sạch
        rho_rmt_clean = np.dot(eigenvectors, np.dot(np.diag(clean_eigenvalues), eigenvectors.T))
        return rho_rmt_clean, clean_eigenvalues

    def compute_bkt_vortex_pairing(self, i, j, X):
        """2. CHUYỂN PHA BKT (Vortex - Antivortex Pair Coupling)"""
        T, D = X.shape
        # Chuyển đổi chỉ số thành tọa độ 2D trên lưới 8x10
        r1, c1 = i // 10, i % 10
        r2, c2 = j // 10, j % 10
        
        # Vectör khoảng cách không gian
        dr = float(r1 - r2)
        dc = float(c1 - c2)
        r_dist = np.sqrt(dr**2 + dc**2) + 1e-5
        
        # Vấn đề xoáy BKT: Năng lượng liên kết tỷ lệ với log(r)
        vortex_energy = np.log(r_dist) * (1.0 if r_dist <= 4.5 else 0.2)
        return vortex_energy

    def compute_cft_conformal_cross_ratio(self, i, j):
        """3. LÝ THUYẾT TRƯỜNG CONFORMAL (Cross-Ratio Invariance)"""
        # Đánh giá tính đẳng hình không gian (Moduli Space)
        mod10_i, mod10_j = (i + 1) % 10, (j + 1) % 10
        mod8_i, mod8_j = (i + 1) % 8, (j + 1) % 8
        
        cross_ratio = abs((mod10_i - mod10_j) + 1e-5) / abs((mod8_i - mod8_j) + 1e-5 + 1)
        # Điểm hội tụ Conformal
        conformal_score = np.exp(-((cross_ratio - 1.618)**2) / 0.5) # Tiệm cận tỷ lệ vàng
        return conformal_score

    def process_bac2_v6(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Khai thác RMT Marchenko-Pastur
        rho_rmt, clean_eigs = self.apply_rmt_marchenko_pastur(X)
        
        # 2. Entropy Tổng lượng tử
        total_signal_energy = np.sum(clean_eigs) + 1e-5
        
        # CHẤM ĐIỂM GIỚI HẠN VẬT LÝ CHO ALL 3,160 CẶP BẬC 2
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                # A. Điểm liên kết Ma trận RMT Sạch
                rmt_coupling = rho_rmt[i, j]
                
                # B. Năng lượng liên kết Xoáy BKT
                bkt_weight = self.compute_bkt_vortex_pairing(i, j, X)
                
                # C. Điểm đồng hình CFT Conformal
                cft_weight = self.compute_cft_conformal_cross_ratio(i, j)
                
                # Tổng điểm năng lượng vật lý chưa phạt
                score = (rmt_coupling * 5.0) + (bkt_weight * 3.0) + (cft_weight * 2.0)
                
                # --- SIẾT CHẶT QUY TẮC TRIỆT HẠ BẪY SAI SỐ VI MÔ ---
                # Phạt số bão hòa tần suất (>= 3 lần xuất hiện trong cửa sổ)
                if freqs[i] >= 3: score *= 0.02
                if freqs[j] >= 3: score *= 0.02
                
                # Phạt cực nặng triệt để nếu CẢ HAI SỐ vừa cùng nổ ở kỳ T-1
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.001
                    
                # Phạt bẫy nổ nhịp đối xứng cách kỳ (Kỳ T-1 và T-3)
                if T >= 3 and X[-1, i] == 1 and X[-3, j] == 1:
                    score *= 0.01
                if T >= 3 and X[-1, j] == 1 and X[-3, i] == 1:
                    score *= 0.01
                    
                # Phạt hai số cùng nổ ở kỳ T-2 mà không có liên kết RMT mạnh
                if T >= 2 and X[-2, i] == 1 and X[-2, j] == 1 and rmt_coupling < 0.1:
                    score *= 0.05
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp danh sách kết quả theo năng lượng sụp đổ giảm dần
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # Khai thác 3 cặp dự phòng có tính độc lập không gian tuyệt đối
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - MDM-IDS v6.0 ULTIMATE
# ==============================================================================
st.title("⚛️ MDM-IDS v6.0: QUANTUM PHASE SINGULARITY")
st.caption("Giới hạn Vật lý: Lọc Ma trận Ngẫu nhiên RMT • Chuyển pha BKT • Trường Conformal CFT")

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
                
        engine = MDM_IDS_Bac2_v6_Ultimate()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2_v6(matrix)
        
        st.success(f"⚡ Đã hoàn tất xử lý Trạng thái Kỳ dị Pha v6.0 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 CHỐT CÓ XÁC SUẤT SỤP ĐỔ SÓNG HÀM CỰC ĐẠI")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT KỲ DỊ N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT KỲ DỊ N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 18px; background-color: #050505; border-radius: 12px; border: 2px solid #FF0055; box-shadow: 0 0 15px rgba(255, 0, 85, 0.4);'>"
            f"<h1 style='color: #FF0055; margin:0; font-size: 2.8rem;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:5px 0 0 0;'>Cường độ kỳ dị lượng tử QPS: {best_score:.6f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 DỰ PHÒNG CHUYỂN PHA BKT")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 BẢNG NĂNG LƯỢNG SỤP ĐỔ TOP 10 CẶP BẬC 2 HIGHEST DENSITY")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Số Bậc 2": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Chỉ Số Kỳ Dị Năng Lượng QPS": [f"{all_sorted[i][1]:.6f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ lọc QPS v6.0.")
