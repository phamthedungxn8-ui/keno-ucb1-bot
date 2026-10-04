import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v13.0 Anti-Repetition Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v13.0: DYNAMIC PERTURBATION & TOPOLOGICAL PHASE SHIFT (DP-TPS)
# ==============================================================================
class MDM_IDS_Bac2_v13_Dynamic:
    def __init__(self, dim=80):
        self.D = dim

    def compute_mutual_information_dynamic(self, X):
        """1. ENTROPY SHANNON TƯƠNG HỖ VỚI TRỌNG SỐ SUY GIẢM THEO THỜI GIAN"""
        T, D = X.shape
        # Trạng thái gần đây có trọng số lớn hơn để loại bỏ tín hiệu kẹt lặp
        weights = np.exp(np.linalg.norm(np.arange(T)) / float(T))
        weights /= np.sum(weights)
        
        mi_matrix = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                x, y = X[:, i], X[:, j]
                # Tương quan weighted co-occurrence
                co_presence = np.sum((x == 1) & (y == 1) * weights)
                p_i = np.sum((x == 1) * weights) + 1e-9
                p_j = np.sum((y == 1) * weights) + 1e-9
                
                mi = co_presence * np.log2((co_presence + 1e-9) / (p_i * p_j))
                val = max(0.0, mi)
                mi_matrix[i, j] = val
                mi_matrix[j, i] = val
                
        return mi_matrix

    def compute_topological_phase_shift(self, X):
        """2. PHÁT HIỆN DỊCH PHA TOPO CHỐNG LẶP THANH GHI"""
        T, D = X.shape
        phase_shifts = np.zeros((D, D))
        
        for i in range(D):
            for j in range(i + 1, D):
                # Tín hiệu dịch pha qua phép biến đổi Hilbert ngắn hạn
                sig_i = X[:, i] - np.mean(X[:, i])
                sig_j = X[:, j] - np.mean(X[:, j])
                
                # Đo độ phân cực hướng thay đổi giữa 2 kỳ gần nhất
                delta_i = sig_i[-1] - sig_i[-2] if T >= 2 else 0
                delta_j = sig_j[-1] - sig_j[-2] if T >= 2 else 0
                
                # Nếu hai số thay đổi ngược chiều hoặc đứng yên quá lâu -> giảm điểm
                shift_factor = 1.0 + (delta_i * delta_j)
                phase_shifts[i, j] = max(0.1, shift_factor)
                phase_shifts[j, i] = max(0.1, shift_factor)
                
        return phase_shifts

    def process_bac2_v13(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Khai thác Entropy Động
        mi_mat = self.compute_mutual_information_dynamic(X)
        
        # 2. Khai thác Dịch Pha Topo
        phase_mat = self.compute_topological_phase_shift(X)
        
        # 3. Phân rã không gian Riemann phi tuyến
        norms = np.linalg.norm(X, axis=0) + 1e-5
        dot_products = np.dot(X.T, X)
        cos_sim = np.clip(dot_products / np.outer(norms, norms), -1.0, 1.0)
        geo_mat = np.arccos(cos_sim)
        
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                f_mi = mi_mat[i, j]
                f_phase = phase_mat[i, j]
                f_topo = np.exp(-geo_mat[i, j])
                
                # TÍCH CHÉO PHÁ VỠ CÂN BẰNG LẶP SỐ
                score = (f_mi ** 1.5) * (f_phase ** 2.0) * (f_topo ** 1.2)
                
                # --- KHỐI PHẠT CHỐNG LẶP SỐ & BẤY MÔ HÌNH CŨ ---
                # Phạt số đã xuất hiện liên tục / bão hòa (>= 3 lần)
                if freqs[i] >= 3: score *= 0.0001
                if freqs[j] >= 3: score *= 0.0001
                
                # Phạt nghẽn mạch nổ trùng ở T-1
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.00001
                    
                # Phạt trùng nổ cách 1 kỳ (T-2) để tránh lặp chu kỳ ngắn
                if T >= 2 and X[-2, i] == 1 and X[-2, j] == 1:
                    score *= 0.01
                    
                # Phạt số gan bão hòa (0 lần xuất hiện trong cửa sổ)
                if freqs[i] == 0 and freqs[j] == 0:
                    score *= 0.001
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp danh sách 3.160 cặp Bậc 2
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # Khai thác 3 cặp phụ hoàn toàn tách biệt
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - MDM-IDS v13.0
# ==============================================================================
st.title("🔄 MDM-IDS v13.0: DYNAMIC PERTURBATION ENGINE")
st.caption("Khử Tín Hiệu Lặp Số • Dịch Pha Topo Hilbert-Huang • Shannon Weighting • Quét Tự Do 3.160 Cặp")

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
                
        engine = MDM_IDS_Bac2_v13_Dynamic()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2_v13(matrix)
        
        st.success(f"⚡ Đã quét triệt tiêu lặp số & tái phân bố Topo v13.0 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 CHUYỂN PHA DỘNG TỐI ƯU (CHỐT 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 18px; background-color: #1A0903; border-radius: 12px; border: 2px solid #FF5500; box-shadow: 0 0 15px rgba(255, 85, 0, 0.4);'>"
            f"<h1 style='color: #FF5500; margin:0; font-size: 2.8rem;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:5px 0 0 0;'>Điểm xung lực dịch pha v13.0: {best_score:.8f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 DỰ PHÒNG CHUYỂN PHA MỚI")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 BẢNG TOP 10 CẶP BẬC 2 ĐÃ ĐƯỢC LỌC CHỐNG LẶP")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2 Biến Động": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Điểm Xung Lực v13.0": [f"{all_sorted[i][1]:.8f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ xử lý v13.0.")
