import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v11.0 Topo-Information Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v11.0: NON-LINEAR TOPO-INFORMATION FIELD (NTIF)
# ==============================================================================
class MDM_IDS_Bac2_v11_NTIF:
    def __init__(self, anchor_num=50, dim=80):
        self.anchor = anchor_num - 1 # Index 0-79
        self.D = dim

    def compute_mutual_information(self, x, y):
        """1. TÍNH LƯỢNG THÔNG TIN TƯƠNG HỖ SHANNON I(X;Y)"""
        p_11 = np.mean((x == 1) & (y == 1)) + 1e-9
        p_10 = np.mean((x == 1) & (y == 0)) + 1e-9
        p_01 = np.mean((x == 0) & (y == 1)) + 1e-9
        p_00 = np.mean((x == 0) & (y == 0)) + 1e-9
        
        px_1 = np.mean(x == 1) + 1e-9
        px_0 = np.mean(x == 0) + 1e-9
        py_1 = np.mean(y == 1) + 1e-9
        py_0 = np.mean(y == 0) + 1e-9
        
        mi = (p_11 * np.log2(p_11 / (px_1 * py_1)) +
              p_10 * np.log2(p_10 / (px_1 * py_0)) +
              p_01 * np.log2(p_01 / (px_0 * py_1)) +
              p_00 * np.log2(p_00 / (px_0 * py_0)))
        return max(0.0, mi)

    def compute_geodesic_manifold_dist(self, X, idx1, idx2):
        """2. ĐỘ CONG HÌNH HỌC PHI TUYẾN (Geodesic Distance)"""
        # Khoảng cách cosine phi tuyến trên không gian ẩn
        v1 = X[:, idx1]
        v2 = X[:, idx2]
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1) + 1e-5
        norm_v2 = np.linalg.norm(v2) + 1e-5
        
        cos_sim = dot_product / (norm_v1 * norm_v2)
        # Độ cong Riemann
        geodesic_dist = np.arccos(np.clip(cos_sim, -1.0, 1.0))
        return geodesic_dist

    def process_bac2_v11(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        anchor_idx = self.anchor
        x_anchor = X[:, anchor_idx]
        
        candidate_scores = {}
        for j in range(D):
            if j == anchor_idx:
                continue
                
            x_j = X[:, j]
            
            # Factor 1: Entropy Tin tức Shannon Tương hỗ I(50; j)
            f_mi = self.compute_mutual_information(x_anchor, x_j)
            
            # Factor 2: Mức độ trùng khớp Topo (Sức căng Riemann)
            g_dist = self.compute_geodesic_manifold_dist(X, anchor_idx, j)
            f_topo = np.exp(-g_dist) # Càng gần trên không gian cong điểm càng cao
            
            # Factor 3: Lợi thế lệch pha tích tụ (Potential Gradient)
            last_seen_anchor = np.where(x_anchor == 1)[0]
            last_seen_j = np.where(x_j == 1)[0]
            
            recency_gap = 1.0
            if len(last_seen_anchor) > 0 and len(last_seen_j) > 0:
                gap = abs(last_seen_anchor[-1] - last_seen_j[-1])
                recency_gap = np.exp(-gap / 2.0)
                
            # Tổng hợp điểm phi tuyến tính (Hội tụ Topo-Information)
            score = (f_mi ** 1.8) * (f_topo ** 1.5) * (recency_gap ** 1.2)
            
            # --- PHẠT LOẠI BỎ SAI SỐ & NGHẼN MẠCH ---
            if freqs[j] >= 3: score *= 0.001 # Phạt mạnh số xuất hiện quá 3 lần
            if X[-1, anchor_idx] == 1 and X[-1, j] == 1: score *= 0.0001 # Tránh nổ lại trùng kỳ
            if freqs[j] == 0: score *= 0.001 # Loại bỏ số bị lỳ âm
            
            candidate_scores[j + 1] = score
            
        # Sắp xếp ứng viên N2
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_n2 = sorted_candidates[0][0]
        best_score = sorted_candidates[0][1]
        
        backup_n2 = [sorted_candidates[1][0], sorted_candidates[2][0], sorted_candidates[3][0]]
        
        return (self.anchor + 1, best_n2), best_score, [(self.anchor + 1, n) for n in backup_n2], sorted_candidates

# ==============================================================================
# STREAMLIT UI - MDM-IDS v11.0
# ==============================================================================
st.title("🌐 MDM-IDS v11.0: TOPO-INFORMATION FIELD")
st.caption("Khử Suy Luận Cổ Điển • Shannon Mutual Information • Độ Cong Riemann Topo • Điểm Tựa Anchor 50")

raw_input = st.text_area(
    "Dán dữ liệu 5 đến 10 kỳ Keno mới nhất vào đây:",
    placeholder="Kỳ 1: 01 02 05 50 ...\nKỳ 2: ...",
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
                
        engine = MDM_IDS_Bac2_v11_NTIF(anchor_num=50)
        best_pair, best_score, backup_pairs, sorted_candidates = engine.process_bac2_v11(matrix)
        
        st.success(f"⚡ Đã quét xong Topo-Information Field lấy điểm tựa N1 = 50 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 CHỐT TOPO-INFORMATION (TARGET 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT CỐ ĐỊNH N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT TƯƠNG HỖ N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 18px; background-color: #0E0712; border-radius: 12px; border: 2px solid #D100FF; box-shadow: 0 0 15px rgba(209, 0, 255, 0.4);'>"
            f"<h1 style='color: #D100FF; margin:0; font-size: 2.8rem;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:5px 0 0 0;'>Cường độ tin tức Shannon tương hỗ: {best_score:.8f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC SỐ N2 BẮT CẶP ĐỘ CONG TOPO DỰ PHÒNG")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **50 — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **50 — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **50 — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 BẢNG TOP 10 SỐ N2 CÓ ĐỘ HỘI TỤ TOPO LỚN NHẤT VỚI 50")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2": [f"(50, {sorted_candidates[i][0]:02d})" for i in range(10)],
            "Điểm Entropy Tương Hỗ": [f"{sorted_candidates[i][1]:.8f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để chạy bộ lọc Topo v11.0.")
