import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v10.0 Anchor-Centric Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v10.0: ANCHOR-CENTRIC PHASE LOCKING (CHỐT THEO SỐ TỰA 50)
# ==============================================================================
class MDM_IDS_Bac2_v10_Anchor:
    def __init__(self, anchor_num=50, dim=80):
        self.anchor = anchor_num - 1 # Chuyển sang index 0-79
        self.D = dim

    def process_bac2_v10(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        anchor_idx = self.anchor
        
        # 1. Kiểm tra trạng thái xuất hiện của số Anchor (50)
        anchor_history = X[:, anchor_idx]
        
        # 2. Ma trận tương quan trực tiếp với Anchor 50 (Anchor Vector Projection)
        co_occur_with_anchor = np.dot(X.T, anchor_history) / float(T)
        
        # 3. Tính Pha Kuramoto khóa theo Anchor 50
        anchor_active_cycles = np.where(anchor_history == 1)[0]
        anchor_phase = 0.0
        if len(anchor_active_cycles) > 0:
            anchor_phase = np.angle(np.sum(np.exp(1j * (2 * np.pi * anchor_active_cycles / float(T)))))
            
        candidate_scores = {}
        for j in range(D):
            if j == anchor_idx:
                continue
                
            # Factor A: Tín hiệu đồng xuất hiện sạch với số 50
            f_co = co_occur_with_anchor[j]
            
            # Factor B: Khóa pha đồng bộ với số 50
            j_active = np.where(X[:, j] == 1)[0]
            j_phase = np.pi
            if len(j_active) > 0:
                j_phase = np.angle(np.sum(np.exp(1j * (2 * np.pi * j_active / float(T)))))
            phase_sync = (np.cos(abs(anchor_phase - j_phase)) + 1.0) / 2.0
            
            # Factor C: Khoảng cách Hamming ECC giữa Anchor 50 và số j
            h_dist = np.sum(X[:, anchor_idx] != X[:, j])
            f_ecc = np.exp(-((h_dist - (T * 0.38))**2) / (2.0 * 1.5))
            
            # Factor D: Khoảng cách không gian 2D trên lưới Keno 8x10
            r1, c1 = anchor_idx // 10, anchor_idx % 10
            r2, c2 = j // 10, j % 10
            grid_dist = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
            f_grid = 1.0 if grid_dist in [1, 2, 3, 5] else 0.5 # Fibonacci Grid Coupling
            
            # Điểm tích chéo tổng hợp với Anchor 50
            score = (f_co ** 1.5) * (phase_sync ** 2.0) * (f_ecc ** 1.2) * f_grid
            
            # --- PHẠT LỌC TRIỆT TÁC SAI SỐ CHO N2 ---
            # Phạt nếu số j đã bão hòa (về >= 3 lần trong cửa sổ)
            if freqs[j] >= 3: score *= 0.005
            
            # Phạt nếu số j vừa nổ cùng số 50 ở kỳ T-1 (tránh bẫy nghẽn mạch)
            if X[-1, anchor_idx] == 1 and X[-1, j] == 1:
                score *= 0.0001
                
            # Phạt nhịp lệch đối xứng cách kỳ
            if T >= 3 and X[-1, anchor_idx] == 1 and X[-3, j] == 1:
                score *= 0.001
                
            # Phạt số gan bão hòa âm
            if freqs[j] == 0:
                score *= 0.001
                
            candidate_scores[j + 1] = score
            
        # Sắp xếp các số N2 tối ưu nhất bắt cặp với số 50
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_n2 = sorted_candidates[0][0]
        best_score = sorted_candidates[0][1]
        
        # 3 Số N2 dự phòng độc lập
        backup_n2 = [sorted_candidates[1][0], sorted_candidates[2][0], sorted_candidates[3][0]]
        
        return (self.anchor + 1, best_n2), best_score, [(self.anchor + 1, n) for n in backup_n2], sorted_candidates

# ==============================================================================
# STREAMLIT UI - MDM-IDS v10.0 ANCHOR ENGINE
# ==============================================================================
st.title("🎯 MDM-IDS v10.0: ANCHOR-CENTRIC PHASE LOCKING")
st.caption("Khóa Pha Điểm Tựa Cố Định (Anchor: 50) • Quét Ma Trận Vướng Víu N2 • Lọc Triệt Tiêu Bẫy 1/2")

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
                
        engine = MDM_IDS_Bac2_v10_Anchor(anchor_num=50)
        best_pair, best_score, backup_pairs, sorted_candidates = engine.process_bac2_v10(matrix)
        
        st.success(f"⚡ Đã khóa điểm tựa N1 = 50 & quét hoàn tất 79 số N2 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 ĐỒNG BỘ ĐIỂM TỰA (CHỐT CHẮC 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT ĐIỂM TỰA N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT KHÓA PHA N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 18px; background-color: #0A192F; border-radius: 12px; border: 2px solid #00F0FF; box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);'>"
            f"<h1 style='color: #00F0FF; margin:0; font-size: 2.8rem;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:5px 0 0 0;'>Chỉ số vướng víu với số 50: {best_score:.8f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC SỐ N2 DỰ PHÒNG TỐT NHẤT ĐI CÙNG SỐ 50")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **50 — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **50 — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **50 — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 TOP 10 SỐ N2 CÓ KHẢ NĂNG NỔ ĐỒNG THỜI VỚI SỐ 50 CAO NHẤT")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2": [f"(50, {sorted_candidates[i][0]:02d})" for i in range(10)],
            "Điểm Lực Kéo Với Số 50": [f"{sorted_candidates[i][1]:.8f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để lấy số N2 bắt cặp với số 50.")
