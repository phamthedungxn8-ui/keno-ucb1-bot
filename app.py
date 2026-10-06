import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v15.3 Specialization: Bậc 2 & Bậc 3", layout="wide")

# ==============================================================================
# MÔ HÌNH MDM-IDS v15.3: CHUYÊN BIỆT TỔNG HỢP CHO BẬC 2 & BẬC 3
# ==============================================================================

class KenoBac2Bac3Engine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def compute_scores(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        p_occ = freqs / float(T)
        
        # 1. Trục Nhị nguyên Duality
        even_mask = np.array([1 if (i+1)%2 == 0 else 0 for i in range(D)])
        large_mask = np.array([1 if (i+1) > 40 else 0 for i in range(D)])
        
        recent_k = X[-1]
        even_ratio = np.sum(recent_k * even_mask) / 20.0
        large_ratio = np.sum(recent_k * large_mask) / 20.0
        
        duality_bias = np.zeros(D)
        for i in range(D):
            bias_even = (1.0 - even_ratio) if even_mask[i] else even_ratio
            bias_large = (1.0 - large_ratio) if large_mask[i] else large_ratio
            duality_bias[i] = 0.5 * (bias_even + bias_large)

        # 2. Vướng víu tương quan Ma trận (Entanglement)
        if T >= 4:
            cov = np.corrcoef(X.T)
            cov = np.nan_to_num(cov)
            entanglement_score = np.sum(np.abs(cov), axis=1) / float(D)
        else:
            entanglement_score = np.ones(D)

        # 3. Khối lọc triệt tiêu Lặp số & Báo hòa
        phase_filter = np.ones(D)
        for i in range(D):
            if freqs[i] == 0:
                phase_filter[i] = 0.30 # Phạt số quá đóng băng
            elif freqs[i] in [1, 2, 3]:
                phase_filter[i] = 1.95 # Ưu tiên ranh giới chuyển pha (vùng dàn phụ hay nổ)
            elif freqs[i] >= 4:
                phase_filter[i] = 0.15 # Phạt số bão hòa

            # Phạt trùng lặp kỳ T-1
            if X[-1, i] == 1:
                phase_filter[i] *= 0.20

        scores = (p_occ * 0.25 + duality_bias * 0.35 + entanglement_score * 0.40) * phase_filter
        return scores

    def process_bac2_bac3(self, X):
        scores = self.compute_scores(X)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]
        
        # --- LẤY DÀN BẬC 2 ---
        # Alpha: Lấy Nút 1 & Nút 3 (Top điểm tích lũy)
        bac2_alpha = sorted([ranked_numbers[0], ranked_numbers[2]])
        # Beta: Lấy Nút 2 & Nút 4 (Dàn phụ bộc phát - lấy vị trí chẵn)
        bac2_beta = sorted([ranked_numbers[1], ranked_numbers[3]])
        # Dự phòng Bậc 2: Nút 5 & Nút 6
        bac2_backup = sorted([ranked_numbers[4], ranked_numbers[5]])

        # --- LẤY DÀN BẬC 3 ---
        # Alpha: Lấy Nút 1, 3, 5
        bac3_alpha = sorted([ranked_numbers[0], ranked_numbers[2], ranked_numbers[4]])
        # Beta: Lấy Nút 2, 4, 6 (Dàn phụ bộc phát)
        bac3_beta = sorted([ranked_numbers[1], ranked_numbers[3], ranked_numbers[5]])
        # Dự phòng Bậc 3: Nút 7, 8, 9
        bac3_backup = sorted([ranked_numbers[6], ranked_numbers[7], ranked_numbers[8]])

        return {
            "bac2_alpha": bac2_alpha,
            "bac2_beta": bac2_beta,
            "bac2_backup": bac2_backup,
            "bac3_alpha": bac3_alpha,
            "bac3_beta": bac3_beta,
            "bac3_backup": bac3_backup,
            "scores": scores,
            "ranked_all": ranked_numbers
        }

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.title("🎯 MDM-IDS v15.3: BẬC 2 & BẬC 3 SPECIALIZATION ENGINE")
st.caption("Tối Ưu Chuyên Biệt Bậc 2 (Ăn 90k) & Bậc 3 (Ăn 200k/Hoàn 20k) • Cơ Chế Cân Bằng Dual-Core Alpha/Beta")

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu (6-8 kỳ gần nhất):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=160
)

if raw_input.strip():
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_input.strip(), flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 6:
        kies_to_use = min(total_kies, 8)
        used_numbers = all_numbers[-kies_to_use * 20:]
        
        matrix = np.zeros((kies_to_use, 80), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0
                
        engine = KenoBac2Bac3Engine()
        res = engine.process_bac2_bac3(matrix)
        
        st.success(f"⚡ Đã quét và tổng hợp xong bộ số chuyên biệt Bậc 2 & Bậc 3 trên {kies_to_use} kỳ dữ liệu!")
        st.markdown("---")
        
        # HIỂN THỊ CHUYÊN BIỆT BẬC 2
        st.subheader("🔥 MỤC TIÊU BẬC 2 (CHỌN 2 - THƯỞNG 90.000 VNĐ)")
        col2_a, col2_b, col2_c = st.columns(3)
        
        with col2_a:
            st.markdown(
                f"<div style='text-align: center; padding: 15px; background-color: #0A192F; border-radius: 10px; border: 2px solid #00F0FF;'>"
                f"<h4 style='color: #00F0FF; margin:0;'>BẬC 2 - ALPHA</h4>"
                f"<h2 style='color: #FFF; margin:5px 0;'>{res['bac2_alpha'][0]:02d} — {res['bac2_alpha'][1]:02d}</h2>"
                f"<p style='color: #AAA; margin:0; font-size:0.8rem;'>Dàn tích lũy pha chuẩn</p>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        with col2_b:
            st.markdown(
                f"<div style='text-align: center; padding: 15px; background-color: #1A0903; border-radius: 10px; border: 2px solid #FF5500;'>"
                f"<h4 style='color: #FF5500; margin:0;'>BẬC 2 - BETA (BÙNG NỔ)</h4>"
                f"<h2 style='color: #FFF; margin:5px 0;'>{res['bac2_beta'][0]:02d} — {res['bac2_beta'][1]:02d}</h2>"
                f"<p style='color: #AAA; margin:0; font-size:0.8rem;'>Dàn phụ đón nhịp chuyển giao</p>"
                f"</div>", 
                unsafe_allow_html=True
            )

        with col2_c:
            st.markdown(
                f"<div style='text-align: center; padding: 15px; background-color: #111; border-radius: 10px; border: 1px solid #555;'>"
                f"<h4 style='color: #AAA; margin:0;'>BẬC 2 - DỰ PHÒNG</h4>"
                f"<h2 style='color: #FFF; margin:5px 0;'>{res['bac2_backup'][0]:02d} — {res['bac2_backup'][1]:02d}</h2>"
                f"<p style='color: #AAA; margin:0; font-size:0.8rem;'>Bọc lót nhịp thứ 3</p>"
                f"</div>", 
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        
        # HIỂN THỊ CHUYÊN BIỆT BẬC 3
        st.subheader("⚡ MỤC TIÊU BẬC 3 (CHỌN 3 - THƯỞNG 200.000 VNĐ / TRÚNG 2/3 HOÀN 20.000 VNĐ)")
        col3_a, col3_b, col3_c = st.columns(3)
        
        with col3_a:
            b3_a_str = " - ".join([f"{n:02d}" for n in res["bac3_alpha"]])
            st.info(f"### **{b3_a_str}**\n*Bậc 3 Dàn Alpha*")
            
        with col3_b:
            b3_b_str = " - ".join([f"{n:02d}" for n in res["bac3_beta"]])
            st.warning(f"### **{b3_b_str}**\n*Bậc 3 Dàn Beta (Tối ưu dàn phụ)*")

        with col3_c:
            b3_bk_str = " - ".join([f"{n:02d}" for n in res["bac3_backup"]])
            st.secondary(f"### **{b3_bk_str}**\n*Bậc 3 Dự phòng*") if hasattr(st, 'secondary') else st.write(f"### **{b3_bk_str}**\n*Bậc 3 Dự phòng*")

        st.markdown("---")
        st.subheader("📊 TOP 12 CON SỐ CÓ ĐIỂM XUNG LỰC ĐƠN LẺ CAO NHẤT")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(12)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(12)],
            "Vị trí phân bổ": ["Bậc 2/3 Alpha" if i%2==0 else "Bậc 2/3 Beta" for i in range(12)],
            "Điểm Xung Lực": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(12)]
        })
        st.table(df_top.T)
        
    else:
        st.warning(f"Cần tối thiểu 6 kỳ cuốn chiếu để chạy mô hình (Hiện có {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu 6-8 kỳ Keno vào khung trên để xuất bộ số Bậc 2 & Bậc 3.")
