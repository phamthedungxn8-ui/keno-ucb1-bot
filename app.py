import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v15.2 Dual-Core Engine", layout="wide")

# ==============================================================================
# MÔ HÌNH MDM-IDS v15.2: DUAL-CORE PHASE BALANCE ENGINE
# ==============================================================================

class DSD_DualCore_Engine:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def compute_superposition_scores(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Tần suất xuất hiện chuẩn hóa
        p_occ = freqs / float(T)
        
        # 2. Xung lực dịch pha nhị nguyên (Duality Phase Bias)
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

        # 3. Mật độ vướng víu thông tin SVD/Corr
        if T >= 4:
            cov = np.corrcoef(X.T)
            cov = np.nan_to_num(cov)
            entanglement_score = np.sum(np.abs(cov), axis=1) / float(D)
        else:
            entanglement_score = np.ones(D)

        # 4. Lọc phạt bão hòa & Ưu tiên vùng ranh giới pha
        phase_filter = np.ones(D)
        for i in range(D):
            if freqs[i] == 0:
                phase_filter[i] = 0.35 # Phạt số đóng băng quá lâu
            elif freqs[i] in [1, 2, 3]:
                phase_filter[i] = 1.95 # Ưu tiên tối đa vùng ranh giới pha (nơi dàn phụ hay bắt trúng)
            elif freqs[i] >= 4:
                phase_filter[i] = 0.15 # Phạt số bão hòa

            # Phạt trùng lặp T-1
            if X[-1, i] == 1:
                phase_filter[i] *= 0.20

        scores = (p_occ * 0.25 + duality_bias * 0.35 + entanglement_score * 0.40) * phase_filter
        return scores

    def process_dual_core(self, X):
        scores = self.compute_superposition_scores(X)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]
        
        # Trích xuất Top 16 con số có điểm tốt nhất
        top_16 = ranked_numbers[:16]
        
        # CƠ CHẾ XEN KẼ DUAL-CORE (Phân bổ cân bằng lực giữa 2 Dàn)
        # Dàn Alpha (Chủ lực 1): Lấy vị trí lẻ trong Top (1, 3, 5, 7, 9, 11, 13, 15)
        dan_alpha_8 = sorted([top_16[i] for i in range(0, 16, 2)])
        
        # Dàn Beta (Chủ lực 2 / Bộc phát): Lấy vị trí chẵn trong Top (2, 4, 6, 8, 10, 12, 14, 16)
        dan_beta_8 = sorted([top_16[i] for i in range(1, 16, 2)])
        
        # Dàn Bậc 7 cho Alpha & Beta
        dan_alpha_7 = sorted(dan_alpha_8[:7])
        dan_beta_7 = sorted(dan_beta_8[:7])

        return {
            "alpha_8": dan_alpha_8,
            "beta_8": dan_beta_8,
            "alpha_7": dan_alpha_7,
            "beta_7": dan_beta_7,
            "scores": scores,
            "ranked_all": ranked_numbers
        }

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.title("🦅 MDM-IDS v15.2: DUAL-CORE PHASE BALANCE ENGINE")
st.caption("Cân Bằng Song Lực Alpha/Beta • Tối Ưu Dàn Phụ Bộc Phát • Lọc Cuốn Chiếu 6-8 Kỳ")

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
                
        engine = DSD_DualCore_Engine()
        res = engine.process_dual_core(matrix)
        
        st.success(f"⚡ Đã hoàn tất tái phân bổ lực Dual-Core trên {kies_to_use} kỳ cuốn chiếu!")
        st.markdown("---")
        
        # HIỂN THỊ CÂN BẰNG SONG LỰC ALPHA VÀ BETA
        col_alpha, col_beta = st.columns(2)
        
        with col_alpha:
            st.subheader("🔥 DÀN ALPHA (TÍCH LŨY PHA)")
            str_a8 = "  •  ".join([f"**{n:02d}**" for n in res["alpha_8"]])
            st.markdown(
                f"<div style='text-align: center; padding: 18px; background-color: #0A192F; border-radius: 12px; border: 2px solid #00F0FF;'>"
                f"<h3 style='color: #00F0FF; margin:0;'>BẬC 8: {str_a8}</h3>"
                f"<p style='color: #AAA; margin:5px 0 0 0;'>Bậc 7: {' - '.join([f'{n:02d}' for n in res['alpha_7']])}</p>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        with col_beta:
            st.subheader("⚡ DÀN BETA (BÙNG NỔ / DÀN PHỤ TỐI ƯU)")
            str_b8 = "  •  ".join([f"**{n:02d}**" for n in res["beta_8"]])
            st.markdown(
                f"<div style='text-align: center; padding: 18px; background-color: #1A0903; border-radius: 12px; border: 2px solid #FF5500;'>"
                f"<h3 style='color: #FF5500; margin:0;'>BẬC 8: {str_b8}</h3>"
                f"<p style='color: #AAA; margin:5px 0 0 0;'>Bậc 7: {' - '.join([f'{n:02d}' for n in res['beta_7']])}</p>"
                f"</div>", 
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Mẹo chiến thuật:** Khi nhận thấy nhịp quay đang biến động mạnh, hãy ưu tiên đánh **DÀN BETA** hoặc đánh song song cả 2 dàn với cùng mức vốn để tối đa hóa xác suất trúng $4/8, 5/8, 6/8$.")

        st.markdown("---")
        st.subheader("📊 BẢNG XẾP HẠNG TOP 16 CÁC NÚT SỐ ĐÃ ĐƯỢC TẢI ĐỀU")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(16)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(16)],
            "Phân bổ Dàn": ["Dàn ALPHA" if i%2==0 else "Dàn BETA (Bùng nổ)" for i in range(16)],
            "Điểm Xung Lực": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(16)]
        })
        st.table(df_top)
        
    else:
        st.warning(f"Cần tối thiểu 6 kỳ cuốn chiếu để chạy mô hình (Hiện có {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu 6-8 kỳ Keno vào khung trên để khởi chạy cấu trúc Dual-Core v15.2.")
