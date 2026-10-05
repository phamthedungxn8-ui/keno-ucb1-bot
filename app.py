import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v15.0 Dual-Superposition Engine", layout="wide")

# ==============================================================================
# HỆ THỐNG AGENT MÔ PHỎNG NHỊ NGUYÊN CHỒNG CHẬP (DSD)
# ==============================================================================

class DSD_AgentSystem:
    def __init__(self, n_numbers=80):
        self.D = n_numbers

    def compute_superposition_amplitudes(self, X):
        """
        Mô phỏng biên độ hàm sóng |beta_i|^2 cho 80 con số
        X: Matrix (T, 80) với T in [6, 8]
        """
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Biên độ xuất hiện cơ sở (Base Probability Amplitude)
        p_occ = freqs / float(T)
        
        # 2. Pha sóng Duality (Đo độ lệch pha giữa Am/Dương và Chẵn/Lẻ)
        even_mask = np.array([1 if (i+1)%2 == 0 else 0 for i in range(D)])
        large_mask = np.array([1 if (i+1) > 40 else 0 for i in range(D)])
        
        # Lực kéo từ sự mất cân bằng hệ thống
        recent_k = X[-1] # Kỳ gần nhất
        even_ratio = np.sum(recent_k * even_mask) / 20.0
        large_ratio = np.sum(recent_k * large_mask) / 20.0
        
        # Hệ số điều chỉnh nhị nguyên (Duality Compensation Factor)
        # Nếu kỳ trước nghiêng về Chẵn -> Tăng ưu tiên Lẻ ở kỳ tới
        duality_bias = np.zeros(D)
        for i in range(D):
            bias_even = (1.0 - even_ratio) if even_mask[i] else even_ratio
            bias_large = (1.0 - large_ratio) if large_mask[i] else large_ratio
            duality_bias[i] = 0.5 * (bias_even + bias_large)

        # 3. Mật độ Entanglement ngắn hạn (Vướng víu thông tin)
        if T >= 4:
            cov = np.corrcoef(X.T)
            cov = np.nan_to_num(cov)
            entanglement_score = np.sum(np.abs(cov), axis=1) / float(D)
        else:
            entanglement_score = np.ones(D)

        # 4. Sụp đổ hàm sóng (Wavefunction Collapse Score)
        # Lọc phạt số quá nhiệt hoặc số quá nguội
        phase_filter = np.ones(D)
        for i in range(D):
            if freqs[i] == 0:
                phase_filter[i] = 0.35 # Phạt số đóng băng
            elif freqs[i] in [1, 2, 3]:
                phase_filter[i] = 1.85 # Ưu tiên vùng ranh giới chuyển pha
            elif freqs[i] >= 4:
                phase_filter[i] = 0.20 # Phạt số bão hòa

            # Sóng dội (Anti-repeat filter)
            if X[-1, i] == 1:
                phase_filter[i] *= 0.25

        # Biên độ tổng hợp: |Beta_i|^2
        beta_squared = (p_occ * 0.3 + duality_bias * 0.3 + entanglement_score * 0.4) * phase_filter
        
        return beta_squared

    def process(self, X):
        scores = self.compute_superposition_amplitudes(X)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]
        
        return {
            "bac8": sorted(ranked_numbers[:8]),
            "bac7": sorted(ranked_numbers[:7]),
            "bac9": sorted(ranked_numbers[:9]),
            "backup8": sorted(ranked_numbers[8:16]),
            "scores": scores,
            "ranked_all": ranked_numbers
        }

# ==============================================================================
# UI STREAMLIT
# ==============================================================================

st.title("🌌 MDM-IDS v15.0: DUAL-SUPERPOSITION KENO ENGINE")
st.caption("Triết Lý Nhị Nguyên Chồng Chập • Mô Phỏng Biên Độ Pha Sóng • Tối Ưu Tích Thu Tiền Lẻ Bậc 7, 8")

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
                
        engine = DSD_AgentSystem()
        res = engine.process(matrix)
        
        st.success(f"⚡ Đã tính toán xong biên độ chồng chập pha cho {kies_to_use} kỳ cuốn chiếu!")
        st.markdown("---")
        
        # DÀN BẬC 8 CHỦ LỰC
        st.subheader("🎯 DÀN CHỦ LỰC BẬC 8 (BIÊN ĐỘ PHẠM VI AN TOÀN CAO NHẤT)")
        b8_str = "  •  ".join([f"**{n:02d}**" for n in res["bac8"]])
        st.markdown(
            f"<div style='text-align: center; padding: 20px; background-color: #0D1117; border-radius: 12px; border: 2px solid #7928CA; box-shadow: 0 0 20px rgba(121, 40, 202, 0.4);'>"
            f"<h2 style='color: #FF0080; margin:0;'>{b8_str}</h2>"
            f"<p style='color: #AAA; margin:8px 0 0 0;'>Mục tiêu: Đạt dải thưởng Trúng 0, 4, 5, 6 để tích góp dòng tiền dương</p>"
            f"</div>", 
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("🔥 DÀN BẬC 7 (CỘNG HƯỞNG PHA)")
            b7_str = " - ".join([f"{n:02d}" for n in res["bac7"]])
            st.info(f"### **{b7_str}**\n*Ứng viên có biên độ vướng víu mạnh nhất.*")
            
        with col_b:
            st.subheader("🛡️ DÀN PHỤ BỌC LÓT BẬC 8 (DSD-BACKUP)")
            bk_str = " - ".join([f"{n:02d}" for n in res["backup8"]])
            st.warning(f"### **{bk_str}**\n*Sử dụng khi hệ thống chuyển đổi trạng thái đột ngột.*")

        st.markdown("---")
        st.subheader("📊 BẢNG XẾP HẠNG TOP 15 BIÊN ĐỘ XÁC SUẤT SỤP ĐỔ |β_i|²")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(15)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(15)],
            "Biên độ Pha |β_i|²": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(15)]
        })
        st.table(df_top.T)
        
    else:
        st.warning(f"Cần tối thiểu 6 kỳ cuốn chiếu để chạy mô hình (Hiện có {total_kies} kỳ).")
