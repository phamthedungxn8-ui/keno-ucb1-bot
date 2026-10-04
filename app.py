import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v8.0 Dual-Register Logic Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v8.0: DUAL-REGISTER LOGIC ARCHITECTURE (RE-ORDERED MATH)
# ==============================================================================
class MDM_IDS_Bac2_v8_DRLA:
    def __init__(self, dim=80):
        self.D = dim

    def layer1_rmt_filter(self, X):
        """TẦNG 1: LỌC PHÂN RÃ KHÔNG GIAN CON RMT (Marchenko-Pastur)"""
        T, D = X.shape
        rho_raw = np.dot(X.T, X) / float(T)
        
        # Ngưỡng Marchenko-Pastur
        q = D / float(T) if T > 0 else 1.0
        lambda_max = (1.0 + np.sqrt(q)) ** 2
        
        # Phân rã trị riêng
        eigs, vecs = np.linalg.eigh(rho_raw)
        # Triệt tiêu toàn bộ trị riêng dưới ngưỡng RMT
        eigs_clean = np.where(eigs > lambda_max, eigs, 0.0)
        
        rho_clean = np.dot(vecs, np.dot(np.diag(eigs_clean), vecs.T))
        return rho_clean

    def layer2_kuramoto_phase_lock(self, X):
        """TẦNG 2: PHƯƠNG TRÌNH KHÓA PHA KURAMOTO DẠNG SỐ SỰ KIỆN"""
        T, D = X.shape
        phases = np.zeros(D)
        
        for d in range(D):
            active_cycles = np.where(X[:, d] == 1)[0]
            if len(active_cycles) > 0:
                # Tính véc-tơ pha trung bình trên vòng tròn lượng giác
                angle_sum = np.sum(np.exp(1j * (2 * np.pi * active_cycles / float(T))))
                phases[d] = np.angle(angle_sum)
            else:
                phases[d] = np.pi
                
        return phases

    def layer3_ecc_bitwise_logic(self, X):
        """TẦNG 3: MA TRẬN SỬA LỖI HAMMING ECC TÍCH HỢP CỔNG LOGIC"""
        T, D = X.shape
        bit_matrix = X.astype(int)
        
        # Phép XOR dồn tích lũy trạng thái thanh ghi
        xor_reg = np.zeros(D, dtype=int)
        for t in range(T):
            xor_reg = np.bitwise_xor(xor_reg, bit_matrix[t, :])
            
        # Ma trận khoảng cách Hamming chuẩn hóa
        hamming_mat = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                h_dist = np.sum(bit_matrix[:, i] != bit_matrix[:, j])
                # Hàm mật độ Gauss cho khoảng cách Hamming ECC tối ưu
                ecc_val = np.exp(-((h_dist - (T * 0.38))**2) / (2.0 * 1.5))
                hamming_mat[i, j] = ecc_val
                hamming_mat[j, i] = ecc_val
                
        return xor_reg, hamming_mat

    def process_bac2_v8(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Chạy Tầng 1: RMT Space
        rho_rmt = self.layer1_rmt_filter(X)
        
        # 2. Chạy Tầng 2: Kuramoto Phase Lock
        phases = self.layer2_kuramoto_phase_lock(X)
        
        # 3. Chạy Tầng 3: Bitwise ECC & Logic Register
        xor_reg, hamming_ecc = self.layer3_ecc_bitwise_logic(X)
        
        # 4. CHUỖI NHÂN TÍCH CHÉO PHƯƠNG TRÌNH (CASCADED MULTIPLICATIVE PIPELINE)
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                # Factor A: Tín hiệu RMT Sạch (>0)
                f_rmt = max(0.0, rho_rmt[i, j])
                
                # Factor B: Khóa Pha Kuramoto (Khóa góc pha)
                phase_diff = abs(phases[i] - phases[j])
                f_phase = (np.cos(phase_diff) + 1.0) / 2.0  # Chuẩn hóa về [0, 1]
                
                # Factor C: Hamming ECC Alignment
                f_ecc = hamming_ecc[i, j]
                
                # Factor D: Trạng thái XOR Register Matching
                f_xor = 1.0 if xor_reg[i] == xor_reg[j] else 0.3
                
                # NĂNG LƯỢNG TÍCH CHÉO DẠNG CHUYỂN PHA (Tụ hội đa tầng)
                # Phép nhân đảm bảo nếu 1 yếu tố bằng 0, điểm cặp sẽ bị loại bỏ hoàn toàn
                score = (f_rmt ** 1.5) * (f_phase ** 2.0) * (f_ecc ** 1.2) * f_xor
                
                # --- PHẠT TRIỆT TIÊU BẪY TRÚNG 1/2 ---
                # Phạt thanh ghi bão hòa (xuất hiện >= 3 lần trong cửa sổ)
                if freqs[i] >= 3: score *= 0.005
                if freqs[j] >= 3: score *= 0.005
                
                # Phạt nghẽn mạch (Cả 2 số cùng xuất hiện ở kỳ T-1)
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.0001
                    
                # Phạt trượt xung nhịp đối xứng (T-1 và T-3)
                if T >= 3 and X[-1, i] == 1 and X[-3, j] == 1:
                    score *= 0.001
                if T >= 3 and X[-1, j] == 1 and X[-3, i] == 1:
                    score *= 0.001
                    
                # Phạt số gan bão hòa âm
                if freqs[i] == 0 and freqs[j] == 0:
                    score *= 0.001
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp danh sách cặp Bậc 2
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # Khai thác 3 cặp phụ hoàn toàn độc lập không gian thanh ghi
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - MDM-IDS v8.0
# ==============================================================================
st.title("🎛️ MDM-IDS v8.0: DUAL-REGISTER LOGIC ENGINE")
st.caption("Tái Cấu Trúc Hệ Phương Trình Nhân Tích Chéo • Khóa Pha Kuramoto • Lọc Chuỗi RMT Cascade")

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
                
        engine = MDM_IDS_Bac2_v8_DRLA()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2_v8(matrix)
        
        st.success(f"⚡ Đã hoàn tất xử lý Chuỗi Phương Trình Tích Chéo v8.0 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 ĐỒNG BỘ THANH GHI DUAL-REGISTER (CHỐT 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("BIT KHÓA PHA N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("BIT KHÓA PHA N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 18px; background-color: #08120C; border-radius: 12px; border: 2px solid #00FF66; box-shadow: 0 0 15px rgba(0, 255, 102, 0.4);'>"
            f"<h1 style='color: #00FF66; margin:0; font-size: 2.8rem;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:5px 0 0 0;'>Cường độ hội tụ đa tầng v8.0: {best_score:.8f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 DỰ PHÒNG CHUẨN THANH GHI")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Thanh Ghi Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Thanh Ghi Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Thanh Ghi Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 BẢNG CHI TIẾT TOP 10 CẶP BẬC 2 ĐẠT ĐIỂM TÍCH CHÉO CỰC ĐẠI")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2 (Dual-Register)": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Điểm Tích Chéo v8.0": [f"{all_sorted[i][1]:.8f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ lọc DRLA v8.0.")
