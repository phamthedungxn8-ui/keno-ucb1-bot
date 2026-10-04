import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v3.0 Ultra-Precision Bậc 2 Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v3.0: ULTRA-PRECISION BAC 2 DYNAMICS
# ==============================================================================
class MDM_IDS_Bac2_Ultra:
    def __init__(self, dim=80):
        self.D = dim

    def calculate_shannon_entropy(self, prob_vector):
        """Tính Entropy Shannon để đo độ hỗn độn thông tin"""
        prob_vector = prob_vector[prob_vector > 0]
        if len(prob_vector) == 0:
            return 0.0
        return -np.sum(prob_vector * np.log2(prob_vector))

    def build_ultra_hamiltonian(self, X):
        """Dựng Hamilton với Phân mảnh Attractor, Nguyên lý Pauli & Kinh Dịch Modulo"""
        T, D = X.shape
        freqs = X.sum(axis=0).astype(float)
        
        # 1. Năng lượng nền Shannon Entropy nghịch đảo cho từng số
        node_energy = np.zeros(D)
        for d in range(D):
            history_d = X[:, d]
            p = np.mean(history_d)
            if 0 < p < 1:
                ent = self.calculate_shannon_entropy(np.array([p, 1 - p]))
                # Lấy năng lượng tỷ lệ nghịch với entropy (nơi trật tự nén lại)
                node_energy[d] = 1.0 / (ent + 1e-5)
            else:
                node_energy[d] = 0.5
                
        # 2. Ma trận tương tác vi mô H_ij (80x80)
        H_interaction = np.zeros((D, D))
        golden_gaps = {8, 13, 21, 34} # Bán kính attractor fractal
        
        for i in range(D):
            for j in range(i + 1, D):
                num1, num2 = i + 1, j + 1
                gap = abs(num1 - num2)
                
                weight = 0.0
                
                # A. Chi tiết Phân mảnh Attractor (Golden Gaps)
                if gap in golden_gaps:
                    weight += 0.45
                    
                # B. Chi tiết Kinh Dịch Modulo (Xung hợp Âm Dương Mod 10 lệch 5)
                if abs((num1 % 10) - (num2 % 10)) == 5:
                    weight += 0.35
                    
                # C. Chi tiết Bát quái Đồng dư Modulo 8
                if (num1 % 8) == (num2 % 8):
                    weight += 0.25
                    
                # D. Nguyên lý Pauli: Phạt nhẹ hai số quá kề nhau (gap = 1) nếu cùng nổ
                if gap == 1:
                    weight -= 0.15
                    
                H_interaction[i, j] = weight
                H_interaction[j, i] = weight
                
        H = np.diag(node_energy) + H_interaction
        return H, node_energy

    def process_bac2_ultra(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # Dựng Ma trận Hamilton Vi mô
        H, node_energy = self.build_ultra_hamiltonian(X)
        
        # Dựng Ma trận Mật độ Đồng xuất hiện (Co-occurrence Matrix)
        rho_co = np.dot(X.T, X) / T
        
        # Lọc SVD nâng cao loại bỏ nhiễu ngẫu nhiên
        U, S, Vt = np.linalg.svd(rho_co)
        S_clean = np.zeros_like(S)
        k_keep = max(1, int(len(S) * 0.25)) # Giữ 25% giá trị riêng lớn nhất
        S_clean[:k_keep] = S[:k_keep]
        rho_filtered = np.dot(U, np.dot(np.diag(S_clean), Vt))
        
        # TÍNH ĐIỂM CHI TIẾT TẤT CẢ 3,160 CẶP BẬC 2
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                # 1. Tương tác Lượng tử Hamilton vi mô
                h_weight = H[i, j]
                
                # 2. Độ rối không gian SVD đã lọc nhiễu
                svd_coupling = rho_filtered[i, j]
                
                # 3. Tổng năng lượng Entropy Shannon
                entropy_pot = node_energy[i] + node_energy[j]
                
                # Tổng hợp điểm chưa phạt
                score = (svd_coupling * 3.5) + (h_weight * 3.0) + (entropy_pot * 1.5)
                
                # --- LỚP BẢO VỆ & PHẠT CHI TIẾT (BẪY MẬT ĐỘ) ---
                # Phạt số xuất hiện quá dày (Mật độ >= 3 lần trong tập cửa sổ)
                if freqs[i] >= 3: score *= 0.10
                if freqs[j] >= 3: score *= 0.10
                
                # Phạt cực nặng nếu CẢ HAI SỐ đều nổ ở kỳ gần nhất T-1
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.01
                    
                # Phạt nếu cặp này cùng nổ cách 1 kỳ (Ví dụ: kỳ T-1 và kỳ T-3)
                if T >= 3 and X[-1, i] == 1 and X[-3, j] == 1:
                    score *= 0.05
                if T >= 3 and X[-1, j] == 1 and X[-3, i] == 1:
                    score *= 0.05
                    
                # Phạt số gan cấm (0 lần xuất hiện)
                if freqs[i] == 0 and freqs[j] == 0:
                    score *= 0.05
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp kết quả Bậc 2
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # 3 Cặp Bậc 2 Phụ có độ độc lập không gian cao nhất với Top 1
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            # Kiểm tra không trùng lặp hoàn toàn với Best Pair
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - MDM-IDS v3.0
# ==============================================================================
st.title("⚡ MDM-IDS v3.0: ULTRA-PRECISION BẬC 2")
st.caption("Chi tiết Vi mô • Entropy Shannon Nghịch đảo • Phân mảnh Fractal • Dịch Lực Modulo")

raw_input = st.text_area(
    "Dán dữ liệu 5 đến 10 kỳ Keno gần nhất vào đây:",
    placeholder="Kỳ 1: 01 02 05 09 14 22 ...\nKỳ 2: ...",
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
                
        engine = MDM_IDS_Bac2_Ultra()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2_ultra(matrix)
        
        st.success(f"⚡ Đã hoàn tất phân tích vi mô 3,160 cặp Bậc 2 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 TỐI ƯU VI MÔ (CHỐT)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT CHÍNH N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT CHÍNH N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 15px; background-color: #1E1E1E; border-radius: 10px; border: 2px solid #FF4B4B;'>"
            f"<h1 style='color: #FF4B4B; margin:0;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:0;'>Điểm năng lượng vi mô: {best_score:.5f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 ĐỘC LẬP KHÔNG GIAN DỰ PHÒNG")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 BẢNG CHI TIẾT TOP 10 CẶP BẬC 2 ĐẠT NĂNG LƯỢNG CỰC ĐẠI")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Chỉ số Năng Lượng Rối Vi Mô": [f"{all_sorted[i][1]:.6f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện quét được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ lọc vi mô Bậc 2 v3.0.")
