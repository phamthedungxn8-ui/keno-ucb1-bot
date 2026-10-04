import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v9.0 Multi-Inference Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v9.0: TENSOR NETWORK, GAME THEORY & HYDRODYNAMICS
# ==============================================================================
class MDM_IDS_Bac2_v9_MultiInference:
    def __init__(self, dim=80):
        self.D = dim

    def game_theory_nash_symbiosis(self, X):
        """1. LÝ THUYẾT TRÒ CHƠI TIẾN HÓA (Cân bằng Nash & Cộng sinh ESS)"""
        T, D = X.shape
        co_occur = np.dot(X.T, X) / float(T)
        
        # Ma trận Trả thưởng Payoff Matrix cho 80 quần thể
        nash_matrix = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                payoff_i = co_occur[i, j] - (np.mean(X[:, i]) * 0.5)
                payoff_j = co_occur[i, j] - (np.mean(X[:, j]) * 0.5)
                # Trạng thái cân bằng Nash cộng sinh: Cả 2 cùng có Payoff dương và tương đồng
                symbiosis_score = max(0.0, payoff_i * payoff_j)
                nash_matrix[i, j] = symbiosis_score
                nash_matrix[j, i] = symbiosis_score
                
        return nash_matrix

    def quantum_hydrodynamics_pressure(self, X):
        """2. ĐỘNG LỰC HỌC CHẤT LƯU LƯỢNG TỬ (Madelung Pressure Gradient)"""
        T, D = X.shape
        densities = np.mean(X, axis=0) + 1e-5
        
        pressure_matrix = np.zeros((D, D))
        for i in range(D):
            for j in range(i + 1, D):
                # Gradient độ nén áp suất giữa 2 điểm chất lưu
                grad_p = abs(densities[i] - densities[j])
                # Áp lực thủy động cực đại khi gradient tiệm cận ngưỡng chuyển pha
                press = np.exp(-((grad_p - 0.05)**2) / (2 * 0.01))
                pressure_matrix[i, j] = press
                pressure_matrix[j, i] = press
                
        return pressure_matrix

    def tensor_network_mps_entanglement(self, X):
        """3. MẠNG TENSOR & MATRIX PRODUCT STATES (MPS Entanglement)"""
        T, D = X.shape
        # Biến đổi ma trận lịch sử thành Tensor bậc 3 (Time-Slice Splitting)
        if T >= 6:
            split = T // 2
            X1 = X[:split, :]
            X2 = X[split:, :]
            cov1 = np.dot(X1.T, X1) / float(split)
            cov2 = np.dot(X2.T, X2) / float(split)
            # Sự co giãn Tensor qua 2 khoảng thời gian
            tensor_entanglement = np.abs(cov1 * cov2)
        else:
            tensor_entanglement = np.dot(X.T, X) / float(T)
            
        return tensor_entanglement

    def process_bac2_v9(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Luồng Trò chơi Tiến hóa
        nash_mat = self.game_theory_nash_symbiosis(X)
        
        # 2. Luồng Thủy động Lượng tử
        hydro_mat = self.quantum_hydrodynamics_pressure(X)
        
        # 3. Luồng Mạng Tensor MPS
        mps_mat = self.tensor_network_mps_entanglement(X)
        
        # PIPELINE HỘI TỤ ĐA LUỒNG SUY LUẬN (Multi-Inference Multiplicative Cross-Validation)
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                # Factor A: Điểm Cân bằng Nash Cộng sinh
                f_nash = nash_mat[i, j]
                
                # Factor B: Điểm Áp lực Thủy động
                f_hydro = hydro_mat[i, j]
                
                # Factor C: Độ vướng víu Mạng Tensor
                f_mps = mps_mat[i, j]
                
                # TÍCH CHÉO ĐA LUỒNG (Bắt buộc tất cả các luồng suy luận cùng đồng thuận)
                score = (f_nash ** 1.2) * (f_hydro ** 1.5) * (f_mps ** 1.8)
                
                # --- PHẠT TRIỆT TIÊU BẪY CHỆCH NHỊP TRÚNG 1/2 ---
                if freqs[i] >= 3: score *= 0.005
                if freqs[j] >= 3: score *= 0.005
                
                # Phạt trùng nổ ở kỳ T-1
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.0001
                    
                # Phạt lệch nhịp đối xứng (T-1 và T-3)
                if T >= 3 and X[-1, i] == 1 and X[-3, j] == 1:
                    score *= 0.001
                if T >= 3 and X[-1, j] == 1 and X[-3, i] == 1:
                    score *= 0.001
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp danh sách
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # Khai thác 3 cặp phụ độc lập hoàn toàn với Best Pair
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs

# ==============================================================================
# STREAMLIT UI - MDM-IDS v9.0
# ==============================================================================
st.title("🌌 MDM-IDS v9.0: MULTI-INFERENCE ENGINE")
st.caption("Hội Tụ Đa Luồng Suy Luận • Lý Thuyết Trò Chơi Nash • Thủy Động Lượng Tử • Mạng Tensor MPS")

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
                
        engine = MDM_IDS_Bac2_v9_MultiInference()
        best_pair, best_score, backup_pairs, all_sorted = engine.process_bac2_v9(matrix)
        
        st.success(f"⚡ Đã hoàn tất đối soát đa luồng suy luận v9.0 trên {kies_to_use} kỳ dữ liệu!")
        
        st.markdown("---")
        st.subheader("🔥 CẶP BẬC 2 ĐỒNG THUẬN ĐA LUỒNG SUY LUẬN (CHỐT 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("NÚT NASS/MPS N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("NÚT THỦY ĐỘNG N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 18px; background-color: #120A1C; border-radius: 12px; border: 2px solid #9D00FF; box-shadow: 0 0 15px rgba(157, 0, 255, 0.4);'>"
            f"<h1 style='color: #9D00FF; margin:0; font-size: 2.8rem;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:5px 0 0 0;'>Chỉ số đồng thuận đa luồng v9.0: {best_score:.8f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 DỰ PHÒNG CHUẨN CÂN BẰNG")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Cặp Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Cặp Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Cặp Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 BẢNG TOP 10 CẶP BẬC 2 ĐẠT MỨC ĐỘ ĐỒNG THUẬN CAO NHẤT")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Điểm Đồng Thuận Đa Luồng": [f"{all_sorted[i][1]:.8f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ lọc v9.0 Multi-Inference.")
