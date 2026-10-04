import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v7.0 Hardware Chipset Engine", layout="centered")

# ==============================================================================
# MÔ HÌNH MDM-IDS v7.0: HARDWARE CHIPSET ARCHITECTURE & BITWISE ALIGNMENT
# ==============================================================================
class MDM_IDS_Bac2_v7_Chipset:
    def __init__(self, dim=80):
        self.D = dim

    def detect_cache_flush(self, X):
        """1. KHỐI PHÁT HIỆN LÀM MỚI BỘ NHỚ ĐỆM (Cache Reset Detection)"""
        T, D = X.shape
        if T < 2:
            return False, 1.0
        
        # Tính khoảng cách Hamming giữa kỳ gần nhất T-1 và kỳ T-2
        last_state = X[-1, :]
        prev_state = X[-2, :]
        hamming_dist = np.sum(last_state != prev_state)
        
        # Nếu khoảng cách Hamming biến động cực đại (> 28/80 bit đổi trạng thái), hệ thống vừa Reset Cache
        is_flushed = hamming_dist > 28
        reset_weight = 0.35 if is_flushed else 1.0
        return is_flushed, reset_weight

    def compute_bitwise_logic_gates(self, X):
        """2. KHỐI CỔNG LOGIC XOR / AND / SHIFT TÍCH HỢP (ALU Simulation)"""
        T, D = X.shape
        bit_matrix = X.astype(int)
        
        # Phép toán XOR nối tiếp qua các kỳ xung nhịp (Clock Cycles)
        xor_accum = np.zeros(D, dtype=int)
        for t in range(T):
            xor_accum = np.bitwise_xor(xor_accum, bit_matrix[t, :])
            
        # Ma trận tương tác cổng AND song song (Parallel Bus Lines)
        and_bus = np.dot(bit_matrix.T, bit_matrix)
        
        return xor_accum, and_bus

    def compute_hamming_ecc_pairs(self, X):
        """3. KHỐI MÃ SỬA LỖI HAMMING (Hamming Parity Alignment)"""
        T, D = X.shape
        hamming_matrix = np.zeros((D, D))
        
        for i in range(D):
            for j in range(i + 1, D):
                # Tính khoảng cách Hamming trên chuỗi thời gian của 2 bit
                dist = np.sum(X[:, i] != X[:, j])
                # Khoảng cách Hamming tối ưu cho cặp ECC (Error-Correcting)
                hamming_matrix[i, j] = dist
                hamming_matrix[j, i] = dist
                
        return hamming_matrix

    def process_bac2_v7(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        
        # 1. Kiểm tra trạng thái Reset Cache Chipset
        is_flushed, reset_weight = self.detect_cache_flush(X)
        
        # 2. Xuất tín hiệu Cổng Logic ALU
        xor_accum, and_bus = self.compute_bitwise_logic_gates(X)
        
        # 3. Tính khoảng cách Parity Hamming
        hamming_matrix = self.compute_hamming_ecc_pairs(X)
        
        # CHẤM ĐIỂM VI MẠCH CHO 3,160 CẶP BẬC 2
        pair_scores = {}
        for i in range(D):
            for j in range(i + 1, D):
                # A. Điểm Bus dữ liệu Cổng AND
                bus_score = and_bus[i, j] / float(T)
                
                # B. Điểm trạng thái XOR Alignment (Cùng bật 1 hoặc cùng chờ)
                xor_parity = 1.0 if xor_accum[i] == xor_accum[j] else 0.25
                
                # C. Tối ưu khoảng cách Hamming Parity (Cân bằng bít lỗi)
                h_dist = hamming_matrix[i, j]
                # Bít ECC lý tưởng khi khoảng cách Hamming nằm trong nhịp ngắt vi mạch
                ecc_score = np.exp(-((h_dist - (T * 0.4))**2) / 2.0)
                
                # Tổng điểm logic vi mạch
                score = (bus_score * 4.5) + (xor_parity * 3.5) + (ecc_score * 3.0)
                
                # Áp dụng trọng số Reset Cache nếu vi mạch vừa xả bộ nhớ đệm
                score *= reset_weight
                
                # --- LỌC KHẮC PHỤC CHỆCH BÍT kiểm tra (TRIỆT TIÊU BẪY 1/2) ---
                # Phạt bit đã bão hòa cổng xuất (>= 3 lần nổ trong cửa sổ)
                if freqs[i] >= 3: score *= 0.01
                if freqs[j] >= 3: score *= 0.01
                
                # Phạt nghẽn mạch: CẢ HAI BÍT cùng vừa kích hoạt ở xung T-1
                if X[-1, i] == 1 and X[-1, j] == 1:
                    score *= 0.001
                    
                # Phạt lệch xung Clock (Bít này nổ T-1, Bít kia nổ T-3)
                if T >= 3 and X[-1, i] == 1 and X[-3, j] == 1:
                    score *= 0.01
                if T >= 3 and X[-1, j] == 1 and X[-3, i] == 1:
                    score *= 0.01
                    
                # Phạt 2 bít nằm ở 2 thanh ghi quá kề nhau nếu bị trùng luồng
                if abs(i - j) == 1 and bus_score < 0.15:
                    score *= 0.10
                    
                pair_scores[(i + 1, j + 1)] = score
                
        # Sắp xếp các đường Bus Bậc 2 theo tín hiệu kích hoạt giảm dần
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_pair = sorted_pairs[0][0]
        best_score = sorted_pairs[0][1]
        
        # Khai thác 3 kênh Bus dự phòng độc lập
        backup_pairs = []
        for pair, sc in sorted_pairs[1:]:
            if pair[0] not in best_pair and pair[1] not in best_pair:
                backup_pairs.append(pair)
            if len(backup_pairs) == 3:
                break
                
        return best_pair, best_score, backup_pairs, sorted_pairs, is_flushed

# ==============================================================================
# STREAMLIT UI - MDM-IDS v7.0
# ==============================================================================
st.title("💻 MDM-IDS v7.0: HARDWARE CHIPSET ENGINE")
st.caption("Mô phỏng Vi mạch ALU • Tín hiệu Cổng Logic XOR/AND • Căn chỉnh Bít ECC & Cache Reset")

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
                
        engine = MDM_IDS_Bac2_v7_Chipset()
        best_pair, best_score, backup_pairs, all_sorted, is_flushed = engine.process_bac2_v7(matrix)
        
        if is_flushed:
            st.warning("⚠️ PHÁT HIỆN TÍN HIỆU FLUSH CACHE (LÀM MỚI BỘ NHỚ ĐỆM): Hệ thống đã tự động chuyển sang chế độ phân tích nhịp ngắt vi mạch ngắn hạn.")
        else:
            st.success(f"⚡ Luồng Bus dữ liệu ổn định! Đã hoàn tất quét logic vi mạch v7.0 trên {kies_to_use} kỳ xung nhịp.")
            
        st.markdown("---")
        st.subheader("🔥 LUỒNG BUS BẬC 2 ĐỢT KÍCH HOẠT ĐỒNG THỜI (CHỐT 2/2)")
        
        c_n1, c_n2 = st.columns(2)
        with c_n1:
            st.metric("BIT CỔNG XUẤT N1", f"{best_pair[0]:02d}")
        with c_n2:
            st.metric("BIT KÍCH HOẠT ECC N2", f"{best_pair[1]:02d}")
            
        st.markdown(
            f"<div style='text-align: center; padding: 18px; background-color: #03101F; border-radius: 12px; border: 2px solid #00E5FF; box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);'>"
            f"<h1 style='color: #00E5FF; margin:0; font-size: 2.8rem;'>CẶP BẬC 2: {best_pair[0]:02d} — {best_pair[1]:02d}</h1>"
            f"<p style='color: #888; margin:5px 0 0 0;'>Chỉ số đồng bộ tín hiệu Chipset: {best_score:.6f}</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )

        st.markdown("<br>", unsafe_allow_dict=True)
        st.subheader("🛡️ CÁC CẶP BẬC 2 KÊNH BUS DỰ PHÒNG CHUẨN PARITY")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"Bus Phụ 1:\n### **{backup_pairs[0][0]:02d} — {backup_pairs[0][1]:02d}**")
        with c2:
            st.info(f"Bus Phụ 2:\n### **{backup_pairs[1][0]:02d} — {backup_pairs[1][1]:02d}**")
        with c3:
            st.info(f"Bus Phụ 3:\n### **{backup_pairs[2][0]:02d} — {backup_pairs[2][1]:02d}**")

        st.markdown("---")
        st.subheader("📊 BẢNG TÍN HIỆU BUS TOP 10 CẶP BẬC 2 ĐỒNG BỘ NĂNG LƯỢNG HIGH-BUS")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Cặp Bậc 2 (Bus Lines)": [f"({all_sorted[i][0][0]:02d}, {all_sorted[i][0][1]:02d})" for i in range(10)],
            "Tín Hiệu Đồng Bộ Logic": [f"{all_sorted[i][1]:.6f}" for i in range(10)]
        })
        st.table(df_top)
    else:
        st.warning(f"Cần tối thiểu 5 kỳ dữ liệu (100 số). Hệ thống hiện nhận diện được {total_kies} kỳ.")
else:
    st.info("Dán dữ liệu Keno vào khung trên để khởi chạy bộ lọc Chipset v7.0.")
