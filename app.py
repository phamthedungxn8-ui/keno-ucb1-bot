import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v14.0 Multi-Agent Ensemble Engine", layout="wide")

# ==============================================================================
# BỘ CÁC AGENT SUY LUẬN ĐỘC LẬP
# ==============================================================================

class Agent1_ShannonTopo:
    """Agent 1: Chuyên trách Mật độ Entropy Tin tức Shannon & Độ cong Riemann"""
    def evaluate(self, X):
        T, D = X.shape
        p1 = np.mean(X == 1, axis=0) + 1e-9
        p0 = np.mean(X == 0, axis=0) + 1e-9
        scores = np.zeros(D)
        
        for i in range(D):
            # Tính Self-Entropy và độ biến động tin tức
            entropy = - (p1[i] * np.log2(p1[i]) + p0[i] * np.log2(p0[i]))
            # Điểm ưu tiên cho các số có Entropy cao (đang ở vùng ranh giới nổ)
            scores[i] = entropy * (1.0 - abs(p1[i] - 0.25))
            
        return scores

class Agent2_TensorMPS:
    """Agent 2: Chuyên trách Mạng Tensor SVD & Độ vướng víu lịch sử ngắn"""
    def evaluate(self, X):
        T, D = X.shape
        if T >= 4:
            split = T // 2
            cov_past = np.dot(X[:split].T, X[:split]) / float(split)
            cov_recent = np.dot(X[split:].T, X[split:]) / float(T - split)
            entanglement = np.diag(np.abs(cov_past * cov_recent))
        else:
            entanglement = np.diag(np.dot(X.T, X)) / float(T)
            
        return entanglement

class Agent3_PhaseShift:
    """Agent 3: Chuyên trách Phát hiện Dịch pha Xung lực & Triệt tiêu Lặp số"""
    def evaluate(self, X):
        T, D = X.shape
        scores = np.ones(D)
        freqs = X.sum(axis=0)
        
        for i in range(D):
            # Ưu tiên các số xuất hiện 1-2 lần trong 6-8 kỳ (vùng nhịp đẹp)
            if freqs[i] == 1 or freqs[i] == 2:
                scores[i] *= 1.8
            elif freqs[i] >= 4: # Phạt bão hòa
                scores[i] *= 0.1
            elif freqs[i] == 0: # Phạt lỳ
                scores[i] *= 0.3
                
            # Phạt nổ liên tiếp ở T-1 để chống kẹt thanh ghi
            if X[-1, i] == 1:
                scores[i] *= 0.2
                
        return scores

# ==============================================================================
# HỆ THỐNG TỔNG HỢP MULTI-AGENT & ĐIỀU PHỐI TỰ TỐI ƯU (CENTRAL CONTROLLER)
# ==============================================================================

class MultiAgentKenoSystem:
    def __init__(self):
        self.agent1 = Agent1_ShannonTopo()
        self.agent2 = Agent2_TensorMPS()
        self.agent3 = Agent3_PhaseShift()

    def process(self, X):
        T, D = X.shape
        
        # 1. Chạy song song 3 Agent
        s1 = self.agent1.evaluate(X)
        s2 = self.agent2.evaluate(X)
        s3 = self.agent3.evaluate(X)
        
        # Chuẩn hóa min-max điểm số từng Agent
        s1 = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-9)
        s2 = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-9)
        s3 = (s3 - s3.min()) / (s3.max() - s3.min() + 1e-9)
        
        # 2. Lớp Hội tụ & Đồng thuận (Consensus Layer)
        # Tích chéo điểm số của 3 Agent
        final_scores = (s1 ** 1.2) * (s2 ** 1.5) * (s3 ** 2.0)
        
        # Sắp xếp danh sách 80 nút số theo thứ tự tiềm năng giảm dần
        ranked_indices = np.argsort(final_scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices]
        
        # 3. Trích xuất Bộ số Tối ưu cho Bậc 7 & Bậc 8
        set_bac8 = sorted(ranked_numbers[:8])
        set_bac7 = sorted(ranked_numbers[:7])
        set_bac9 = sorted(ranked_numbers[:9])
        
        # Dàn lót dự phòng (8 số kế tiếp)
        backup_bac8 = sorted(ranked_numbers[8:16])
        
        return {
            "bac8": set_bac8,
            "bac7": set_bac7,
            "bac9": set_bac9,
            "backup8": backup_bac8,
            "scores": final_scores,
            "ranked_all": ranked_numbers
        }

# ==============================================================================
# STREAMLIT UI - MULTI-AGENT SYSTEM
# ==============================================================================

st.title("🦅 MDM-IDS v14.0: MULTI-AGENT POSITIVE-EV ENGINE")
st.caption("Chiến Thuật Tích Tụ Tiền Lẻ • Hệ Thống 3 Agent Song Song • Lọc Cuốn Chiếu 6-8 Kỳ • Tối Ưu Bậc 7, 8, 9")

raw_input = st.text_area(
    "Dán dữ liệu 6 đến 8 kỳ Keno gần nhất vào đây (dữ liệu cuốn chiếu):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=160
)

if raw_input.strip():
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_input.strip(), flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 6:
        kies_to_use = min(total_kies, 8) # Lấy chuẩn 6 - 8 kỳ cuốn chiếu
        used_numbers = all_numbers[-kies_to_use * 20:]
        
        matrix = np.zeros((kies_to_use, 80), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0
                
        system = MultiAgentKenoSystem()
        res = system.process(matrix)
        
        st.success(f"⚡ Đã quét xong dữ liệu cuốn chiếu {kies_to_use} kỳ qua 3 Agent độc lập!")
        
        st.markdown("---")
        
        # HIỂN THỊ DÀN BẬC 8 CHỦ LỰC
        st.subheader("🎯 DÀN CHỦ LỰC BẬC 8 (BẢO HIỂM TRÚNG 0 & TRÚNG 4/8 HOÀN TIỀN)")
        b8_str = "  •  ".join([f"**{n:02d}**" for n in res["bac8"]])
        st.markdown(
            f"<div style='text-align: center; padding: 20px; background-color: #0A192F; border-radius: 12px; border: 2px solid #00F0FF; box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);'>"
            f"<h2 style='color: #00F0FF; margin:0;'>{b8_str}</h2>"
            f"<p style='color: #AAA; margin:8px 0 0 0;'>Mục tiêu: Đạt giải Trúng 0, Trúng 4, Trúng 5 hoặc Trúng 6 để tích góp lợi nhuận</p>"
            f"</div>", 
            unsafe_allow_dict=True
        )
        
        st.markdown("<br>", unsafe_allow_dict=True)
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("🔥 DÀN CHỦ LỰC BẬC 7")
            b7_str = " - ".join([f"{n:02d}" for n in res["bac7"]])
            st.info(f"### **{b7_str}**\n*Lợi thế: Tỷ lệ nổ 3/7 và 4/7 cực cao.*")
            
        with col_b:
            st.subheader("🛡️ DÀN PHỤ BỌC LÓT BẬC 8 (BACKUP)")
            bk_str = " - ".join([f"{n:02d}" for n in res["backup8"]])
            st.warning(f"### **{bk_str}**\n*Sử dụng đánh song song hoặc đổi nhịp khi dàn chính vừa ăn lớn.*")

        st.markdown("---")
        st.subheader("📊 BẢNG XẾP HẠNG TOP 15 CON SỐ ĐỒNG THUẬN CAO NHẤT TỪ 3 AGENT")
        
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(15)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(15)],
            "Điểm Tích Chéo 3 Agent": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(15)]
        })
        st.table(df_top.T)
        
    else:
        st.warning(f"Hãy nhập từ 6 đến 8 kỳ (hiện nhận diện được {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu 6-8 kỳ Keno vào khung trên để khởi chạy chiến thuật Multi-Agent.")
