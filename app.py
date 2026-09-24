import re
import numpy as np
import pandas as pd
import streamlit as st

# Cấu hình trang Streamlit tối giản
st.set_page_config(page_title="Keno Pattern Agent", layout="centered")

# ==============================================================================
# 1. AGENT QUÉT DẤU VẾT KÉO THEO (PATTERN ENTANGLEMENT AGENT)
# ==============================================================================
class PatternEntanglementAgent:
    """
    Agent tự động phát hiện 'điểm lạ' trong chuỗi lặp:
    1. Tìm các con số lặp lại (Core Hot Numbers) trong cửa sổ 5 kỳ.
    2. Quét lực kéo (Follow Impulse) giữa các số lặp và các số xuất hiện xung quanh.
    3. Tự chốt Cặp Bậc 2 có tính liên kết 'kéo theo' cao nhất.
    """
    def __init__(self, num_dim=80):
        self.D = num_dim

    def discover_entangled_pair(self, matrix):
        # matrix có dạng (5, 80) - dòng là kỳ, cột là số (0 hoặc 1)
        T, D = matrix.shape
        
        # 1. Đếm tần suất xuất hiện của từng số trong 5 kỳ
        freqs = matrix.sum(axis=0) # Mảng 80 phần tử
        
        # 2. Xกำหนด Hạt nhân lặp (Core Numbers xuất hiện từ 2 lần trở lên)
        max_freq = np.max(freqs)
        if max_freq < 2:
            # Nếu không có số nào lặp, lấy 2 số có chỉ số xuất hiện cao nhất
            top_indices = np.argsort(freqs)[-2:]
            return (int(top_indices[0] + 1), int(top_indices[1] + 1)), 1.0, "Mẫu ngẫu nhiên (Không phát hiện số lặp đặc biệt)"
        
        hot_indices = np.where(freqs >= 2)[0] # Danh sách các số lặp
        
        # 3. Tính Lực Kéo Theo (Co-occurrence & Lead-Follow Dynamics)
        entanglement_matrix = np.zeros((D, D))
        
        for t in range(T):
            active_nums = np.where(matrix[t] == 1)[0]
            for i in active_nums:
                for j in active_nums:
                    if i != j:
                        # Trọng số gia tăng nếu i hoặc j là 'Số Lặp'
                        weight = 1.0
                        if i in hot_indices: weight += 1.5
                        if j in hot_indices: weight += 1.5
                        entanglement_matrix[i, j] += weight

        # Xóa đường chéo chính (không xét cặp số trùng nhau)
        np.fill_diagonal(entanglement_matrix, 0)
        
        # 4. Tìm cặp (i, j) có điểm Kéo theo cao nhất
        i, j = np.unravel_index(np.argmax(entanglement_matrix, axis=None), entanglement_matrix.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        score = float(entanglement_matrix[i, j])
        
        hot_nums_str = ", ".join([f"{h+1:02d}" for h in hot_indices])
        detail_msg = f"Số lặp phát hiện: [{hot_nums_str}] ➔ Kéo theo cặp ({num1:02d}, {num2:02d})"
        
        return (num1, num2), score, detail_msg

# ==============================================================================
# 2. GIAO DIỆN TỰ ĐỘNG BÓC TÁCH & HIỂN THỊ KẾT QUẢ TRỰC TIẾP
# ==============================================================================
st.title("⚡ Keno Agent: Dấu Vết Kéo Theo")
st.caption("Tự động bóc tách chuỗi số 5 kỳ • Khám phá mẫu lặp & Lực kéo giữa các con số")

# Khung dán dữ liệu duy nhất
raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 04 06 10 18 22 26 35 39 40 45 46 48 49 51 52 59 68 69 70 79\nKì 2: ...",
    height=160,
    key="raw_text_keno"
)

# TỰ ĐỘNG XỬ LÝ NGAY KHI CÓ DỮ LIỆU DÁN VÀO
if raw_text_input.strip():
    raw_data = raw_text_input.strip()
    
    # 1. Bóc tách và làm sạch dữ liệu bằng Regex (Loại bỏ chữ 'Kì X:', 'Kỳ X:')
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_data, flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        # Lấy đúng 100 số (5 kỳ mới nhất)
        valid_numbers = all_numbers[:100]
        
        # Dựng Ma trận Trạng thái 5x80 (Binary State Matrix)
        matrix = np.zeros((5, 80), dtype=float)
        for k in range(5):
            ky_nums = valid_numbers[k * 20 : (k + 1) * 20]
            for num in ky_nums:
                matrix[k, num - 1] = 1.0
                
        st.success(f"🎉 Agent đã phân tách thành công 5 kỳ lịch sử (100 con số)!")
        
        # 2. CHẠY AGENT QUÉT DẤU VẾT & XUẤT KẾT QUẢ
        agent = PatternEntanglementAgent(num_dim=80)
        best_pair, score, explanation = agent.discover_entangled_pair(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ BẬC 2 PHÁT HIỆN BỞI AGENT")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="CẶP SỐ BẬC 2 TỐI ƯU",
                value=f"{best_pair[0]:02d} — {best_pair[1]:02d}"
            )
        with col2:
            st.metric(
                label="Chỉ Số Lực Kéo (Entanglement)",
                value=f"{score:.1f}"
            )
            
        st.info(f"🔍 **Phán đoán của Agent:** {explanation}")
        
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Hãy dán chuỗi số 5 kỳ vào khung phía trên để Agent tự động bóc tách và phân tích.")
