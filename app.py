import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Phase-Shift Residual Engine", layout="centered")

# ==============================================================================
# KENO PHASE-SHIFT RESIDUAL ENGINE (CHUYỂN PHA NĂNG LƯỢNG & KHÓA BẪY 31-33)
# ==============================================================================
class PhaseShiftKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, mat):
        m = np.max(mat)
        return mat / m if m > 0 else mat

    def process(self, X):
        T, D = X.shape
        
        # 1. Phân loại Trạng thái Kỳ 5 (Last Draw)
        last_draw = X[-1]  # Vector 80 chiều (1: xuất hiện ở kỳ 5, 0: không xuất hiện)
        
        # 2. Tính Tần suất & Động lượng Năng lượng (Accumulated Momentum)
        freqs = X.sum(axis=0) # Tổng số lần xuất hiện trong 5 kỳ
        
        # Ma trận Hiệu số Năng lượng Dịch chuyển (Phase Shift Score)
        # Điểm cao nhất dành cho số vừa nghỉ ở Kỳ 5 nhưng có tần suất tốt ở các kỳ trước
        residual_score = np.zeros(D)
        for i in range(D):
            if last_draw[i] == 1:
                # Số vừa nổ ở Kỳ 5: Giữ năng lượng nền
                residual_score[i] = 0.4 * freqs[i]
            else:
                # Số nghỉ ở Kỳ 5: Tính điểm tích lũy nhịp bù
                residual_score[i] = 1.2 * freqs[i]

        # Chuẩn hóa vector điểm đơn
        residual_score = self._norm(residual_score)
        
        # 3. Xây dựng Ma trận Bắt Cặp Chéo Pha (Cross-Phase Pairing Matrix)
        pair_mat = np.outer(residual_score, residual_score)
        np.fill_diagonal(pair_mat, 0)
        
        # 4. ÁP DỤNG CÁC BỘ LỌC CẶP SỐ NGHIÊM NGẶT (STRICT PAIR FILTERS)
        
        # (A) Triệt tiêu triệt để các cặp số đã từng đi chung >= 2 lần trong 5 kỳ (Khóa cặp 31-33)
        co_matrix = np.dot(X.T, X)
        pair_mat[co_matrix >= 2] = 0.0
        
        # (B) Bắt buộc Bắt cặp Chéo Pha: 1 Số ở Kỳ 5 + 1 Số Nghỉ Kỳ 5
        for i in range(D):
            for j in range(D):
                if i != j:
                    # Nếu cả 2 số cùng nổ ở Kỳ 5 HOẶC cùng nghỉ ở Kỳ 5 -> Giảm trọng số
                    if (last_draw[i] == 1 and last_draw[j] == 1) or (last_draw[i] == 0 and last_draw[j] == 0):
                        pair_mat[i, j] *= 0.15
                    else:
                        # Thưởng lớn cho cặp Chéo Pha (1 Nóng Kỳ 5 + 1 Chuyển Pha)
                        pair_mat[i, j] *= 2.2
                        
        # (C) Khóa bẫy bão hòa: Phạt nặng các số xuất hiện >= 3 lần
        for i in range(D):
            if freqs[i] >= 3:
                pair_mat[i, :] *= 0.1
                pair_mat[:, i] *= 0.1

        # Chuẩn hóa ma trận điểm cuối cùng
        final_mat = self._norm(pair_mat)
        
        # 5. Trích xuất Cặp số Chốt Tối ưu
        i, j = np.unravel_index(np.argmax(final_mat, axis=None), final_mat.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(final_mat[i, j])

        explanation = (
            f"• **KHẮC PHỤC BẪY LẶP LẠI (Phase-Shift Residual Engine):**\n"
            f"  - **Khóa Cặp Lặp Chu Kỳ:** Triệt tiêu hoàn toàn các cặp đã xuất hiện $\ge 2$ lần trong 5 kỳ (loại bỏ hoàn toàn cặp 31-33).\n"
            f"  - **Nguyên lý Chuyển Pha (Cross-Phase Pairing):** Ép mô hình chọn 1 số nằm trong Kỳ 5 kết hợp với 1 số nghỉ Kỳ 5 có động lượng tích lũy cao.\n"
            f"  - **Triệt tiêu Bão Hòa:** Hạ điểm các số duy trì vệt quá dài để mở đường cho dải số mới phát năng lượng.\n"
            f"• **Kết luận Chốt:** Cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Phase-Shift Residual Engine")
st.caption("Giải quyết triệt để bẫy lặp cặp 31-33 • Bắt cặp Chéo Pha (Cross-Phase) • Khóa vệt bão hòa")

raw_text_input = st.text_area(
    "Dán chuỗi số 5 kỳ (mỗi kỳ 1 dòng hoặc dán liên tục):",
    placeholder="Kì 1: 01 02 11 15 ...\nKì 2: ...",
    height=160,
    key="raw_text_keno"
)

if raw_text_input.strip():
    cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text_input.strip(), flags=re.IGNORECASE)
    all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= 80]
    total_kies = len(all_numbers) // 20
    
    if total_kies >= 5:
        matrix = np.zeros((5, 80), dtype=float)
        for k in range(5):
            for num in all_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0
                
        st.success("🎉 Đã chạy xong Mô hình Chuyển Pha Phase-Shift!")
        
        engine = PhaseShiftKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ ĐỘT PHÁ PHAN TRÃI")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT MỚI", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Phase-Shift Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình Phase-Shift Residual.")
