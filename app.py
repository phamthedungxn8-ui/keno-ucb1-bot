import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Hedged-Pairing Engine", layout="centered")

# ==============================================================================
# HEDGED-PAIRING QUANTUM ENGINE (ĐỘNG CƠ BẮT CẶP ĐỐI KHÁNG CHỐNG HỤT)
# ==============================================================================
class HedgedKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, mat):
        m = np.max(mat)
        return mat / m if m > 0 else mat

    # --------------------------------------------------------------------------
    # 1. HEDGED MOMENTUM VECTOR (VECTOR ĐỘNG LƯỢNG BÙ TRỪ NÓNG - TÍCH LŨY)
    # --------------------------------------------------------------------------
    def _engine_hedged_momentum(self, X):
        """Phân loại số Nóng (Hot) và số Tích lũy (Accumulator) để ghép cặp bù trừ"""
        T, D = X.shape
        last_frame = X[-1]
        prev_frame = X[-2] if T >= 2 else np.zeros(D)
        
        scores = np.zeros(D)
        status = np.zeros(D) # 1: Hot, 2: Accumulator, 0: Cold/Overheated
        
        for i in range(D):
            # Nhịp Tích Lũy Tối Ưu: Vừa nghỉ kỳ vừa rồi, nhưng nổ ở kỳ t-2 hoặc t-3
            if last_frame[i] == 0 and prev_frame[i] == 1:
                scores[i] = 1.0
                status[i] = 2 # Accumulator (Số tích lũy chuẩn bị nổ lại)
            # Nhịp Nóng Tối Ưu: Nổ kỳ vừa rồi, tần suất 2-3 lần/5 kỳ
            elif last_frame[i] == 1 and X[:, i].sum() <= 3:
                scores[i] = 0.8
                status[i] = 1 # Hot (Số đang trong luồng)
            else:
                scores[i] = 0.2
                status[i] = 0 # Quá bão hòa hoặc quá nguội
                
        # Ma trận bắt cặp: BẮT BỘC 1 số Hot (1) đi với 1 số Accumulator (2)
        hedged_mat = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                if i != j:
                    # Thưởng điểm cao nhất nếu ghép 1 Hot + 1 Accumulator
                    if (status[i] == 1 and status[j] == 2) or (status[i] == 2 and status[j] == 1):
                        hedged_mat[i, j] = scores[i] * scores[j] * 2.0
                    elif status[i] == status[j] and status[i] != 0:
                        # Phạt nếu ghép 2 số cùng loại (2 Hot hoặc 2 Accumulator)
                        hedged_mat[i, j] = scores[i] * scores[j] * 0.3
                        
        return self._norm(hedged_mat)

    # --------------------------------------------------------------------------
    # 2. CLUSTER DECAY FILTER (LỌC PHÂN RÃ CỤM NỔ CÙNG BẠN)
    # --------------------------------------------------------------------------
    def _engine_cluster_decay(self, X):
        """Triệt tiêu các cặp đã đi chung với nhau quá nhiều ở 3 kỳ gần nhất"""
        T, D = X.shape
        co_recent = np.dot(X[-3:].T, X[-3:]) # Đồng xuất hiện 3 kỳ gần nhất
        
        decay_mat = np.ones((D, D))
        # Nếu đã đi chung với nhau >= 2 lần trong 3 kỳ gần đây -> Phạt nặng vì cụm đã phân rã
        decay_mat[co_recent >= 2] = 0.10
        decay_mat[co_recent >= 3] = 0.00
        
        return decay_mat

    # --------------------------------------------------------------------------
    # 3. GAP-INTERVAL DYNAMIC (KHOẢNG CÁCH NHỊP TẬP TRUNG)
    # --------------------------------------------------------------------------
    def _engine_gap_interval(self, X):
        T, D = X.shape
        gaps = np.zeros(D)
        for i in range(D):
            # Tính số kỳ nghỉ liên tiếp tính từ kỳ gần nhất
            idx = np.where(X[:, i] == 1)[0]
            if len(idx) > 0:
                gaps[i] = (T - 1) - idx[-1]
            else:
                gaps[i] = T
                
        # Ưu tiên ghép số có Gap = 0 (vừa nổ) với số có Gap = 1 (nghỉ 1 kỳ)
        gap_mat = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                if (gaps[i] == 0 and gaps[j] == 1) or (gaps[i] == 1 and gaps[j] == 0):
                    gap_mat[i, j] = 1.0
                elif gaps[i] == 0 and gaps[j] == 0:
                    gap_mat[i, j] = 0.2 # Phạt 2 số cùng vừa nổ
                    
        return gap_mat

    # --------------------------------------------------------------------------
    # PROCESSOR TỔNG HỢP VỚI CƠ CHẾ BẮT CẶP ĐỐI KHÁNG
    # --------------------------------------------------------------------------
    def process(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)

        # Chạy 3 mô hình bù trừ
        e_hedged = self._engine_hedged_momentum(X)
        e_decay = self._engine_cluster_decay(X)
        e_gap = self._engine_gap_interval(X)

        # TỔNG HỢP CỘNG HƯỞNG BÙ TRỪ (HEDGED AGGREGATION)
        fused = (2.5 * e_hedged + 1.8 * e_gap) * e_decay

        np.fill_diagonal(fused, 0)

        # Trích xuất Cặp Bậc 2 Tối Ưu
        i, j = np.unravel_index(np.argmax(fused, axis=None), fused.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(fused[i, j])

        explanation = (
            f"• **Chiến lược Bắt Cặp Đối Kháng (Hedged Pairing):**\n"
            f"  - *Cấu trúc chọn cặp:* Ghép **1 số Nóng (Vừa nổ kỳ t-1)** + **1 số Tích Lũy (Nghỉ 1 kỳ, chuẩn bị quay lại)**.\n"
            f"  - *Triệt tiêu Phân rã Cụm (Cluster Decay):* Đã loại bỏ các cặp số dính liền từng nổ chung quá nhiều ở các kỳ trước (như 31-33 hay 25-26).\n"
            f"  - *Chỉ số Khoảng cách Nhịp (Gap Dynamic):* Đảm bảo nhịp xuất hiện của 2 số bù trừ rủi ro cho nhau.\n"
            f"• **Kết luận Chốt:** Cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Hedged-Pairing Engine")
st.caption("Khắc phục bẫy hụt 1 con • Ghép cặp Nóng - Tích lũy • Phân rã cụm lặp")

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
                
        st.success("🎉 Đã chạy xong Động cơ Bắt cặp Đối kháng!")
        
        engine = HedgedKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ CHỐT BÙ TRỪ TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Bù Trừ Hedged Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình bù trừ.")
