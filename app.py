import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Markov-Entropy Physics Engine", layout="centered")

# ==============================================================================
# MARKOV-ENTROPY QUANTUM ENGINE (ĐỘNG CƠ CHUYỂN PHA VÀ CÂN BẰNG ENTROPY)
# ==============================================================================
class MarkovEntropyKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, vec):
        m = np.max(vec)
        return vec / m if m > 0 else vec

    # --------------------------------------------------------------------------
    # 1. MARKOV STATE TRANSITION MATRIX (MA TRẬN CHUYỂN DỊCH TRẠNG THÁI MARKOV)
    # --------------------------------------------------------------------------
    def _engine_markov_transitions(self, X):
        """Tính ma trận xác suất chuyển đổi trạng thái từ kỳ t-1 sang kỳ t"""
        T, D = X.shape
        trans_mat = np.zeros((D, D))
        
        for t in range(T - 1):
            # Các số xuất hiện ở kỳ t
            active_t = np.where(X[t] == 1)[0]
            # Các số xuất hiện ở kỳ t+1
            active_next = np.where(X[t+1] == 1)[0]
            
            for i in active_t:
                for j in active_next:
                    trans_mat[i, j] += 1.0

        # Chuẩn hóa xác suất Markov dòng
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        prob_trans = trans_mat / row_sums
        
        # Dự báo trạng thái kỳ kế tiếp dựa trên kỳ cuối X[-1]
        last_active = np.where(X[-1] == 1)[0]
        future_state_prob = prob_trans[last_active].sum(axis=0)
        
        return self._norm(future_state_prob)

    # --------------------------------------------------------------------------
    # 2. INFORMATION ENTROPY COMPRESSION (LƯỢNG TIN NÉN & CHUYỂN PHA)
    # --------------------------------------------------------------------------
    def _engine_entropy_compression(self, X):
        """Đo độ hỗn loạn Entropy của từng vùng số để tìm điểm hội tụ năng lượng"""
        T, D = X.shape
        entropy_scores = np.zeros(D)
        
        for i in range(D):
            p1 = np.mean(X[:, i]) # Xác suất xuất hiện
            p0 = 1.0 - p1
            
            if 0 < p1 < 1:
                # Công thức Shannon Entropy: H(X) = - (p0 log2 p0 + p1 log2 p1)
                h = - (p0 * np.log2(p0) + p1 * np.log2(p1))
            else:
                h = 0.0 # Entropy cực tiểu (Trạng thái nén tuyệt đối)
                
            # Điểm nén năng lượng: Entropy thấp + có xuất hiện trong 2 kỳ gần nhất
            compression = (1.0 - h) * (1.5 if X[-1, i] == 1 or X[-2, i] == 1 else 0.5)
            entropy_scores[i] = compression
            
        return self._norm(entropy_scores)

    # --------------------------------------------------------------------------
    # 3. PHASE TRANSITION CO-PAIRING (MA TRẬN CỘNG HƯỞNG CHUYỂN PHA CẶP)
    # --------------------------------------------------------------------------
    def _engine_phase_co_resonance(self, X, markov_vec, entropy_vec):
        """Xây dựng ma trận điểm cặp dựa trên sự cân bằng Entropy và nhịp Markov"""
        D = self.D
        base_pair = np.outer(markov_vec, entropy_vec)
        
        # Ma trận đối xứng cộng hưởng 2 chiều
        resonance = (base_pair + base_pair.T) / 2.0
        np.fill_diagonal(resonance, 0)
        
        # Phạt các cặp bão hòa (vừa xuất hiện cùng nhau ở kỳ cuối)
        last_co = np.outer(X[-1], X[-1])
        resonance[last_co == 1] *= 0.15 # Phạt 85% nguy cơ rẽ nhánh hỗn loạn
        
        # Phạt các cặp lặp lại >= 3 kỳ
        total_co = np.dot(X.T, X)
        resonance[total_co >= 3] *= 0.05
        
        return self._norm(resonance)

    # --------------------------------------------------------------------------
    # PROCESSOR TỔNG HỢP HỆ THỐNG MARKOV - ENTROPY
    # --------------------------------------------------------------------------
    def process(self, X):
        T, D = X.shape

        # 1. Tính toán Mạng Markov và Entropy Lượng tin
        vec_markov = self._engine_markov_transitions(X)
        vec_entropy = self._engine_entropy_compression(X)

        # 2. Ma trận Chuyển pha Cặp
        pair_mat = self._engine_phase_co_resonance(X, vec_markov, vec_entropy)

        # 3. Trích xuất Cặp số Chuyển Pha Tối Ưu
        i, j = np.unravel_index(np.argmax(pair_mat, axis=None), pair_mat.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(pair_mat[i, j])

        explanation = (
            f"• **MÔ HÌNH CHUYỂN PHA MARKOV - ENTROPY (System Physics):**\n"
            f"  - **Xích Markov State Matrix:** Dự báo chuyển dịch xác suất trạng thái từ kỳ $t-1$ sang kỳ tiếp theo.\n"
            f"  - **Lượng tin Shannon Entropy:** Lọc vùng số có độ hỗn loạn giảm (năng lượng đang nén hội tụ).\n"
            f"  - **Khử Rẽ Nhánh Hỗn Loạn (Bifurcation Gate):** Loại bỏ bẫy bão hòa của các cặp số vừa nổ chung ở trạng thái Entropy cao.\n"
            f"• **Kết luận Chốt:** Cặp số **({num1:02d}, {num2:02d})** đạt điểm chuyển pha hệ thống cao nhất."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Markov-Entropy Physics Engine")
st.caption("Lý thuyết Hệ thống Động lực • Xích Markov Chuyển dịch Trạng thái • Shannon Entropy Compression")

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
                
        st.success("🎉 Đã chạy xong Động cơ Chuyển pha Markov - Entropy!")
        
        engine = MarkovEntropyKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ CHUYỂN PHA TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT MARKOV", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Cân Bằng State Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt động cơ Markov-Entropy.")
