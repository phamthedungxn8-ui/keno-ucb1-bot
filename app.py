import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Self-Optimizing Ensemble Engine", layout="centered")

# ==============================================================================
# SELF-OPTIMIZING MULTI-LAYER KENO ENGINE
# ==============================================================================
class AutoOptimizingKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, vec):
        m = np.max(vec)
        return vec / m if m > 0 else vec

    # --------------------------------------------------------------------------
    # 1. BASE ENGINES (CÁC THUẬT TOÁN THÀNH PHẦN)
    # --------------------------------------------------------------------------
    def _engine_gnn(self, X):
        """Spatial Graph Convolution"""
        T, D = X.shape
        A = np.dot(X.T, X)
        np.fill_diagonal(A, 0)
        deg = np.sum(A, axis=1)
        deg_inv = np.power(deg, -0.5, where=deg>0)
        deg_inv[deg == 0] = 0
        D_mat = np.diag(deg_inv)
        L = np.dot(np.dot(D_mat, A), D_mat)
        
        h = np.tanh(np.dot(L, X[-1]))
        return self._norm(h)

    def _engine_attention(self, X):
        """Temporal Transformer Attention"""
        T, D = X.shape
        scores = np.dot(X, X.T) / np.sqrt(D)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        ctx = np.dot(attn, X)
        return self._norm(np.mean(ctx, axis=0))

    def _engine_markov(self, X):
        """Markov State Transition Probability"""
        T, D = X.shape
        trans = np.dot(X[:-1].T, X[1:])
        r_sum = trans.sum(axis=1, keepdims=True)
        r_sum[r_sum == 0] = 1.0
        p_trans = trans / r_sum
        return self._norm(p_trans[np.where(X[-1] == 1)[0]].sum(axis=0))

    def _engine_kalman(self, X):
        """Kalman Momentum Accumulation"""
        T, D = X.shape
        rates = np.zeros(D)
        for i in range(D):
            x_hat, P, Q, R = 0.25, 1.0, 0.05, 0.2
            for t in range(T):
                P += Q
                K = P / (P + R)
                x_hat += K * (X[t, i] - x_hat)
                P = (1 - K) * P
            rates[i] = x_hat
        return self._norm(rates)

    # --------------------------------------------------------------------------
    # 2. AUTO-OPTIMIZATION & BACKTESTING MODULE (VÒNG LẶP TỰ TỐI ƯU TRỌNG SỐ)
    # --------------------------------------------------------------------------
    def _auto_optimize_weights(self, X):
        """Thử nghiệm ngược trên 4 kỳ đầu để đánh giá và tối ưu trọng số thuật toán"""
        X_train = X[:-1] # 4 kỳ đầu
        y_true = X[-1]   # Kỳ thứ 5 làm nhãn kiểm chứng
        
        # Chạy thử 4 thuật toán trên dữ liệu huấn luyện
        pred_gnn = self._engine_gnn(X_train)
        pred_attn = self._engine_attention(X_train)
        pred_markov = self._engine_markov(X_train)
        pred_kalman = self._engine_kalman(X_train)
        
        preds = [pred_gnn, pred_attn, pred_markov, pred_kalman]
        names = ["GNN Graph", "Transformer Attention", "Markov Transition", "Kalman Momentum"]
        
        # Tính độ khớp (Cross-Entropy Loss / Hit Rate Score)
        scores = []
        for p in preds:
            # Đo tỷ lệ khớp giữa vector dự báo p và thực tế y_true
            hit_score = np.dot(p, y_true) / (np.sum(y_true) + 1e-5)
            scores.append(hit_score)
            
        scores = np.array(scores)
        # Softmax để quy đổi thành Trọng số Động (Dynamic Weights)
        exp_w = np.exp(scores * 3.0) # Scale factor = 3.0
        weights = exp_w / np.sum(exp_w)
        
        return weights, names

    # --------------------------------------------------------------------------
    # 3. FINAL ENSEMBLE & PAIR SELECTION (TỔNG HỢP VÀ CHỐT CẶP SỐ)
    # --------------------------------------------------------------------------
    def process(self, X):
        # 1. Tự động thử nghiệm và tối ưu trọng số
        opt_weights, engine_names = self._auto_optimize_weights(X)
        
        # 2. Chạy toàn bộ mô hình trên đủ 5 kỳ
        p_gnn = self._engine_gnn(X)
        p_attn = self._engine_attention(X)
        p_markov = self._engine_markov(X)
        p_kalman = self._engine_kalman(X)
        
        # 3. Tổng hợp vector dự báo theo Trọng số Tối ưu
        fused_vec = (
            opt_weights[0] * p_gnn +
            opt_weights[1] * p_attn +
            opt_weights[2] * p_markov +
            opt_weights[3] * p_kalman
        )
        fused_vec = self._norm(fused_vec)
        
        # 4. Ma trận Bắt Cặp Bù Trừ (Hedged Pair Matrix)
        pair_mat = np.outer(fused_vec, fused_vec)
        np.fill_diagonal(pair_mat, 0)
        
        # Phạt nặng các cặp số vừa nổ chung ở kỳ cuối (Tránh bẫy bão hòa)
        last_co = np.outer(X[-1], X[-1])
        pair_mat[last_co == 1] *= 0.1
        
        # Ưu tiên cặp số có nhịp bù trừ (1 Nóng + 1 Tích lũy)
        for i in range(self.D):
            for j in range(self.D):
                if i != j and (X[-1, i] != X[-1, j]):
                    pair_mat[i, j] *= 1.8
                    
        # Trích xuất Cặp số Chốt
        i, j = np.unravel_index(np.argmax(pair_mat, axis=None), pair_mat.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(pair_mat[i, j])

        # Báo cáo trọng số đã tự tối ưu
        weight_report = "\n".join([f"  - **{names}:** {w*100:.1f}%" for names, w in zip(engine_names, opt_weights)])
        
        explanation = (
            f"• **KẾT QUẢ TỰ ĐỘNG THỬ NGHIỆM & TỐI ƯU TRỌNG SỐ (AutoML Weight Adaptation):**\n"
            f"{weight_report}\n"
            f"• **Cơ chế Tổng hợp:** Hệ thống đã tự động chạy Backtest trên 4 kỳ đầu, đối chiếu với kỳ 5 để đánh giá thuật toán nào đang hoạt động tốt nhất và tự điều chỉnh phân bổ trọng số.\n"
            f"• **Kết luận Chốt:** Cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Self-Optimizing Ensemble Engine")
st.caption("Tự động Thử nghiệm • Tự tối ưu Trọng số Thuật toán • AutoML Backtesting Loop")

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
                
        st.success("🎉 Đã chạy xong Mô hình Tự tối ưu Trọng số!")
        
        engine = AutoOptimizingKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ CHỐT TỰ ĐỘNG TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT TỐI ƯU", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Auto-Ensemble Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình tự động tối ưu.")
