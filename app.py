import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Keno Hyper-Layered Deep Engine", layout="centered")

# ==============================================================================
# HỆ THỐNG ĐA TẦNG ĐAN XEN PHÂN TÍCH CHỐNG HỤT SỐ (HYPER-LAYERED ENGINE)
# ==============================================================================
class HyperLayeredKenoEngine:
    def __init__(self, num_dim=80):
        self.D = num_dim

    def _norm(self, mat):
        m = np.max(mat)
        return mat / m if m > 0 else mat

    # --------------------------------------------------------------------------
    # 1. MARKOV CONDITIONAL PAIR TRANSITION (XÁC SUẤT CHUYỂN ĐỔI ĐIỀU KIỆN)
    # --------------------------------------------------------------------------
    def _engine_markov_conditional(self, X):
        """Đo xác suất bùng nổ kỳ tiếp theo dựa trên trạng thái kỳ cuối (t-1)"""
        T, D = X.shape
        markov_mat = np.zeros((D, D))
        last_state = X[-1] # Trạng thái kỳ liền trước
        
        # Điểm động lượng đơn lẻ từ Markov
        single_momentum = np.zeros(D)
        for i in range(D):
            # Nếu vừa nổ ở kỳ cuối -> Nhịp lặp lại (Rebound)
            # Nếu nghỉ ở kỳ cuối nhưng nổ ở t-2 -> Nhịp nhả lại (Rest-Release)
            if last_state[i] == 1:
                single_momentum[i] = 0.4 # Phạt nhẹ nhịp vừa nổ để tránh bị bẫy
            elif T >= 2 and X[-2, i] == 1:
                single_momentum[i] = 1.0 # Thưởng mạnh cho nhịp vừa nghỉ 1 kỳ
            else:
                single_momentum[i] = 0.6

        markov_mat = np.outer(single_momentum, single_momentum)
        np.fill_diagonal(markov_mat, 0)
        return self._norm(markov_mat)

    # --------------------------------------------------------------------------
    # 2. CROSS-EXCLUSION & CONSECUTIVE PENALTY (MÀNG KHỬ LẶP ĐỒNG THỜI)
    # --------------------------------------------------------------------------
    def _engine_consecutive_repulsion(self, X):
        """Loại bỏ bẫy: Cặp vừa nổ chung ở kỳ t-1 thì KHÔNG CHỌN LẠI ở kỳ t"""
        T, D = X.shape
        repulsion_mat = np.ones((D, D))
        
        if T >= 1:
            last_frame = X[-1]
            # Lấy ma trận đồng xuất hiện chỉ riêng ở kỳ gần nhất
            last_co = np.outer(last_frame, last_frame)
            # Phạt 90% nếu cặp số này VỪA NỔ CHUNG ở kỳ gần nhất
            repulsion_mat[last_co == 1] = 0.10
            
        return repulsion_mat

    # --------------------------------------------------------------------------
    # 3. GRAPH CLUSTER DENSITY (MẠNG ĐỒ THỊ CỤM LIÊN KẾT KHÔNG GIANG)
    # --------------------------------------------------------------------------
    def _engine_graph_clustering(self, X):
        """Phát hiện cụm số dính liền (ví dụ dải 21-22-24-25)"""
        T, D = X.shape
        co_occur = np.dot(X.T, X)
        
        # Ma trận khoảng cách số (Tần số xuất hiện gần nhau về mặt vị trí 1-80)
        adj_mat = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                if abs(i - j) <= 3 and i != j: # Liền kề khoảng cách <= 3
                    adj_mat[i, j] = 1.0
                    
        # Cộng hưởng giữa tần suất đồng xuất hiện và khoảng cách vị trí
        cluster_mat = co_occur * adj_mat
        return self._norm(cluster_mat)

    # --------------------------------------------------------------------------
    # 4. KALMAN LATENT VELOCITY (VẬN TỐC TẦNG ẨN KALMAN)
    # --------------------------------------------------------------------------
    def _engine_kalman_velocity(self, X):
        T, D = X.shape
        velocities = np.zeros(D)
        for i in range(D):
            x_hat, P, Q, R = 0.25, 1.0, 0.05, 0.2
            for t in range(T):
                P += Q
                K = P / (P + R)
                x_hat += K * (X[t, i] - x_hat)
                P = (1 - K) * P
            velocities[i] = x_hat
        
        v_mat = np.outer(velocities, velocities)
        np.fill_diagonal(v_mat, 0)
        return self._norm(v_mat)

    # --------------------------------------------------------------------------
    # PROCESSOR TỔNG HỢP VỚI LỚP PHÂN TÍCH ĐAN XEN ĐA TẦNG
    # --------------------------------------------------------------------------
    def process(self, X):
        T, D = X.shape
        freqs = X.sum(axis=0)
        co_occur = np.dot(X.T, X)

        # Trích xuất 4 tầng phân tích
        e_markov = self._engine_markov_conditional(X)
        e_repulsion = self._engine_consecutive_repulsion(X)
        e_cluster = self._engine_graph_clustering(X)
        e_kalman = self._engine_kalman_velocity(X)

        # Tầng Tương quan chéo Lag-0 chuẩn hóa
        norm_X = X - np.mean(X, axis=0)
        cross_corr = np.maximum(np.dot(norm_X.T, norm_X) / T, 0)
        np.fill_diagonal(cross_corr, 0)
        e_cross_corr = self._norm(cross_corr)

        # MÀNG LỌC PHẠT TỔNG HỢP (ANTI-SATURATION GATEWAY)
        penalty = np.ones((D, D))
        penalty[co_occur >= 3] = 0.05  # Phạt 95% nếu lặp >= 3 kỳ
        penalty[co_occur >= 4] = 0.00  # Khóa vĩnh viễn nếu lặp >= 4 kỳ

        # TỔNG HỢP TƯƠNG TÁC ĐAN XEN (INTERLACED AGGREGATION)
        # Điểm = (Markov + Kalman + CrossCorr + Cluster) * Lực cản nổ lặp * Phạt quá tải
        fused = (
            2.2 * e_markov + 
            2.0 * e_kalman + 
            1.8 * e_cross_corr + 
            1.5 * e_cluster
        ) * e_repulsion * penalty

        np.fill_diagonal(fused, 0)

        # Trích xuất Cặp Bậc 2 Tối Ưu
        i, j = np.unravel_index(np.argmax(fused, axis=None), fused.shape)
        num1, num2 = sorted([int(i + 1), int(j + 1)])
        final_score = float(fused[i, j])

        hot_str = ", ".join([f"{h+1:02d}" for h in np.where(freqs >= 2)[0]]) or "Không có"

        explanation = (
            f"• **Hạt nhân lặp ghi nhận:** [{hot_str}]\n"
            f"• **Lớp Phân Tích Đan Xen Đa Tầng:**\n"
            f"  - *Markov Conditional:* Ưu tiên nhịp vừa nghỉ 1 kỳ hơn nhịp vừa nổ lặp.\n"
            f"  - *Consecutive Repulsion:* Khóa triệt để các cặp vừa nổ chung ở kỳ vừa rồi (như 25-26).\n"
            f"  - *Graph Clustering:* Bắt tín hiệu cụm số liên kề không gian.\n"
            f"  - *Kalman Velocity:* Ước lượng vận tốc bùng nổ tiềm năng tầng ẩn.\n"
            f"• **Kết luận Chốt:** Cặp số **({num1:02d}, {num2:02d})**."
        )

        return (num1, num2), final_score, explanation

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("⚡ Keno Hyper-Layered Deep Engine")
st.caption("Phân tích đan xen đa tầng • Khử bẫy lặp cặp liền kề • Graph Clustering & Markov Conditional")

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
                
        st.success("🎉 Đã chạy xong Hệ thống Phân tích Đan xen Đa tầng!")
        
        engine = HyperLayeredKenoEngine(num_dim=80)
        best_pair, score, explanation = engine.process(matrix)
        
        st.markdown("---")
        st.subheader("🎯 CẶP SỐ CHỐT ĐA TẦNG TỐI ƯU")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CẶP SỐ CHỐT", value=f"{best_pair[0]:02d} — {best_pair[1]:02d}")
        with col2:
            st.metric(label="Chỉ Số Đan Xen Interlaced Score", value=f"{score:.4f}")
            
        st.info(explanation)
    else:
        st.warning(f"⚠️ Mới nhận diện được {len(all_numbers)} số ({total_kies}/5 kỳ). Vui lòng dán đủ 5 kỳ (100 số)!")
else:
    st.info("👆 Dán chuỗi số 5 kỳ vào khung phía trên để kích hoạt mô hình đa tầng.")
