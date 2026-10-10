import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDM-IDS v25.0 Sub-Network Interconnection Engine", layout="wide")

# ==============================================================================
# LỚP 0: SUB-NETWORK TOPOLOGY PIPELINE (CHIA MẢNG & XÁC LẬP LIÊN KẾT)
# ==============================================================================
class SubNetworkTopologyPipeline:
    """Phân rã không gian 80 số thành 8 cụm mảng nhỏ và xây dựng ma trận liên kết."""
    def __init__(self, n_numbers=80, cluster_size=10):
        self.D = n_numbers
        self.cluster_size = cluster_size
        self.num_clusters = n_numbers // cluster_size # 8 cụm

    def build_clusters(self, raw_text: str, min_kies=8, max_kies=25):
        cleaned_data = re.sub(r'(?:Kì|Kỳ)\s*\d+[:\s]*', '\n', raw_text.strip(), flags=re.IGNORECASE)
        all_numbers = [int(n) for n in re.findall(r'\b\d{1,2}\b', cleaned_data) if 1 <= int(n) <= self.D]
        total_kies = len(all_numbers) // 20

        if total_kies < min_kies:
            return None, total_kies

        kies_to_use = min(total_kies, max_kies)
        used_numbers = all_numbers[-kies_to_use * 20:]

        matrix = np.zeros((kies_to_use, self.D), dtype=float)
        for k in range(kies_to_use):
            for num in used_numbers[k * 20 : (k + 1) * 20]:
                matrix[k, num - 1] = 1.0

        return matrix, kies_to_use

    def compute_inter_cluster_matrix(self, X):
        T, D = X.shape
        nc = self.num_clusters
        cs = self.cluster_size

        # Chuyển ma trận T x 80 thành T x 8 (Ma trận năng lượng cụm)
        cluster_energy = np.zeros((T, nc))
        for c in range(nc):
            cluster_energy[:, c] = np.sum(X[:, c*cs : (c+1)*cs], axis=1)

        # Tính ma trận liên kết chuyển đổi (Transition Coupling Matrix 8x8)
        coupling_matrix = np.zeros((nc, nc))
        for t in range(T - 1):
            curr_state = cluster_energy[t, :]
            next_state = cluster_energy[t + 1, :]
            coupling_matrix += np.outer(curr_state, next_state)

        # Chuẩn hóa ma trận liên kết
        row_sums = coupling_matrix.sum(axis=1, keepdims=True) + 1e-9
        coupling_matrix = coupling_matrix / row_sums

        return cluster_energy, coupling_matrix


# ==============================================================================
# LỚP 1: LOCAL SUB-NETWORK RESONANCE & ROUTING
# ==============================================================================
class SubNetworkResonanceEngine:
    """Phân tích dòng chảy năng lượng nội tại và liên mảng để chọn cụm tối ưu."""
    def process(self, X, cluster_energy, coupling_matrix):
        T, D = X.shape
        nc = cluster_energy.shape[1]
        cs = D // nc

        # Dự báo năng lượng cụm ở kỳ tiếp theo dựa trên trạng thái kỳ hiện tại và ma trận liên kết
        latest_state = cluster_energy[-1, :]
        predicted_flux = np.dot(latest_state, coupling_matrix)

        # Xác định cụm mảng có dòng chảy thông tin mạnh nhất (Active Cluster)
        active_cluster_idx = int(np.argmax(predicted_flux))

        # Trích xuất các con số bên trong cụm mảng đang hoạt động mạnh
        cluster_start = active_cluster_idx * cs
        cluster_end = (active_cluster_idx + 1) * cs
        cluster_numbers = list(range(cluster_start + 1, cluster_end + 1))

        # Tính điểm xung lực chi tiết cho từng số bên trong cụm đó kết hợp toàn cục
        global_freqs = X.sum(axis=0)
        local_scores = np.zeros(D)

        for num in cluster_numbers:
            idx = num - 1
            # Ưu tiên số có tần suất vừa phải trong cụm (1-2 lần trong chuỗi gần)
            recent_freq = X[-min(6, T):, idx].sum()
            if recent_freq in [1, 2]:
                local_scores[idx] = 2.5 * (1.0 / (global_freqs[idx] + 1.0))
            else:
                local_scores[idx] = 1.0 * (1.0 / (global_freqs[idx] + 1.0))

        # Sắp xếp các số trong cụm chủ lực và phần mở rộng liên kết
        ranked_indices = np.argsort(local_scores)[::-1]
        ranked_numbers = [idx + 1 for idx in ranked_indices if idx + 1 in cluster_numbers]

        # Nếu cụm không đủ số, bổ sung từ các cụm có liên kết mạnh nhì
        if len(ranked_numbers) < 8:
            for idx in ranked_indices:
                num = idx + 1
                if num not in ranked_numbers:
                    ranked_numbers.append(num)

        return active_cluster_idx + 1, cluster_numbers, ranked_numbers, local_scores


# ==============================================================================
# LỚP 2: COMBINATORIAL ROUTER CHO BẬC 2 & BẬC 3
# ==============================================================================
class SubNetworkCombinatorialRouter:
    """Định tuyến các con số từ cụm liên kết thành dàn Bậc 2 và Bậc 3 tối ưu."""
    def route(self, X, ranked_numbers, local_scores):
        D = X.shape[1]
        final_scores = local_scores.copy()

        # Triệt tiêu lặp số kỳ T-1
        for i in range(D):
            if X[-1, i] == 1:
                final_scores[i] *= 0.20

        # Tái sắp xếp theo điểm số thực tế sau khi lọc lặp
        final_ranked_indices = np.argsort(final_scores)[::-1]
        final_ranked_numbers = [idx + 1 for idx in final_ranked_indices]

        # Phân bổ Dual-Core Alpha / Beta từ cụm liên kết
        alpha_core = [final_ranked_numbers[i] for i in range(0, 12, 2)]
        beta_core = [final_ranked_numbers[i] for i in range(1, 12, 2)]

        bac2_alpha = sorted(alpha_core[:2])
        bac2_beta = sorted(beta_core[:2])

        bac3_alpha = sorted(alpha_core[:3])
        bac3_beta = sorted(beta_core[:3])

        return {
            "bac2_alpha": bac2_alpha,
            "bac2_beta": bac2_beta,
            "bac3_alpha": bac3_alpha,
            "bac3_beta": bac3_beta,
            "ranked_all": final_ranked_numbers,
            "scores": final_scores
        }


# ==============================================================================
# UI STREAMLIT (v25.0)
# ==============================================================================
st.title("🌐 MDM-IDS v25.0: SUB-NETWORK INTERCONNECTION ENGINE")
st.caption("Kiến Trúc Phân Rã Mảng Nhỏ • Ma Trận Liên Kết Liên Mảng (Coupling Flux) • Tối Ưu Bậc 2 & Bậc 3")

with st.sidebar:
    st.header("⚙️ Cấu Hình Mảng & Cửa Sổ")
    min_window = st.slider("Cửa sổ tối thiểu (Kỳ):", 6, 15, 8)
    max_window = st.slider("Cửa sổ tối đa (Kỳ):", 15, 30, 20)

raw_input = st.text_area(
    "Dán dữ liệu cuốn chiếu Keno (8 - 25 kỳ):",
    placeholder="Kỳ 1: 01 02 05 08 ...\nKỳ 2: ...",
    height=180
)

if raw_input.strip():
    pipeline = SubNetworkTopologyPipeline()
    matrix, total_kies = pipeline.build_clusters(raw_input, min_kies=min_window, max_kies=max_window)

    if matrix is not None:
        # Lớp 0: Xây dựng ma trận liên kết cụm
        cluster_energy, coupling_matrix = pipeline.compute_inter_cluster_matrix(matrix)

        # Lớp 1: Cảm ứng cộng hưởng liên mảng
        resonance_engine = SubNetworkResonanceEngine()
        active_cluster, cluster_nums, ranked_nums, local_scores = resonance_engine.process(matrix, cluster_energy, coupling_matrix)

        # Lớp 2: Định tuyến Bậc 2 & Bậc 3
        router = SubNetworkCombinatorialRouter()
        res = router.route(matrix, ranked_nums, local_scores)

        st.success(f"⚡ Đã phân rã 80 số thành 8 mảng nhỏ, phát hiện Dòng chảy Cụm chủ lực số #{active_cluster} trên {total_kies} kỳ!")
        st.markdown("---")

        st.info(f"📍 **Cụm mảng đang có dòng chảy thông tin tương tác mạnh nhất:** `Cụm #{active_cluster}` (Gồm các số: {', '.join([str(n) for n in cluster_nums])})")

        # HIỂN THỊ BẬC 2 & BẬC 3
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔥 MỤC TIÊU BẬC 2 (CHỌN 2 - ĂN 90.000 VNĐ)")
            st.markdown(f"<div style='text-align:center; padding:12px; background:#0A192F; border-radius:8px; border:2px solid #00F0FF;'>"
                        f"<span style='color:#00F0FF; font-weight:bold;'>BẬC 2 - ALPHA (TÍCH LŨY)</span>"
                        f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_alpha'][0]:02d} — {res['bac2_alpha'][1]:02d}</h2></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; padding:12px; background:#1A0903; border-radius:8px; border:2px solid #FF5500;'>"
                        f"<span style='color:#FF5500; font-weight:bold;'>BẬC 2 - BETA (BÙNG NỔ)</span>"
                        f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_beta'][0]:02d} — {res['bac2_beta'][1]:02d}</h2></div>", unsafe_allow_html=True)

        with col_b:
            st.subheader("⚡ MỤC TIÊU BẬC 3 (CHỌN 3 - HOÀN VỐN / THƯỞNG)")
            b3_a_str = " - ".join([f"{n:02d}" for n in res["bac3_alpha"]])
            b3_b_str = " - ".join([f"{n:02d}" for n in res["bac3_beta"]])
            st.info(f"**BẬC 3 - ALPHA:**\n### **{b3_a_str}**")
            st.warning(f"**BẬC 3 - BETA (BỘC PHÁT):**\n### **{b3_b_str}**")

        st.markdown("---")
        st.subheader("📊 BẢNG XẾP HẠNG XUNG LỰC NỘI TẠI CỤM MẢNG")
        df_top = pd.DataFrame({
            "Thứ hạng": [f"Top {i+1}" for i in range(10)],
            "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(10)],
            "Điểm Tương Tác Vảng": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(10)]
        })
        st.table(df_top.T)

    else:
        st.warning(f"Cần tối thiểu {min_window} kỳ dữ liệu để phân tích mảng liên kết (Hiện có {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu Keno vào khung trên để khởi chạy mô hình Sub-Network v25.0.")ETA (BÙNG NỔ)</span>"
                            f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_beta'][0]:02d} — {res['bac2_beta'][1]:02d}</h2>"
                            f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Dàn Phụ Bộc Phát</p></div>", unsafe_allow_html=True)
            with c2c:
                st.markdown(f"<div style='text-align:center; padding:15px; background:#111; border-radius:10px; border:1px solid #555;'>"
                            f"<span style='color:#AAA; font-weight:bold;'>BẬC 2 - DỰ PHÒNG</span>"
                            f"<h2 style='color:#FFF; margin:5px 0;'>{res['bac2_backup'][0]:02d} — {res['bac2_backup'][1]:02d}</h2>"
                            f"<p style='color:#AAA; margin:0; font-size:0.8rem;'>Bọc Lót Pha 3</p></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("⚡ BẬC 3 VÀ BẬC 4 (BẢO HIỂM HOÀN VỐN TRÚNG 2/3 VÀ 3/4)")
            col3, col4 = st.columns(2)
            with col3:
                st.info(f"**BẬC 3 ALPHA:** {' - '.join([f'{n:02d}' for n in res['bac3_alpha']])}\n\n"
                        f"**BẬC 3 BETA:** {' - '.join([f'{n:02d}' for n in res['bac3_beta']])}")
            with col4:
                st.warning(f"**BẬC 4 ALPHA:** {' - '.join([f'{n:02d}' for n in res['bac4_alpha']])}\n\n"
                           f"**BẬC 4 BETA:** {' - '.join([f'{n:02d}' for n in res['bac4_beta']])}")

        with tab_large:
            st.subheader("🎯 DÀN CHỦ LỰC BẬC 8 (BẢO HIỂM HOÀN TIỀN TRÚNG 0 & TRÚNG 4/8)")
            b8_str = "  •  ".join([f"**{n:02d}**" for n in res["bac8_master"]])
            st.markdown(
                f"<div style='text-align: center; padding: 20px; background-color: #0D1117; border-radius: 12px; border: 2px solid #7928CA; box-shadow: 0 0 20px rgba(121, 40, 202, 0.4);'>"
                f"<h2 style='color: #FF0080; margin:0;'>{b8_str}</h2>"
                f"<p style='color: #AAA; margin:8px 0 0 0;'>Tối ưu dải thưởng: Trúng 0, 4, 5, 6, 7, 8</p>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                st.subheader("🔥 DÀN BẬC 7")
                st.info(f"### **{' - '.join([f'{n:02d}' for n in res['bac7_master']])}**")
            with col_l2:
                st.subheader("🌟 DÀN BẬC 9")
                st.success(f"### **{' - '.join([f'{n:02d}' for n in res['bac9_master']])}**")

            st.markdown("---")
            st.subheader("🛡️ DÀN BỌC LÓT BẬC 8 (BACKUP SET)")
            st.warning(f"### **{' - '.join([f'{n:02d}' for n in res['bac8_backup']])}**")

        with tab_analytics:
            st.subheader("📊 BẢNG CHI TIẾT ĐIỂM XUNG LỰC KALMAN V21.0")
            df_top = pd.DataFrame({
                "Thứ hạng": [f"Top {i+1}" for i in range(20)],
                "Con số": [f"Số {res['ranked_all'][i]:02d}" for i in range(20)],
                "Cấu trúc Core": ["Alpha Core" if i % 2 == 0 else "Beta Core (Bộc phát)" for i in range(20)],
                "Điểm Tín Hiệu v21.0": [f"{res['scores'][res['ranked_all'][i]-1]:.6f}" for i in range(20)]
            })
            st.table(df_top)

    else:
        st.warning(f"Cần tối thiểu {min_window} kỳ dữ liệu để khởi chạy mô hình (Hiện có {total_kies} kỳ).")
else:
    st.info("Dán dữ liệu cuốn chiếu 10-30 kỳ Keno vào khung trên để thực thi mô hình v21.0.")
