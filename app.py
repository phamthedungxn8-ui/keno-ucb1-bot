import itertools
import random
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =============================================================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="AlphaVietlott Unified Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔮 ALPHAVIETLOTT UNIFIED QUANTUM ENGINE v3.0")
st.caption(
    "Nền tảng Tối ưu hóa Vietlott: Hỗn Độn (Chaos Space) + Di Truyền Đột Biến +"
    " Lý Thuyết Trận Thế + Tâm Thức Trực Giác"
)


# =============================================================================
# 2. BỘ THU THẬP & LỌC DỮ LIỆU TỰ ĐỘNG (REALTIME CÀO 0 ĐỒNG)
# =============================================================================
def fetch_live_keno(limit=100):
    """Tự động cào dữ liệu Keno mới nhất từ web công khai."""
    url = "https://minhchinh.com/live/keno.php"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            lines = res.text.split("\n")
            history = []
            for line in lines:
                nums = [int(s) for s in re.findall(r"\b\d+\b", line)]
                valid = [n for n in nums if 1 <= n <= 80]
                seen = set()
                uniq = [x for x in valid if not (x in seen or seen.add(x))]
                if len(uniq) >= 20:
                    history.append(uniq[:20])
            return history[:limit]
    except Exception:
        pass
    return None


def fetch_live_max3d(limit=60):
    """Tự động cào dữ liệu Max 3D mới nhất."""
    url = "https://minhchinh.com/ket-qua-vietlott-max3d.html"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            tokens = re.findall(r"\b\d{3}\b", res.text)
            if tokens:
                return tokens[:limit]
    except Exception:
        pass
    return None


def parse_file_upload(uploaded_file):
    """Xử lý file CSV/Excel người dùng nạp thủ công."""
    try:
        if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(
            ".txt"
        ):
            df = pd.read_csv(uploaded_file, header=None, on_bad_lines="skip")
        else:
            df = pd.read_excel(uploaded_file, header=None)

        draws = []
        for _, row in df.iterrows():
            row_str = " ".join([str(v) for v in row.values if pd.notna(v)])
            tokens = re.findall(r"\b\d+\b", row_str)
            valid = [int(t) for t in tokens if 1 <= int(t) <= 80]
            seen = set()
            uniq = [n for n in valid if not (n in seen or seen.add(n))]
            if len(uniq) >= 3:
                draws.append(uniq)
        return draws
    except Exception:
        return None


# =============================================================================
# 3. ENGINE TỐI ƯU HÓA KENO (CHAOS-GENETIC & GAME THEORY)
# =============================================================================
class KenoAdvancedEngine:

    def __init__(self, total_numbers=80):
        self.N = total_numbers
        self.weights = np.ones(self.N + 1)

    def process_chaos_and_game_theory(
        self, history_draws, intuition_numbers=None
    ):
        draw_count = len(history_draws)
        if draw_count == 0:
            return

        last_seen = {i: draw_count for i in range(1, 81)}
        freq = np.zeros(81)

        for t, draw in enumerate(history_draws):
            for num in draw:
                if 1 <= num <= 80:
                    freq[num] += 1
                    last_seen[num] = draw_count - t

        for i in range(1, 81):
            compression = np.log(last_seen[i] + 1)
            self.weights[i] = (freq[i] / draw_count) * 0.6 + (
                compression * 0.4
            )

        # Trận thế Đảo ngược Đám đông
        crowd_favorite = [1, 2, 3, 6, 8, 9, 10, 11, 22, 33, 66, 68, 79, 80]
        for num in range(1, 81):
            if num not in crowd_favorite:
                self.weights[num] *= 1.25

        # Bơm hạt giống Tâm thức
        if intuition_numbers:
            for num in intuition_numbers:
                if 1 <= num <= 80:
                    self.weights[num] *= 1.6

    def run_genetic_evolution(
        self, target_size=6, population_size=100, generations=35, mutation=0.2
    ):
        # Giải phóng seed để mỗi lần bấm nút là 1 lần tính toán mới hoàn toàn
        random.seed()
        np.random.seed()
        probs = self.weights[1:] / np.sum(self.weights[1:])
        population = []

        for _ in range(population_size):
            ticket = np.random.choice(
                range(1, 81), size=target_size, replace=False, p=probs
            )
            population.append(sorted(list(ticket)))

        def fitness(t):
            score = sum(self.weights[num] for num in t)
            spread = np.std(t)
            return score * (1 + spread * 0.04)

        for gen in range(generations):
            population.sort(key=lambda t: fitness(t), reverse=True)
            parents = population[: population_size // 4]
            new_pop = list(parents)

            while len(new_pop) < population_size:
                p1, p2 = random.sample(parents, 2)
                split = target_size // 2
                child = list(set(p1[:split] + p2[split:]))

                if random.random() < mutation or len(child) < target_size:
                    missing = target_size - len(child)
                    avail = [n for n in range(1, 81) if n not in child]
                    child.extend(random.sample(avail, missing))

                new_pop.append(sorted(child))
            population = new_pop

        population.sort(key=lambda t: fitness(t), reverse=True)
        unique_tickets = []
        seen = set()

        for t in population:
            t_tuple = tuple(t)
            if t_tuple not in seen:
                seen.add(t_tuple)
                # Chuyển kiểu dữ liệu np.int64 về int chuẩn Python
                clean_ticket = [int(x) for x in t]
                unique_tickets.append((clean_ticket, round(fitness(t), 2)))
            if len(unique_tickets) == 5:
                break

        return unique_tickets


# =============================================================================
# 4. ENGINE TỐI ƯU HÓA MAX 3D / MAX 3D+ (MARKOV POSITIONAL MATRIX)
# =============================================================================
class Max3DAdvancedEngine:

    def __init__(self):
        self.p1 = np.ones((10, 10)) * 0.1
        self.p2 = np.ones((10, 10)) * 0.1
        self.p3 = np.ones((10, 10)) * 0.1

    def fit_history(self, history_draws):
        if len(history_draws) < 2:
            return

        for t in range(len(history_draws) - 1):
            curr_code = f"{int(history_draws[t]):03d}"
            next_code = f"{int(history_draws[t+1]):03d}"

            c1, n1 = int(curr_code[0]), int(next_code[0])
            c2, n2 = int(curr_code[1]), int(next_code[1])
            c3, n3 = int(curr_code[2]), int(next_code[2])

            self.p1[c1][n1] += 1.0
            self.p2[c2][n2] += 1.0
            self.p3[c3][n3] += 1.0

        self.p1 /= self.p1.sum(axis=1, keepdims=True)
        self.p2 /= self.p2.sum(axis=1, keepdims=True)
        self.p3 /= self.p3.sum(axis=1, keepdims=True)

    def generate_combos(
        self, last_draw, intuition_seeds=None, is_plus=False, top_k=5
    ):
        code = f"{int(last_draw):03d}"
        c1, c2, c3 = int(code[0]), int(code[1]), int(code[2])

        prob1 = np.copy(self.p1[c1])
        prob2 = np.copy(self.p2[c2])
        prob3 = np.copy(self.p3[c3])

        if intuition_seeds:
            for seed in intuition_seeds:
                s_str = f"{int(seed):03d}"
                prob1[int(s_str[0])] *= 1.5
                prob2[int(s_str[1])] *= 1.5
                prob3[int(s_str[2])] *= 1.5

        scored = []
        for num in range(1000):
            s = f"{num:03d}"
            d1, d2, d3 = int(s[0]), int(s[1]), int(s[2])

            # Lọc cân bằng Gauss (Tổng 3 chữ số từ 7 đến 20)
            if not (7 <= d1 + d2 + d3 <= 20):
                continue

            score = prob1[d1] * prob2[d2] * prob3[d3]
            scored.append((s, round(score * 1000, 2)))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not is_plus:
            return scored[:top_k]
        else:
            top_singles = [x[0] for x in scored[:10]]
            pairs = list(itertools.combinations(top_singles, 2))
            pair_results = []
            for p1, p2 in pairs[:top_k]:
                score_p = next(x[1] for x in scored if x[0] == p1) + next(
                    x[1] for x in scored if x[0] == p2
                )
                pair_results.append((f"{p1} - {p2}", round(score_p, 2)))
            return pair_results


# =============================================================================
# 5. ĐIỀU HƯỚNG GIAO DIỆN STREAMLIT
# =============================================================================
st.sidebar.header("🕹️ TÙY CHỌN HẠNG MỤC")
game_type = st.sidebar.selectbox(
    "Chọn trò chơi phân tích:",
    ["VIETLOTT KENO (20/80)", "VIETLOTT MAX 3D / MAX 3D+"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Nguồn Dữ Liệu Lịch Sử")
data_mode = st.sidebar.radio(
    "Chế độ nạp:",
    ["📡 Cào Tự Động Realtime", "Dùng Dữ Liệu Mẫu Giả Lập", "Tải File CSV/Excel"],
)

history_data = []

if data_mode == "📡 Cào Tự Động Realtime":
    if st.sidebar.button("🔄 Cập Nhật Dữ Liệu Mới Nhất"):
        with st.spinner("Đang kết nối cào dữ liệu trực tiếp..."):
            if "KENO" in game_type:
                history_data = fetch_live_keno(100)
            else:
                history_data = fetch_live_max3d(60)

            if history_data:
                st.sidebar.success(
                    f"✅ Đã cào thành công {len(history_data)} kỳ quay!"
                )
            else:
                st.sidebar.warning(
                    "⚠️ Cổng cào dữ liệu bận. Hệ thống sẽ tự dùng dữ liệu"
                    " mẫu."
                )

elif data_mode == "Tải File CSV/Excel":
    file_up = st.sidebar.file_uploader(
        "Tải file dữ liệu:", type=["csv", "xlsx", "txt"]
    )
    if file_up:
        history_data = parse_file_upload(file_up)
        if history_data:
            st.sidebar.success(f"✅ Đã nạp {len(history_data)} kỳ từ file!")

# Tự động cấp dữ liệu mẫu nếu chưa có dữ liệu
if not history_data:
    
    if "KENO" in game_type:
        history_data = [
            list(np.random.choice(range(1, 81), size=20, replace=False))
            for _ in range(80)
        ]
    else:
        history_data = [f"{np.random.randint(0, 1000):03d}" for _ in range(50)]

# THÔNG TIN TÂM THỨC
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Tâm Thức & Trực Giác")
if "KENO" in game_type:
    intuition_in = st.sidebar.text_input(
        "Các số Keno trực giác (cách nhau bởi dấu phẩy):", value="37, 68, 15"
    )
    intuition_list = [
        int(x.strip()) for x in intuition_in.split(",") if x.strip().isdigit()
    ]
    target_k = st.sidebar.slider("Loại vé Keno (Bậc số):", 2, 10, 6)
    mutation_val = (
        st.sidebar.slider("Tỷ lệ Đột biến Di truyền (%):", 5, 50, 20) / 100.0
    )
else:
    max3d_mode = st.sidebar.radio(
        "Thể thức:", ["Max 3D (1 Bộ số)", "Max 3D+ (Cặp 2 bộ số)"]
    )
    intuition_in = st.sidebar.text_input(
        "Các bộ 3 số trực giác (VD: 378, 519):", value="378"
    )
    intuition_list = [
        x.strip() for x in intuition_in.split(",") if x.strip().isdigit()
    ]


# =============================================================================
# 6. KHU VỰC THỰC THI THUẬT TOÁN & HIỂN THỊ KẾT QUẢ
# =============================================================================
st.markdown("---")

if "KENO" in game_type:
    if st.button("🚀 KÍCH HOẠT THUẬT TOÁN KENO CHAOS-GENETIC", type="primary"):
        with st.spinner("🌀 Đang xử lý Không gian Dải Hút & Cho Vé Tiến Hóa..."):
            engine = KenoAdvancedEngine()
            engine.process_chaos_and_game_theory(
                history_data, intuition_numbers=intuition_list
            )
            results = engine.run_genetic_evolution(
                target_size=target_k,
                population_size=120,
                generations=40,
                mutation=mutation_val,
            )

        st.subheader(f"🔥 Top 5 Vé Keno Bậc {target_k} Tiến Hóa Tối Ưu")
        for idx, (ticket, score) in enumerate(results, 1):
            st.markdown(
                f"""
            <div style="background-color: #161b22; padding: 18px; border-radius: 12px; margin-bottom: 12px; border-left: 6px solid #00dfd8;">
                <h4 style="margin:0; color: #00dfd8;">🧬 Bộ Vé Tiến Hóa #{idx} — (Fitness Score: {score})</h4>
                <h2 style="margin:8px 0; color: #ffffff; letter-spacing: 3px;">{ticket}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

else:
    if st.button("🚀 KÍCH HOẠT THUẬT TOÁN MAX 3D MARKOV VECTOR", type="primary"):
        with st.spinner("🌀 Đang tính toán Ma trận Markov Vị trí..."):
            engine = Max3DAdvancedEngine()
            engine.fit_history(history_data)

            last_draw = history_data[0]
            is_plus = True if "Max 3D+" in max3d_mode else False

            results = engine.generate_combos(
                last_draw,
                intuition_seeds=intuition_list,
                is_plus=is_plus,
                top_k=5,
            )

        st.info(f"📌 Kỳ quay gần nhất ($T$): **[{last_draw}]**")
        st.subheader(f"🔥 Top 5 Bộ Số {max3d_mode} Được Khuyên Nghị")

        for idx, (code, score) in enumerate(results, 1):
            st.markdown(
                f"""
            <div style="background-color: #161b22; padding: 18px; border-radius: 12px; margin-bottom: 12px; border-left: 6px solid #ff0055;">
                <h4 style="margin:0; color: #ff0055;">🎯 Bộ Vé #{idx} — Điểm Năng Lượng: {score}</h4>
                <h1 style="margin:5px 0; color: #ffffff; letter-spacing: 4px;">{code}</h1>
            </div>
            """,
                unsafe_allow_html=True,
    )
