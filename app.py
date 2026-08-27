import itertools
import random
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="AlphaKeno Chaos-Genetic Engine", page_icon="🌌", layout="wide"
)

st.title("🌌 ALPHAKENO CHAOS-GENETIC QUANTUM ENGINE")
st.caption(
    "Mô hình Hợp nhất: Dải Hút Kỳ Dị (Chaos) + Lý Thuyết Trận Thế + Thuật Toán Di Truyền Biến Dị"
)


# =============================================================================
# 1. BỘ THU THẬP & SỬA LỖI DỮ LIỆU
# =============================================================================
def parse_draws_from_dataframe(df):
    history_draws = []
    for _, row in df.iterrows():
        row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
        tokens = re.findall(r"\b\d+\b", row_str)
        valid_nums = [int(t) for t in tokens if 1 <= int(t) <= 80]
        seen = set()
        unique_nums = [n for n in valid_nums if not (n in seen or seen.add(n))]
        if len(unique_nums) >= 10:
            history_draws.append(unique_nums[:20])
    return history_draws


# =============================================================================
# 2. CORE CHAOS-GENETIC & GAME THEORY ENGINE
# =============================================================================
class ChaosGeneticEngine:

    def __init__(self, total_numbers=80):
        self.N = total_numbers
        self.weights = np.ones(self.N + 1)  # Trọng số năng lượng 80 số
        self.gravity_matrix = np.zeros(
            (self.N + 1, self.N + 1)
        )  # Trường hấp dẫn

    def compute_chaos_attractors(self, history_draws):
        """1. VẬT LÝ & HỖN ĐỘN: Tính dải hút kỳ dị và lực nén không gian."""
        draw_count = len(history_draws)
        if draw_count == 0:
            return

        # Tính tần suất và khoảng cách kỳ gần nhất (Lực nén)
        last_seen = {i: draw_count for i in range(1, 81)}
        freq = np.zeros(81)

        for t, draw in enumerate(history_draws):
            for num in draw:
                if 1 <= num <= 80:
                    freq[num] += 1
                    last_seen[num] = draw_count - t

        # Tính điểm hấp dẫn dựa trên Sự cân bằng Hỗn độn (Chaos Attractor Score)
        for i in range(1, 81):
            # Kết hợp tần suất + độ nén dồn (chu kỳ tích tụ năng lượng)
            compression_force = np.log(last_seen[i] + 1)
            self.weights[i] = (freq[i] / draw_count) * 0.6 + (
                compression_force * 0.4
            )

    def apply_game_theory_anti_crowd(self):
        """2. LÝ THUYẾT TRẬN THẾ: Đảo ngược điểm mù đám đông.

        Đám đông hay chọn các số đẹp (1-10), số lặp (11,22..), phong thủy.
        Thuật toán bơm lực đẩy cho các vùng bị đám đông bỏ quên.
        """
        crowd_favorite = [
            1,
            2,
            3,
            6,
            8,
            9,
            10,
            11,
            22,
            33,
            66,
            68,
            79,
            80,
        ]  # Các số tâm lý
        for num in range(1, 81):
            if num not in crowd_favorite:
                self.weights[num] *= 1.25  # Bơm trọng số cho điểm mù

    def apply_intuition(self, intuition_numbers, bias=1.5):
        """3. TRIẾT HỌC TÂM THỨC: Bơm hạt giống trực giác."""
        for num in intuition_numbers:
            if 1 <= num <= 80:
                self.weights[num] *= bias

    def genetic_evolution(self, target_size=8, population_size=100, generations=30, mutation_rate=0.2):
        """4. THUẬT TOÁN DI TRUYỀN & ĐỘT BIẾN: Cho các bộ vé tự tiến hóa."""
        # Khởi tạo quần thể vé ngẫu nhiên dựa trên trọng số năng lượng
        probabilities = self.weights[1:] / np.sum(self.weights[1:])

        population = []
        for _ in range(population_size):
            ticket = np.random.choice(
                range(1, 81), size=target_size, replace=False, p=probabilities
            )
            population.append(sorted(list(ticket)))

        # Hàm đánh giá độ thích nghi (Fitness Function)
        def fitness(ticket):
            score = sum(self.weights[num] for num in ticket)
            # Thưởng điểm cho sự phân bổ đều trong không gian 80 số
            spread = np.std(ticket)
            return score * (1 + spread * 0.05)

        # Vòng lặp Tiến hóa
        for gen in range(generations):
            population.sort(key=lambda t: fitness(t), reverse=True)
            parents = population[: population_size // 4]  # Chọn 25% xuất sắc nhất

            new_population = list(parents)
            while len(new_population) < population_size:
                # Lai ghép (Crossover)
                p1, p2 = random.sample(parents, 2)
                split = target_size // 2
                child = list(set(p1[:split] + p2[split:]))

                # Đột biến forced mutation
                if random.random() < mutation_rate or len(child) < target_size:
                    missing = target_size - len(child)
                    available = [n for n in range(1, 81) if n not in child]
                    mutated_nodes = random.sample(available, missing)
                    child.extend(mutated_nodes)

                new_population.append(sorted(child))

            population = new_population

        # Đánh giá cuối cùng và lọc trùng
        population.sort(key=lambda t: fitness(t), reverse=True)
        unique_tickets = []
        seen = set()

        for t in population:
            t_tuple = tuple(t)
            if t_tuple not in seen:
                seen.add(t_tuple)
                unique_tickets.append((t, round(fitness(t), 2)))
            if len(unique_tickets) == 5:
                break

        return unique_tickets


# =============================================================================
# 3. GIAO DIỆN STREAMLIT
# =============================================================================
st.sidebar.header("⚙️ Cấu Hình Thuật Toán")
data_source = st.sidebar.radio(
    "Nguồn dữ liệu:", ["📡 Cào Tự Động Web", "Dùng Dữ Liệu Giả Lập Mẫu", "Tải File CSV"]
)

history_draws = []

if data_source == "📡 Cào Tự Động Web":
    if st.sidebar.button("🔄 Lấy Keno Mới Nhất"):
        try:
            url = "https://minhchinh.com/live/keno.php"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if res.status_code == 200:
                lines = res.text.split("\n")
                for line in lines:
                    nums = [int(s) for s in re.findall(r"\b\d+\b", line)]
                    valid = [n for n in nums if 1 <= n <= 80]
                    seen = set()
                    uniq = [x for x in valid if not (x in seen or seen.add(x))]
                    if len(uniq) >= 20:
                        history_draws.append(uniq[:20])
                history_draws = history_draws[:100]
                st.sidebar.success(f"✅ Đã cào {len(history_draws)} kỳ quay!")
        except Exception as e:
            st.sidebar.error("⚠️ Không cào được dữ liệu. Hãy chọn nguồn giả lập.")

if not history_draws and data_source != "Tải File CSV":
    np.random.seed(42)
    history_draws = [
        list(np.random.choice(range(1, 81), size=20, replace=False))
        for _ in range(80)
    ]
    st.sidebar.info("💡 Đang dùng 80 kỳ quay mẫu giả lập.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Tâm Thức & Đột Biến")
intuition_input = st.sidebar.text_input("Số trực giác:", value="37, 68, 15")
intuition_numbers = [
    int(x.strip()) for x in intuition_input.split(",") if x.strip().isdigit()
]

mutation_rate = (
    st.sidebar.slider("Tỷ lệ Đột biến Di truyền (%):", 5, 50, 20) / 100.0
)
target_k = st.sidebar.slider("Loại vé (Bậc số):", 2, 10, 6)

# THỰC THI
if st.button("🚀 KÍCH HOẠT MÔ HÌNH HỢP NHẤT HỖN ĐỘN - DI TRUYỀN", type="primary"):
    with st.spinner("🌀 Đang xử lý Không gian Dải Hút & Cho Vé Tiến Hóa..."):
        engine = ChaosGeneticEngine()
        engine.compute_chaos_attractors(history_draws)
        engine.apply_game_theory_anti_crowd()

        if intuition_numbers:
            engine.apply_intuition(intuition_numbers, bias=1.6)

        best_tickets = engine.genetic_evolution(
            target_size=target_k,
            population_size=120,
            generations=40,
            mutation_rate=mutation_rate,
        )

    st.subheader(f"🔥 Top 5 Vé Keno Bậc {target_k} Tiến Hóa Tối Ưu")
    for idx, (ticket, score) in enumerate(best_tickets, 1):
        st.markdown(
            f"""
        <div style="background-color: #161b22; padding: 18px; border-radius: 12px; margin-bottom: 12px; border-left: 6px solid #7928ca;">
            <h4 style="margin:0; color: #00dfd8;">🧬 Bộ Vé Tiến Hóa #{idx} — (Fitness Score: {score})</h4>
            <h2 style="margin:8px 0; color: #ffffff; letter-spacing: 3px;">{ticket}</h2>
        </div>
        """,
            unsafe_allow_html=True,
)
