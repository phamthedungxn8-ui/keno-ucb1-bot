import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIG & PAGE TITLE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KENO QUANT TIMING ENGINE v6.2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ KENO QUANT TIMING ENGINE v6.2")
st.caption(
    "Bộ Lọc Định Lượng Bất Điểm Nổ Tức Thì & Quản Trị Khung Cửa Số 3 Kỳ (Live Engine)"
)


# -----------------------------------------------------------------------------
# CORE QUANT LOGIC (CACHED & AUTO-HANDLING MISSING DATA)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def process_philosophical_quant(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1. Làm sạch tên cột
    df.columns = df.columns.str.strip().str.lower()

    # 2. Chuẩn hóa cột pair (thêm số 0 vào trước các số có 1 chữ số: 4 -> 04)
    if "pair" in df.columns:

        def format_pair(val):
            val_str = str(val).strip()
            # Tách các số bởi dấu phẩy hoặc gạch ngang
            parts = [
                p.strip().zfill(2)
                for p in val_str.replace("-", ",").split(",")
                if p.strip()
            ]
            return ",".join(parts)

        df["pair"] = df["pair"].apply(format_pair)

    # 3. Chuẩn hóa cột zone (xóa khoảng trắng thừa)
    if "zone" in df.columns:
        df["zone"] = df["zone"].astype(str).str.strip()
    else:
        df["zone"] = "Bậc 2"

    # 4. Chuyển đổi và xử lý các cột số (Bù ô trống NaN)
    numeric_cols = ["hits", "a_gap", "c_gap", "max_gap", "std_gap"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Xử lý lấp đầy dữ liệu nếu thiếu/bỏ trống trên Google Sheets
    if "c_gap" in df.columns:
        df["c_gap"] = df["c_gap"].fillna(0)
    if "max_gap" in df.columns:
        df["max_gap"] = df["max_gap"].fillna(15)
    if "std_gap" in df.columns:
        df["std_gap"] = df["std_gap"].fillna(2.0)

    df = df.fillna(0)

    # 5. Các công thức Định lượng
    # Energy Index (Tỷ lệ nén năng lượng)
    df["energy_index"] = np.where(
        df["a_gap"] > 0, np.round(df["c_gap"] / df["a_gap"], 2), 0
    )

    # Z-Score Timing
    df["z_score"] = np.where(
        df["std_gap"] > 0,
        np.round((df["c_gap"] - df["a_gap"]) / df["std_gap"], 2),
        0,
    )

    # Quant Artistry Score
    df["quant_artistry_score"] = np.round(
        (df["energy_index"] * 10)
        + (df["z_score"] * 5)
        + (df["hits"] * 0.5)
        - (df["c_gap"] * 0.2),
        1,
    )

    # Status Categorization
    def assign_status(row):
        z = row["z_score"]
        if z >= 1.8:
            return "🔴 Bùng Nổ Tức Thời (Khung 1-3 Kỳ)"
        elif z >= 1.0:
            return "🟡 Vùng Cảnh Báo (Sắp Bùng Nổ)"
        elif z >= 0:
            return "⚪ An Toàn / Chu Kỳ Mới"
        else:
            return "🔵 Đang Tích Lũy (Bỏ qua - Chưa nên nuôi)"

    df["timing_status"] = df.apply(assign_status, axis=1)

    # Kích thước bong bóng biểu đồ
    df["bubble_size"] = df["quant_artistry_score"].apply(
        lambda x: max(float(x), 0.1) * 2 + 5
    )

    return df.sort_values(by="quant_artistry_score", ascending=False)


# -----------------------------------------------------------------------------
# SIDEBAR - LIVE DATA INPUT (GOOGLE SHEETS)
# -----------------------------------------------------------------------------
st.sidebar.header("🔄 Dữ Liệu Live từ Google Sheets")

sheet_url = st.sidebar.text_input(
    "Nhập link Google Sheets:",
    value="",
    placeholder="https://docs.google.com/spreadsheets/d/...",
    help="Đảm bảo Google Sheets đã bật chế độ 'Ai có link cũng xem được'",
)

if st.sidebar.button("⚡ Cập nhật dữ liệu mới"):
    st.cache_data.clear()
    st.rerun()

raw_df = None


# Tự động tải dữ liệu mới từ Google Sheets mỗi 30 giây
@st.cache_data(ttl=30, show_spinner=False)
def load_data_from_gsheets(url):
    try:
        if "/edit" in url:
            csv_url = url.split("/edit")[0] + "/export?format=csv"
            if "gid=" in url:
                gid = url.split("gid=")[1].split("&")[0]
                csv_url += f"&gid={gid}"
        else:
            csv_url = url
        return pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"Chưa thể tải dữ liệu từ Google Sheets: {e}")
        return None


if sheet_url and "docs.google.com" in sheet_url:
    raw_df = load_data_from_gsheets(sheet_url)
else:
    st.sidebar.info(
        "💡 Dán đường link Google Sheets ở trên để bắt đầu phân tích Live."
    )


# -----------------------------------------------------------------------------
# MAIN APP FLOW
# -----------------------------------------------------------------------------
if raw_df is not None:
    df_processed = process_philosophical_quant(raw_df)

    # REAL-TIME TRIGGER INPUT
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Real-Time Trigger (Kỳ Vừa Ra)")
    last_draw_input = st.sidebar.text_input(
        "Nhập các số kỳ vừa ra (cách nhau dấu phẩy):",
        value="",
        placeholder="VD: 04, 15, 37, 52...",
    )

    # BỘ LỌC PHÂN VÙNG BẬC
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Lọc Bổ Sung")
    available_zones = sorted(df_processed["zone"].unique().tolist())
    selected_zones = st.sidebar.multiselect(
        "Lọc Phân Vùng:", options=available_zones, default=available_zones
    )

    # Filter dataframe theo zone chọn
    filtered_df = df_processed[df_processed["zone"].isin(selected_zones)]

    # Real-time trigger check
    triggered_pairs = []
    if last_draw_input.strip():
        drawn_numbers = [
            n.strip().zfill(2)
            for n in last_draw_input.replace(";", ",").split(",")
            if n.strip()
        ]
        st.sidebar.success(f"Đã ghi nhận {len(drawn_numbers)} số kỳ vừa ra.")

        for idx, row in filtered_df.iterrows():
            p_str = str(row["pair"])
            p_nums = [x.strip().zfill(2) for x in p_str.split(",") if x.strip()]
            if any(num in drawn_numbers for num in p_nums):
                triggered_pairs.append(row["pair"])

    # DASHBOARD TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📍 Point-of-Impact",
            "🚀 Real-Time Trigger",
            "🛡️ Quy Tắc 3 Kỳ",
            "📊 Phân Bổ Vốn Kelly",
            "📋 Full Data Matrix",
        ]
    )

    with tab1:
        st.subheader("🎯 Ma Trận Timing Z-Score & Điểm Bùng Nổ")
        if not filtered_df.empty:
            fig_z = px.scatter(
                filtered_df,
                x="energy_index",
                y="z_score",
                color="timing_status",
                size="bubble_size",
                hover_name="pair",
                text="pair",
                title="Bản Đồ Điểm Bùng Nổ Tức Thời",
                labels={
                    "energy_index": "Tỷ lệ nén (c_gap/a_gap)",
                    "z_score": "Z-Score Timing",
                },
            )
            fig_z.add_hline(
                y=1.8,
                line_dash="dash",
                line_color="red",
                annotation_text="Vùng Bùng Nổ (1.8)",
            )
            fig_z.add_vline(
                x=1.0,
                line_dash="dash",
                line_color="orange",
                annotation_text="Nén Cao (1.0)",
            )
            fig_z.update_traces(textposition="top center")
            st.plotly_chart(fig_z, use_container_width=True)
        else:
            st.warning("Không có dữ liệu phù hợp với bộ lọc đã chọn!")

    with tab2:
        st.subheader("🔗 Lọc Số Mồi Từ Kỳ Vừa Quay")
        if triggered_pairs:
            st.success(
                f"🔥 Tìm thấy {len(triggered_pairs)} cặp số hội tụ Tín Hiệu Kích Hoạt từ kỳ vừa ra!"
            )
            trig_df = filtered_df[filtered_df["pair"].isin(triggered_pairs)]
            st.dataframe(
                trig_df[
                    [
                        "pair",
                        "quant_artistry_score",
                        "z_score",
                        "timing_status",
                        "c_gap",
                        "a_gap",
                        "zone",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.info(
                "Nhập danh sách các số vừa về ở thanh menu bên trái để kích hoạt bộ lọc."
            )

    with tab3:
        st.subheader("🛡️ Khung Kỷ Luật 3 Kỳ & Cắt Lỗ Tự Động")
        st.warning(
            "⚠️ QUY TẮC BẮT BUỘC: Chỉ giao dịch tối đa 3 kỳ cho 1 tín hiệu. Kỳ thứ 3 không nổ ➔ CẮT LỖ NGAY!"
        )
        high_z_df = filtered_df[filtered_df["z_score"] >= 1.0]
        if not high_z_df.empty:
            st.dataframe(
                high_z_df[
                    [
                        "pair",
                        "z_score",
                        "timing_status",
                        "c_gap",
                        "max_gap",
                        "zone",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.info(
                "Hiện chưa có cặp số nào chạm ngưỡng Z-Score căng cứng (>= 1.0)."
            )

    with tab4:
        st.subheader("🎲 Phân Bổ Vốn Tối Ưu Tốc Độ (Kelly Standard)")
        capital = st.number_input(
            "Tổng vốn dành cho khung 3 kỳ (VNĐ):",
            value=1000000,
            step=100000,
        )
        top_pairs = filtered_df.head(5).copy()
        if not top_pairs.empty:
            total_score = (
                top_pairs["quant_artistry_score"].clip(lower=0.1).sum()
            )
            top_pairs["kelly_ratio"] = np.round(
                top_pairs["quant_artistry_score"].clip(lower=0.1) / total_score,
                2,
            )
            top_pairs["allocated_cash"] = (
                top_pairs["kelly_ratio"] * capital
            ).astype(int)
            st.dataframe(
                top_pairs[
                    [
                        "pair",
                        "quant_artistry_score",
                        "z_score",
                        "kelly_ratio",
                        "allocated_cash",
                        "zone",
                    ]
                ],
                use_container_width=True,
            )

    with tab5:
        st.subheader("📋 Ma Trận Định Lượng Toàn Phần")
        st.dataframe(
            filtered_df[
                [
                    "pair",
                    "zone",
                    "quant_artistry_score",
                    "timing_status",
                    "z_score",
                    "c_gap",
                    "a_gap",
                    "max_gap",
                    "hits",
                ]
            ],
            use_container_width=True,
        )

else:
    st.info(
        "👈 Vui lòng nhập đường link Google Sheets vào thanh menu bên trái để tải dữ liệu."
                        )
