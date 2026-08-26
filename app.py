import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


def scrape_free_keno_history(limit_draws=100):
    """Cào dữ liệu Keno 100% Miễn phí từ web hiển thị công khai (Ví dụ từ

    xoso.com.vn hoặc minhchinh)
    """
    url = "https://xoso.com.vn/ket-qua-xs-keno.html"

    # Giả lập trình duyệt người dùng để không bị chặn (Bypass Cloudflare/Bot detection)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Tìm tất cả các khối chứa kết quả kỳ quay
            history_draws = []

            # Giả định phân tích cấu trúc DOM trang web kết quả
            # (Hệ thống sẽ bóc tách các danh sách có 20 số Keno)
            draw_rows = soup.find_all("div", class_="keno-draw-item")

            for row in draw_rows:
                numbers = [
                    int(span.text)
                    for span in row.find_all("span", class_="number")
                ]
                if len(numbers) == 20:
                    history_draws.append(numbers)

            # Trường hợp cấu trúc web thay đổi, dùng Regex quét toán bộ số
            if not history_draws:
                # Quét theo thẻ text chuẩn
                lines = soup.get_text().split("\n")
                for line in lines:
                    tokens = [int(s) for s in line.split() if s.isdigit()]
                    valid_keno = [n for n in tokens if 1 <= n <= 80]
                    if len(valid_keno) == 20:
                        history_draws.append(valid_keno)

            return history_draws[:limit_draws]
        else:
            return None
    except Exception as e:
        st.error(f"Lỗi kết nối cào dữ liệu: {e}")
        return None
