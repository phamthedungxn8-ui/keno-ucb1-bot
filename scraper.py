import json
import os
import urllib.request

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# Endpoint API chính thức của Vietlott Keno
API_URL = "https://vietlott.vn/api/front/keno/latest"


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = json.dumps(
        {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ),
        },
    )
    try:
        with urllib.request.urlopen(req) as response:
            pass
    except Exception as e:
        print(f"Lỗi gửi Telegram: {e}")


def main():
    # Thêm Header giả lập trình duyệt chuẩn để vượt rào chặn HTTP 403
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://vietlott.vn/",
        "Origin": "https://vietlott.vn",
    }

    req = urllib.request.Request(API_URL, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode("utf-8")
            data = json.loads(res_body)

            results = []
            if isinstance(data, dict):
                results = data.get("ResultList", []) or data.get(
                    "resultList", []
                )
            elif isinstance(data, list):
                results = data

            if not results:
                send_telegram_message(
                    "⚠️ API chưa có dữ liệu mở thưởng mới."
                )
                return

            latest = results[0]
            draw_id = f"#{str(latest.get('DrawId', '')).zfill(7)}"
            raw_nums = latest.get("QueryResult", [])

            if isinstance(raw_nums, str):
                numbers = sorted(
                    [
                        int(n)
                        for n in raw_nums.replace("|", ",").split(",")
                        if n.strip()
                    ]
                )
            else:
                numbers = sorted([int(n) for n in raw_nums])

            str_nums = " ".join([f"{n:02d}" for n in numbers])

            msg = (
                f"🎉 **KẾT QUẢ KENO MỚI ({draw_id})**\n\n"
                f"📌 **20 số trúng thưởng:**\n`{str_nums}`"
            )

            send_telegram_message(msg)
            print(f"✅ Đã gửi thành công kỳ {draw_id}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        send_telegram_message(f"❌ Lỗi cào dữ liệu: {e}")


if __name__ == "__main__":
    main()
