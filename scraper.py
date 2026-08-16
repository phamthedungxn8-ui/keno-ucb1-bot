import json
import os
import urllib.request

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
API_URL = "https://vietlott.vn/api/front/keno/latest"


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = json.dumps(
        {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            pass
    except Exception as e:
        print(f"Lỗi gửi tin nhắn Telegram: {e}")


def main():
    req = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            results = (
                data.get("ResultList", [])
                if isinstance(data, dict)
                else data
            )

            if results:
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
                    f"📌 20 số trúng thưởng:\n`{str_nums}`"
                )
                send_telegram_message(msg)
                print(f"✅ Đã gửi kết quả kỳ {draw_id}")
    except Exception as e:
        print(f"❌ Lỗi khi cào dữ liệu: {e}")


if __name__ == "__main__":
    main()
