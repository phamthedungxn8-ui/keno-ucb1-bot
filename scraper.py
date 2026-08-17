import json
import os
import urllib.request

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
API_URL = "https://www.minhngoc.com.vn/json/keno.json"


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
    req = urllib.request.Request(
        API_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

            draw_id = f"#{data.get('ky', '')}"
            raw_nums = data.get("ketqua", [])

            numbers = sorted([int(n) for n in raw_nums if str(n).isdigit()])
            str_nums = " ".join([f"{n:02d}" for n in numbers])

            msg = (
                f"🎉 **KẾT QUẢ KENO MỚI ({draw_id})**\n\n"
                f"📌 **20 số trúng thưởng:**\n`{str_nums}`"
            )
            send_telegram_message(msg)
            print(f"✅ Đã gửi kết quả kỳ {draw_id}")
    except Exception as e:
        print(f"❌ Lỗi cào dữ liệu: {e}")
        send_telegram_message(f"❌ Lỗi cào dữ liệu: {e}")


if __name__ == "__main__":
    main()
