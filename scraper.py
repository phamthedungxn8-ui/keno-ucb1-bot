import json
import re
import os
import urllib.request

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
WEB_URL = "https://m.ketqua.vn/xo-so-keno"

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}).encode('utf-8')
    req = urllib.request.Request(
        url, 
        data=payload, 
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
    )
    try:
        with urllib.request.urlopen(req) as response:
            pass
    except Exception as e:
        print(f"Lỗi gửi Telegram: {e}")

def main():
    req = urllib.request.Request(
        WEB_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')
            
            # Tìm danh sách các số trúng thưởng từ trang web
            numbers_match = re.findall(r'class="btn-keno[^"]*">(\d{1,2})<', html)
            draw_match = re.search(r'Kỳ\s*#?(\d+)', html, re.IGNORECASE)
            
            if numbers_match:
                # Lấy 20 số đầu tiên thu được
                nums = [int(n) for n in numbers_match[:20]]
                nums_sorted = sorted(nums)
                str_nums = " ".join([f"{n:02d}" for n in nums_sorted])
                
                draw_id = f"#{draw_match.group(1)}" if draw_match else ""
                
                msg = (
                    f"🎉 **KẾT QUẢ KENO MỚI ({draw_id})**\n\n"
                    f"📌 **20 số trúng thưởng:**\n`{str_nums}`"
                )
                send_telegram_message(msg)
                print("✅ Đã gửi tin nhắn thành công!")
            else:
                send_telegram_message("⚠️ Không tìm thấy cấu trúc số Keno trên trang web.")
    except Exception as e:
        print(f"❌ Lỗi cào dữ liệu: {e}")
        send_telegram_message(f"❌ Lỗi cào dữ liệu: {e}")

if __name__ == "__main__":
    main()
