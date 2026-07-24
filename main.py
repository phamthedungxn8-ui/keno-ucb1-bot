import os
import asyncio
import numpy as np
from aiohttp import web
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = os.environ.get("BOT_TOKEN", "8954250463:AAFvdLym7wBkHAWkBPFjtnKRRvqmrr86Bn0")

# ==========================================
# UCB1 BANDIT ENGINE
# ==========================================
class UCB1BanditEngine:
    def __init__(self, num_strategies=3):
        self.num_strategies = num_strategies
        self.counts = np.zeros(num_strategies)
        self.rewards = np.zeros(num_strategies)
        self.total_rounds = 0

    def select_strategy(self):
        self.total_rounds += 1
        for i in range(self.num_strategies):
            if self.counts[i] == 0:
                return i
        ucb_values = np.zeros(self.num_strategies)
        for i in range(self.num_strategies):
            avg_reward = self.rewards[i] / self.counts[i]
            bonus = np.sqrt((2 * np.log(self.total_rounds)) / self.counts[i])
            ucb_values[i] = avg_reward + bonus
        return int(np.argmax(ucb_values))

    def update_reward(self, strat_idx, reward):
        self.counts[strat_idx] += 1
        self.rewards[strat_idx] += reward

bandit_agent = UCB1BanditEngine(num_strategies=3)
STRATEGY_NAMES = [
    "Ma Trận Đồ Thị Không Gian (Graph Dynamics)",
    "Mô Hình Chuỗi Markov (Transition Matrix)",
    "Cầu Tần Suất Lặp & Nhịp Điệu (Frequency)"
]

def strategy_graph(nums):
    scores = np.zeros(81)
    for n in nums:
        for adj in [n-1, n+1, n-10, n+10]:
            if 1 <= adj <= 80: scores[adj] += 1.5
    return np.argsort(scores)[::-1][:10]

def strategy_markov(nums):
    scores = np.zeros(81)
    for n in nums:
        rev = int(str(n)[::-1]) if n >= 10 else n * 10
        if 1 <= rev <= 80: scores[rev] += 2.0
        scores[n] += 0.8
    return np.argsort(scores)[::-1][:10]

def strategy_frequency(nums):
    scores = np.zeros(81)
    for n in nums:
        scores[n] += 1.0
        if n % 2 == 0: scores[min(80, n+2)] += 0.5
    return np.argsort(scores)[::-1][:10]

# ==========================================
# TELEGRAM BOT HANDLERS
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "<b>🤖 AGENT UCB1 MULTI-ARMED BANDIT KENO</b>\n\n"
        "Hệ thống tự động chọn <b>Strategy đang vào dây đỏ nhất</b> để dự đoán.\n\n"
        "Gửi 20 số Keno hiện tại để kích hoạt Agent!"
    )
    await update.message.reply_text(msg, parse_mode="HTML")

async def process_numbers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    nums = [int(n) for n in text.replace(',', ' ').split() if n.isdigit() and 1 <= int(n) <= 80]
    nums = list(set(nums))

    if len(nums) != 20:
        await update.message.reply_text(f"⚠️ Vui lòng nhập đúng 20 số Keno (Phát hiện {len(nums)} số).")
        return

    best_strat_idx = bandit_agent.select_strategy()
    strat_name = STRATEGY_NAMES[best_strat_idx]

    if best_strat_idx == 0:
        top_10 = strategy_graph(nums)
    elif best_strat_idx == 1:
        top_10 = strategy_markov(nums)
    else:
        top_10 = strategy_frequency(nums)

    bandit_agent.update_reward(best_strat_idx, reward=1.0)

    res = f"🧠 <b>AGENT UCB1 DECISION ENGINE</b>\n"
    res += f"━━━━━━━━━━━━━━━━━━━\n"
    res += f"🎯 <b>Chiến thuật được chọn:</b>\n<code>{strat_name}</code>\n\n"
    res += f"🔥 <b>TOP 6 SỐ TỐI ƯU NHẤT:</b>\n"
    res += f"👉 <b>[ {', '.join([f'{x:02d}' for x in top_10[:6]])} ]</b>\n\n"
    res += f"📊 <b>Cặp Lô Xiên 2:</b> [{top_10[0]:02d} - {top_10[1]:02d}], [{top_10[2]:02d} - {top_10[3]:02d}]\n"
    res += f"━━━━━━━━━━━━━━━━━━━\n"
    res += f"📈 <i>Tổng số lần Agent tự điều chỉnh: {bandit_agent.total_rounds} kỳ.</i>"

    await update.message.reply_text(res, parse_mode="HTML")

# ==========================================
# WEB SERVER & ASYNC MAIN
# ==========================================
async def handle_health_check(request):
    return web.Response(text="Bot Telegram UCB1 is Live & Healthy!")

async def main():
    # 1. Khởi tạo Web Server cho Render Health Check
    app_web = web.Application()
    app_web.router.add_get("/", handle_health_check)
    runner = web.AppRunner(app_web)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    # 2. Khởi tạo Telegram Bot
    tg_app = ApplicationBuilder().token(TOKEN).build()
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_numbers))

    async with tg_app:
        await tg_app.start()
        await tg_app.updater.start_polling()
        print("🤖 Agent UCB1 Bot đang chạy mượt mà...")
        # Giữ loop chạy mãi mãi
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
