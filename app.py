import os
import asyncio
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from groq import Groq

# =========================
# CONFIG (NO SECRETS HERE)
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")
BASE_URL = os.getenv("BASE_URL")  # e.g. https://your-service.onrender.com

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY env var")

# =========================
# BOT STATE (IN-MEMORY)
# =========================
# Messages per chat_id
channel_messages = {}

# Chats where auto-summary is enabled
auto_summary_chats = set()


# =========================
# HELPERS
# =========================
def get_messages_by_timeframe(chat_id: int, hours: int = 24):
    messages = channel_messages.get(chat_id, [])
    if not messages:
        return []

    tz = messages[-1]["timestamp"].tzinfo
    now = datetime.now(tz=tz) if tz else datetime.now()
    cutoff = now - timedelta(hours=hours)
    return [m for m in messages if m["timestamp"] >= cutoff]


async def generate_summary(messages):
    """
    IMPORTANT:
    - For active chats this runs hierarchical summarization:
      it creates internal mini-summaries in chunks (NOT posted to Telegram),
      then returns ONE final summary to post in chat.
    """
    if not messages:
        return "No messages to summarize."

    # Collect readable lines
    lines = []
    for m in messages:
        t = (m.get("text") or "").strip()
        if not t:
            continue
        lines.append(f"[{m['timestamp'].strftime('%H:%M')}] {m['user']}: {t}")

    if not lines:
        return "No text messages found to summarize."

    # Split into blocks to avoid Groq request-too-large (413)
    blocks = []
    current = []
    current_tokens = 0
    max_tokens_per_block = 3200  # safe-ish block size

    for line in lines:
        est = max(1, len(line) // 4)  # rough tokens estimate
        if current and current_tokens + est > max_tokens_per_block:
            blocks.append("\n".join(current))
            current = [line]
            current_tokens = est
        else:
            current.append(line)
            current_tokens += est

    if current:
        blocks.append("\n".join(current))

    client = Groq(api_key=GROQ_API_KEY)

    # 1) Internal partial summaries (NOT sent to Telegram)
    partials = []
    partial_prompt_tpl = """Ты делаешь краткое резюме ЧАСТИ мамского чата.

Вытащи только полезное:
- рекомендации (врачи/товары/сервисы) с коротким "почему"
- массовые покупки / итог выбора (если виден консенсус: "я тоже", "беру", "заказала" и т.п.)
- конкретику (цены, сроки, контакты, чек-листы)
Случайные одиночные ссылки без поддержки — не включай.

Верни очень коротко и структурировано:
- Рекомендации:
- Покупки/итоги:
- Конкретика:
- Болталка (1 строка):

Сообщения:
{block}
"""

    for block in blocks:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": partial_prompt_tpl.format(block=block)}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=600,
        )
        partials.append(completion.choices[0].message.content)

    # 2) Final summary (THIS is the only thing posted to Telegram)
    final_prompt = f"""Ты объединяешь несколько кратких резюме частей мамского чата в ОДНО итоговое summary.

Правила:
- Повторы объединяй.
- Считай "итог/массовая покупка" только если видно поддержку нескольких людей (пример: "я тоже", "беру", "заказала").
- Ссылку указывай ТОЛЬКО если это рекомендация/итог/массовая покупка.
- Болталку сжать до 1–2 строк.
- Добавь Mood одной строкой.

Формат (строго):
Mood: одна короткая строка.

Полезное:
- Массовые покупки / что решили брать:
- Рекомендации (врачи / товары / сервисы):
- Полезные списки и конкретика:

Болталка (1–2 строки):

Резюме частей:
{chr(10).join(partials)}
"""

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=900,
    )
    return completion.choices[0].message.content


# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот-суммаризатор.\n\n"
        "Команды (работают в рамках текущего чата):\n"
        "/summary — summary за 24 часа\n"
        "/summary_yesterday — summary за вчера (24–48ч назад)\n"
        "/summary_custom N — summary за N часов\n"
        "/summary_days N — summary за N дней (пример: /summary_days 7)\n"
        "/clear — очистить сохраненные сообщения\n"
        "/enable_auto — включить авто-summary в 01:00 (для ЭТОГО чата)\n"
        "/disable_auto — выключить авто-summary (для ЭТОГО чата)\n\n"
        "ℹ️ Я НЕ отвечаю на обычные сообщения — я их сохраняю для summary."
    )


async def collect_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post or update.message
    if not msg:
        return

    # Ignore commands
    if msg.text and msg.text.startswith("/"):
        return

    chat_id = msg.chat.id
    channel_messages.setdefault(chat_id, [])

    # username / source
    if update.message and msg.from_user:
        user = msg.from_user.username or msg.from_user.first_name or "Unknown"
    else:
        user = msg.sender_chat.title if msg.sender_chat else "Channel"

    channel_messages[chat_id].append(
        {
            "text": msg.text or msg.caption or "",
            "timestamp": msg.date,
            "user": user,
        }
    )


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("⏳ Генерирую summary...")

    msgs = get_messages_by_timeframe(chat_id, hours=24)
    if not msgs:
        await update.message.reply_text("📭 Нет сообщений за последние 24 часа.")
        return

    summary = await generate_summary(msgs)
    await update.message.reply_text(
        f"📊 **Summary (24 часа)** ({len(msgs)} сообщений)\n\n{summary}",
        parse_mode="Markdown",
    )


async def summary_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    try:
        days = int(context.args[0]) if context.args else 1
        if days < 1 or days > 30:
            raise ValueError()
    except Exception:
        await update.message.reply_text("Пример: /summary_days 7 (1..30 дней)")
        return

    hours = days * 24
    await update.message.reply_text(f"⏳ Генерирую summary за {days} дней...")

    msgs = get_messages_by_timeframe(chat_id, hours=hours)
    if not msgs:
        await update.message.reply_text(f"📭 Нет сообщений за последние {days} дней.")
        return

    summary = await generate_summary(msgs)
    await update.message.reply_text(
        f"📊 **Summary ({days} дней)** ({len(msgs)} сообщений)\n\n{summary}",
        parse_mode="Markdown",
    )


async def summary_yesterday(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("⏳ Генерирую summary за вчера...")

    all_msgs = channel_messages.get(chat_id, [])
    if not all_msgs:
        await update.message.reply_text("📭 Нет сообщений.")
        return

    tz = all_msgs[-1]["timestamp"].tzinfo
    now = datetime.now(tz=tz) if tz else datetime.now()

    start = now - timedelta(hours=48)
    end = now - timedelta(hours=24)

    msgs = [m for m in all_msgs if start <= m["timestamp"] < end]
    if not msgs:
        await update.message.reply_text("📭 Нет сообщений за вчерашнее окно.")
        return

    summary = await generate_summary(msgs)
    await update.message.reply_text(
        f"📊 **Summary (вчера)** ({len(msgs)} сообщений)\n\n{summary}",
        parse_mode="Markdown",
    )


async def summary_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    try:
        hours = int(context.args[0]) if context.args else 24
        if hours < 1 or hours > 168:
            raise ValueError()
    except Exception:
        await update.message.reply_text("Пример: /summary_custom 12 (1..168 часов)")
        return

    await update.message.reply_text(f"⏳ Генерирую summary за {hours} часов...")

    msgs = get_messages_by_timeframe(chat_id, hours=hours)
    if not msgs:
        await update.message.reply_text(f"📭 Нет сообщений за последние {hours} часов.")
        return

    summary = await generate_summary(msgs)
    await update.message.reply_text(
        f"📊 **Summary ({hours}ч)** ({len(msgs)} сообщений)\n\n{summary}",
        parse_mode="Markdown",
    )


async def clear_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    count = len(channel_messages.get(chat_id, []))
    channel_messages[chat_id] = []
    await update.message.reply_text(f"✅ Очищено {count} сообщений.")


async def enable_auto_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    auto_summary_chats.add(chat_id)
    await update.message.reply_text(
        "✅ Авто-summary включено для ЭТОГО чата.\n"
        "Я буду отправлять daily summary в 01:00 (по времени сервера)."
    )
    print(f"✅ Auto-summary enabled for chat: {chat_id}")


async def disable_auto_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    auto_summary_chats.discard(chat_id)
    await update.message.reply_text("❌ Авто-summary выключено для ЭТОГО чата.")
    print(f"❌ Auto-summary disabled for chat: {chat_id}")


async def send_auto_summary(ptb: Application):
    if not auto_summary_chats:
        print("⏭️ Skipping auto-summary: no chats enabled")
        return

    for chat_id in list(auto_summary_chats):
        msgs = get_messages_by_timeframe(chat_id, hours=24)
        if not msgs:
            print(f"📭 No messages for chat {chat_id}")
            continue

        summary = await generate_summary(msgs)

        try:
            await ptb.bot.send_message(
                chat_id=chat_id,
                text=f"🌙 **Daily Summary**\n📅 24 часа — {len(msgs)} сообщений\n\n{summary}",
                parse_mode="Markdown",
            )
            print(f"✅ Auto-summary sent to chat {chat_id}")
        except Exception as e:
            print(f"❌ Error sending auto-summary to {chat_id}: {e}")


async def schedule_daily_summary(ptb: Application):
    while True:
        now = datetime.now()
        target = now.replace(hour=1, minute=0, second=0, microsecond=0)
        if now.hour >= 1:
            target += timedelta(days=1)

        wait_s = (target - now).total_seconds()
        print(f"⏰ Next auto-summary scheduled for: {target.isoformat()}")
        await asyncio.sleep(wait_s)

        await send_auto_summary(ptb)
        await asyncio.sleep(60)


# =========================
# FASTAPI + WEBHOOK
# =========================
app = FastAPI()
ptb_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()


@app.on_event("startup")
async def on_startup():
    # Commands
    ptb_app.add_handler(CommandHandler("start", start))
    ptb_app.add_handler(CommandHandler("summary", summary_command))
    ptb_app.add_handler(CommandHandler("summary_yesterday", summary_yesterday))
    ptb_app.add_handler(CommandHandler("summary_custom", summary_custom))
    ptb_app.add_handler(CommandHandler("summary_days", summary_days))
    ptb_app.add_handler(CommandHandler("clear", clear_messages))
    ptb_app.add_handler(CommandHandler("enable_auto", enable_auto_summary))
    ptb_app.add_handler(CommandHandler("disable_auto", disable_auto_summary))

    # Collect ANY message except commands
    ptb_app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, collect_message))

    await ptb_app.initialize()
    await ptb_app.start()

    # Scheduler
    asyncio.create_task(schedule_daily_summary(ptb_app))

    # Webhook
    if BASE_URL:
        webhook_url = f"{BASE_URL}/telegram/{WEBHOOK_SECRET}"
        await ptb_app.bot.set_webhook(url=webhook_url)
        print(f"✅ Webhook set: {webhook_url}")
    else:
        print("⚠️ BASE_URL is not set yet. Webhook not configured.")


@app.on_event("shutdown")
async def on_shutdown():
    await ptb_app.stop()
    await ptb_app.shutdown()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/telegram/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}

    payload = await request.json()
    update = Update.de_json(payload, ptb_app.bot)
    await ptb_app.process_update(update)
    return {"ok": True}
