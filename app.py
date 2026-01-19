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
BASE_URL = os.getenv("BASE_URL")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY env var")

# =========================
# BOT STATE (IN-MEMORY)
# =========================
channel_messages = {}
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
    Hierarchical summarization:
    - internal chunk summaries (NOT sent to chat)
    - ONE final summary returned
    """
    if not messages:
        return "Нет сообщений для саммари."

    lines = []
    for m in messages:
        t = (m.get("text") or "").strip()
        if t:
            lines.append(f"[{m['timestamp'].strftime('%H:%M')}] {m['user']}: {t}")

    if not lines:
        return "Нет текстовых сообщений."

    # Split into safe blocks
    blocks = []
    current = []
    current_tokens = 0
    max_tokens_per_block = 3200

    for line in lines:
        est = max(1, len(line) // 4)
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

    # ---------- PARTIAL SUMMARIES ----------
    partials = []

    partial_prompt_tpl = """Ты делаешь ОЧЕНЬ краткое резюме ЧАСТИ мамского чата.

ЖЁСТКИЕ ПРАВИЛА:
- НИЧЕГО НЕ ВЫДУМЫВАЙ.
- НЕ добавляй упоминания без конкретики.
- "Блины на районе", "косметичка", "магазин" БЕЗ:
  что именно + где/как найти → НЕ добавлять.
- Ссылки игнорируй, если это не явная рекомендация.
- Рекомендация = конкретный объект + где + почему нравится.
- Консенсус = минимум 2 человека ("я тоже", "беру", "заказала").

Формат:
- Подтверждённые рекомендации (2+ человек, с конкретикой):
- Одиночные рекомендации (ТОЛЬКО если явно "советую" и есть конкретика):
- Итоги / массовые покупки:
- Цены / скидки / конкретика:
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

    # ---------- FINAL SUMMARY ----------
    final_prompt = f"""Ты объединяешь несколько резюме частей мамского чата в ОДНО итоговое summary.

ЖЁСТКИЕ ПРАВИЛА:
- НИЧЕГО НЕ ВЫДУМЫВАЙ.
- Если нет конкретики (что + где) — НЕ добавляй.
- Если нет консенсуса или явного "советую" — НЕ добавляй.
- Ссылки выводи ТОЛЬКО если это подтверждённая рекомендация или итог покупки.
- Максимум 10–15 ссылок на ВСЁ summary.
- Если раздел пуст — пиши "— нет".

ФОРМАТ (строго):
Mood: одна короткая строка.

Полезное:
- Подтверждённые рекомендации (2+ человек, с конкретикой): ...
- Итоги / массовые покупки (что именно и где): ...
- Одиночные рекомендации (явно "советую", с конкретикой): ...
- Цены / скидки / конкретика: ...

Болталка (1–2 строки): ...

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
        "👋 Я делаю итоговые саммари чата.\n\n"
        "Команды:\n"
        "/summary — за 24 часа\n"
        "/summary_custom N — за N часов\n"
        "/summary_days N — за N дней\n"
        "/enable_auto — авто-саммари\n"
        "/disable_auto — отключить авто-саммари\n\n"
        "Я НЕ отвечаю на обычные сообщения."
    )


async def collect_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post or update.message
    if not msg:
        return
    if msg.text and msg.text.startswith("/"):
        return

    chat_id = msg.chat.id
    channel_messages.setdefault(chat_id, [])

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
    await update.message.reply_text("⏳ Готовлю саммари...")
    msgs = get_messages_by_timeframe(chat_id, 24)
    if not msgs:
        await update.message.reply_text("Нет сообщений.")
        return
    summary = await generate_summary(msgs)
    await update.message.reply_text(summary, parse_mode="Markdown")


async def summary_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        hours = int(context.args[0])
    except Exception:
        await update.message.reply_text("Пример: /summary_custom 12")
        return
    msgs = get_messages_by_timeframe(chat_id, hours)
    summary = await generate_summary(msgs)
    await update.message.reply_text(summary, parse_mode="Markdown")


async def summary_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        days = int(context.args[0])
    except Exception:
        await update.message.reply_text("Пример: /summary_days 7")
        return
    msgs = get_messages_by_timeframe(chat_id, days * 24)
    summary = await generate_summary(msgs)
    await update.message.reply_text(summary, parse_mode="Markdown")


async def enable_auto_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    auto_summary_chats.add(update.effective_chat.id)
    await update.message.reply_text("✅ Авто-саммари включено.")


async def disable_auto_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    auto_summary_chats.discard(update.effective_chat.id)
    await update.message.reply_text("❌ Авто-саммари выключено.")


async def send_auto_summary(ptb: Application):
    for chat_id in list(auto_summary_chats):
        msgs = get_messages_by_timeframe(chat_id, 24)
        if msgs:
            summary = await generate_summary(msgs)
            await ptb.bot.send_message(chat_id=chat_id, text=summary, parse_mode="Markdown")


async def schedule_daily_summary(ptb: Application):
    while True:
        now = datetime.now()
        target = now.replace(hour=1, minute=0, second=0, microsecond=0)
        if now.hour >= 1:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        await send_auto_summary(ptb)
        await asyncio.sleep(60)


# =========================
# FASTAPI + WEBHOOK
# =========================
app = FastAPI()
ptb_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()


@app.on_event("startup")
async def on_startup():
    ptb_app.add_handler(CommandHandler("start", start))
    ptb_app.add_handler(CommandHandler("summary", summary_command))
    ptb_app.add_handler(CommandHandler("summary_custom", summary_custom))
    ptb_app.add_handler(CommandHandler("summary_days", summary_days))
    ptb_app.add_handler(CommandHandler("enable_auto", enable_auto_summary))
    ptb_app.add_handler(CommandHandler("disable_auto", disable_auto_summary))
    ptb_app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, collect_message))

    await ptb_app.initialize()
    await ptb_app.start()

    asyncio.create_task(schedule_daily_summary(ptb_app))

    if BASE_URL:
        await ptb_app.bot.set_webhook(f"{BASE_URL}/telegram/{WEBHOOK_SECRET}")


@app.post("/telegram/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}
    data = await request.json()
    update = Update.de_json(data, ptb_app.bot)
    await ptb_app.process_update(update)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"ok": True}
