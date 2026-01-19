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
# CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")
BASE_URL = os.getenv("BASE_URL")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

# =========================
# STATE
# =========================
channel_messages = {}
auto_summary_chats = set()

MAX_TG_LEN = 3500


# =========================
# HELPERS
# =========================
def get_messages_by_timeframe(chat_id: int, hours: int):
    messages = channel_messages.get(chat_id, [])
    if not messages:
        return []

    tz = messages[-1]["timestamp"].tzinfo
    now = datetime.now(tz=tz) if tz else datetime.now()
    cutoff = now - timedelta(hours=hours)
    return [m for m in messages if m["timestamp"] >= cutoff]


async def safe_reply(update: Update, text: str):
    """Send long messages safely (Telegram limit)."""
    if not text:
        await update.message.reply_text("Пустой ответ от модели.")
        return

    if len(text) <= MAX_TG_LEN:
        await update.message.reply_text(text)
        return

    parts = []
    buf = ""
    for block in text.split("\n\n"):
        if len(buf) + len(block) + 2 <= MAX_TG_LEN:
            buf = (buf + "\n\n" + block).strip()
        else:
            parts.append(buf)
            buf = block
    if buf:
        parts.append(buf)

    for p in parts:
        await update.message.reply_text(p)


async def run_with_timeout(coro, seconds=60):
    return await asyncio.wait_for(coro, timeout=seconds)


# =========================
# SUMMARY LOGIC
# =========================
async def generate_summary(messages):
    lines = []
    for m in messages:
        txt = (m.get("text") or "").strip()
        if txt:
            lines.append(f"[{m['timestamp'].strftime('%H:%M')}] {m['user']}: {txt}")

    if not lines:
        return "Нет текстовых сообщений."

    # Chunking
    blocks, cur, size = [], [], 0
    for line in lines:
        est = max(1, len(line) // 4)
        if cur and size + est > 3000:
            blocks.append("\n".join(cur))
            cur, size = [line], est
        else:
            cur.append(line)
            size += est
    if cur:
        blocks.append("\n".join(cur))

    client = Groq(api_key=GROQ_API_KEY)
    partials = []

    partial_prompt = """Выжимка ЧАСТИ чата.

Правила:
- Не выдумывай.
- Вытаскивай конкретные сцены, факты, цифры.
- Покупки: только если реально купили/заказали + что именно + где.
- Ссылки: только реальные URL из сообщений.

Формат:
Сцены:
Факты:
Покупки:
Рекомендации:
Ссылки:
Планы:
"""

    for b in blocks:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=700,
            messages=[{"role": "user", "content": partial_prompt + "\n" + b}],
        )
        partials.append(r.choices[0].message.content)

    final_prompt = f"""Сделай сторителлинг-саммари дня по переписке.

Тон:
- без приветствий и пожеланий
- живо, но по делу
- без воды

Правила:
- не обобщай ("делились историями"), вытаскивай факты
- если нет конкретики — не пиши
- ссылки только реальные

Формат:
Заголовок (1 строка)

Текст (5–10 коротких абзацев)

Покупки:
— пункты или "— не было"

Рекомендации:
— пункты или "— не было"

Ссылки:
— реальные URL или "— не было"

Планы:
— или "— не было"

Материал:
{chr(10).join(partials)}
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1100,
        messages=[{"role": "user", "content": final_prompt}],
    )
    return res.choices[0].message.content


# =========================
# HANDLERS
# =========================
async def collect_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message or update.channel_post
    if not msg or (msg.text and msg.text.startswith("/")):
        return

    channel_messages.setdefault(msg.chat.id, []).append(
        {
            "text": msg.text or msg.caption or "",
            "timestamp": msg.date,
            "user": msg.from_user.first_name if msg.from_user else "Channel",
        }
    )


async def summary_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        hours = int(context.args[0]) if context.args else 1
    except Exception:
        await update.message.reply_text("Пример: /summary_custom 1")
        return

    await update.message.reply_text(f"⏳ Делаю саммари за {hours}ч...")

    try:
        msgs = get_messages_by_timeframe(update.effective_chat.id, hours)
        if not msgs:
            await update.message.reply_text("Нет сообщений.")
            return

        text = await run_with_timeout(generate_summary(msgs))
        await safe_reply(update, text)

    except asyncio.TimeoutError:
        await update.message.reply_text("Groq не ответил вовремя. Попробуй меньший период.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.args = ["24"]
    await summary_custom(update, context)


# =========================
# FASTAPI
# =========================
app = FastAPI()
ptb = Application.builder().token(TELEGRAM_BOT_TOKEN).build()


@app.on_event("startup")
async def startup():
    ptb.add_handler(CommandHandler("summary", summary_command))
    ptb.add_handler(CommandHandler("summary_custom", summary_custom))
    ptb.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, collect_message))

    await ptb.initialize()
    await ptb.start()

    if BASE_URL:
        await ptb.bot.set_webhook(f"{BASE_URL}/telegram/{WEBHOOK_SECRET}")


@app.post("/telegram/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}
    data = await request.json()
    update = Update.de_json(data, ptb.bot)
    await ptb.process_update(update)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"ok": True}
