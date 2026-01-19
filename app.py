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
    - ONE final story-style summary returned
    """
    if not messages:
        return "Нет сообщений для саммари."

    lines = []
    for m in messages:
        t = (m.get("text") or "").strip()
        if not t:
            continue
        lines.append(f"[{m['timestamp'].strftime('%H:%M')}] {m['user']}: {t}")

    if not lines:
        return "Нет текстовых сообщений."

    # Chunking to avoid 413 / token limits
    blocks = []
    current = []
    current_tokens = 0
    max_tokens_per_block = 3200

    for line in lines:
        est = max(1, len(line) // 4)  # rough token estimate
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

    # ---------- PARTIAL SUMMARIES (internal) ----------
    partials = []
    partial_prompt_tpl = """Сделай выжимку ЧАСТИ чата. Нужны факты и конкретика.

Жёсткие правила:
- Не выдумывай.
- Не пиши "в целом обсуждали..." — вытаскивай конкретные детали: кто/что/сколько/где.
- Краши/мемы/болтовня — ок, но 1–2 строки, с конкретными примерами (имена/смешные детали).
- Покупки фиксируй только если есть:
  (а) явный сигнал "купила/заказала/беру/в итоге решили",
  (б) конкретика: что именно (бренд/модель/тип) + где/как найти + цена/скидка (если было).
- Ссылки: выписывай только реальные URL, которые есть в тексте. Не добавляй просто "vinted.com" если не было URL.
- Рекомендации (врачи/места/сервисы) фиксируй только если есть конкретное название/специализация/место/причина.

Верни строго в формате:
Сюжеты/сцены (2–5 коротких строк, с конкретикой):
Факты/цифры/интересные детали (0–5 пунктов, иначе "— нет"):
Покупки (0–5 пунктов, иначе "— нет"):
Рекомендации (0–5 пунктов, иначе "— нет"):
Ссылки (только реальные URL, иначе "— нет"):
Планы (иначе "— нет"):

Сообщения:
{block}
"""

    for block in blocks:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": partial_prompt_tpl.format(block=block)}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=750,
        )
        partials.append(completion.choices[0].message.content)

    # ---------- FINAL SUMMARY (only this is posted) ----------
    final_prompt = f"""Сделай итоговый пост-резюме дня по переписке. Это должно читаться как короткий сторителл, но без слащавости.

Тон и стиль:
- Без "привет/доброй ночи/желаю" и без мотивационных финалов.
- Без эмодзи (если очень нужно — максимум 0–1 на весь текст).
- Человечно и по делу: не канцелярит, но и не "праздничная открытка".

Жёсткие правила:
- Не выдумывай ничего.
- Не перечисляй "кто что сказал" длинным списком. Имена допустимы точечно, если делают текст живым.
- Не пиши абстракции ("делились историями о весе") — вместо этого вытащи 1–3 конкретных факта (например: "минус 3 кг", "муж минус 13", "цель минус 30").
- Ссылки показывай только реальные URL, которые были в сообщениях. Никаких "vinted.com" если не было прямой ссылки.
- Если по покупкам нет конкретики — не добавляй в покупки.
- Рекомендации по врачам/местам/сервисам — добавляй только если есть конкретные названия/специализация/место/почему.

Формат (строго):
Заголовок: 1 строка (коротко, по главному сюжету дня).

Текст: 4–10 коротких абзацев (1–3 предложения каждый). Это "что происходило" с конкретными деталями.

Покупки:
- 0–5 пунктов. Если не было — "— не было".

Рекомендации (врачи/места/сервисы):
- 0–5 пунктов. Если не было — "— не было".

Ссылки:
- только реальные URL (0–8 строк). Если не было — "— не было".

Планы:
- если были; иначе "— не было".

Материал (резюме частей):
{chr(10).join(partials)}
"""

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1100,
    )
    return completion.choices[0].message.content


# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Делаю сторителл-саммари чата без сюсюканья, но с конкретикой.\n\n"
        "Команды:\n"
        "/summary — за 24 часа\n"
        "/summary_custom N — за N часов\n"
        "/summary_days N — за N дней\n"
        "/enable_auto — авто-саммари в 01:00 (для этого чата)\n"
        "/disable_auto — отключить авто-саммари"
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
    await update.message.reply_text("⏳ Делаю саммари...")
    msgs = get_messages_by_timeframe(chat_id, 24)
    if not msgs:
        await update.message.reply_text("📭 Нет сообщений за период.")
        return
    summary = await generate_summary(msgs)
    await update.message.reply_text(summary)


async def summary_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        hours = int(context.args[0]) if context.args else 24
        if hours < 1 or hours > 168:
            raise ValueError()
    except Exception:
        await update.message.reply_text("Пример: /summary_custom 12 (1..168 часов)")
        return

    await update.message.reply_text(f"⏳ Саммари за {hours}ч...")
    msgs = get_messages_by_timeframe(chat_id, hours)
    if not msgs:
        await update.message.reply_text("📭 Нет сообщений за период.")
        return
    summary = await generate_summary(msgs)
    await update.message.reply_text(summary)


async def summary_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        days = int(context.args[0]) if context.args else 1
        if days < 1 or days > 30:
            raise ValueError()
    except Exception:
        await update.message.reply_text("Пример: /summary_days 7 (1..30 дней)")
        return

    await update.message.reply_text(f"⏳ Саммари за {days}д...")
    msgs = get_messages_by_timeframe(chat_id, days * 24)
    if not msgs:
        await update.message.reply_text("📭 Нет сообщений за период.")
        return
    summary = await generate_summary(msgs)
    await update.message.reply_text(summary)


async def enable_auto_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    auto_summary_chats.add(update.effective_chat.id)
    await update.message.reply_text("✅ Авто-саммари включено для этого чата (01:00 по времени сервера).")


async def disable_auto_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    auto_summary_chats.discard(update.effective_chat.id)
    await update.message.reply_text("❌ Авто-саммари выключено для этого чата.")


async def send_auto_summary(ptb: Application):
    if not auto_summary_chats:
        return

    for chat_id in list(auto_summary_chats):
        msgs = get_messages_by_timeframe(chat_id, 24)
        if not msgs:
            continue
        summary = await generate_summary(msgs)
        try:
            await ptb.bot.send_message(chat_id=chat_id, text=summary)
        except Exception as e:
            print(f"❌ Auto-summary send error for {chat_id}: {e}")


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
