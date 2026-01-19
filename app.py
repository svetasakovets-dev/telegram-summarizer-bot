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

# Telegram message length safety
MAX_TG_LEN = 3500  # Telegram max is 4096, keep margin


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


async def safe_send_message(ptb: Application, chat_id: int, text: str):
    """Send long text to Telegram in chunks."""
    if not text:
