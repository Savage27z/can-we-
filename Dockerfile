FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    DATA_DIR=/data

WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Only what the bot imports at runtime (live/ builds on backtest/'s rule modules).
# Tests, docs, local data and .env are deliberately not copied — see .dockerignore.
COPY data_pipeline/ data_pipeline/
COPY backtest/ backtest/
COPY live/ live/
COPY news/ news/
COPY narration/ narration/
COPY tgbot/ tgbot/

# Candle data, the news cache and the sent-alerts log live here. On Railway, mount a
# volume at /data so they survive redeploys.
RUN mkdir -p /data

CMD ["python", "-m", "tgbot.run_bot"]
