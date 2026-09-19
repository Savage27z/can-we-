"""Sets the bot's public profile text on Telegram: the "What can this bot do?" card
shown in an empty chat, and the short description on the bot's profile page.

That text lives on Telegram's side, not in this repo, so this script is the source of
truth for it. Re-run after editing:  python -m tgbot.set_profile
"""
import asyncio

from telegram import Bot

from . import config

# Telegram's limits: 512 characters for the description, 120 for the short one.
DESCRIPTION = (
    "Forex structural-analysis bot.\n\n"
    "Reads liquidity sweeps, fair value gaps and confirmation across the Daily, "
    "H4 and H1 timeframes, with a high-impact news check.\n\n"
    "Structural analysis, not financial advice. You make your own decisions."
)
SHORT_DESCRIPTION = (
    "Forex structural analysis: sweeps, FVGs and confirmation on Daily/H4/H1. "
    "Not financial advice."
)


async def apply_profile(token: str) -> None:
    async with Bot(token) as bot:
        await bot.set_my_description(DESCRIPTION)
        await bot.set_my_short_description(SHORT_DESCRIPTION)


def main() -> None:
    asyncio.run(apply_profile(config.require_token()))
    print("Bot description and short description updated.")


if __name__ == "__main__":
    main()
