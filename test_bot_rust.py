import asyncio
import logging
from forex_bot.core.config import Settings
from forex_bot.execution.bot import ForexBot

async def main():
    logging.basicConfig(level=logging.INFO)
    # Settings() will load from config.yaml via YamlConfigSettingsSource
    settings = Settings()
    bot = ForexBot(settings)
    
    print("Testing ForexBot.train()...")
    await bot.train()
    print("ForexBot.train() complete.")

if __name__ == "__main__":
    asyncio.run(main())
