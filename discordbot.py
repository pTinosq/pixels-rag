import os
import logging
import asyncio
import discord
from dotenv import load_dotenv
import httpx

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

ALLOWED_GUILD_ID = 857732000890748998


async def confirm_send(message_preview: str) -> bool:
    """Prompt user for confirmation before sending a message."""
    loop = asyncio.get_event_loop()
    print("\n" + "=" * 60)
    print("Message to send (first 200 chars):")
    preview = message_preview[:1500] + ("..." if len(message_preview) > 1500 else "")
    print(preview)
    print("=" * 60)
    response = await loop.run_in_executor(None, input, "Send this message? (y/n): ")
    return response.lower().strip() == "y"


@client.event
async def on_ready():
    logger.info(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.guild is None or message.guild.id != ALLOWED_GUILD_ID:
        return

    if message.author == client.user:
        return

    if message.content.lower().startswith("$prompt"):
        parts = message.content.split(maxsplit=1)
        prompt = parts[1] if len(parts) > 1 else None
        if prompt is None:
            logger.warning(f"User {message.author} sent $prompt without a prompt")
            await message.channel.send("Please provide a prompt after $prompt.")
            return

        logger.info(f"Received prompt from {message.author}: {prompt}")
        url = "http://localhost:5000/"
        params = {"prompt": prompt}
        try:
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(url, params=params, timeout=120.0)
                response.raise_for_status()
                data = response.text
                logger.info(f"Successfully got response (length: {len(data)})")
                if await confirm_send(data):
                    await message.channel.send(data)
                    logger.info("Message sent to Discord")
                else:
                    logger.info("Message sending cancelled by user")
                    await message.channel.send("Message sending was cancelled.")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code} for prompt '{prompt}': {e.response.text}",
                exc_info=True,
            )
            error_msg = (
                f"Server returned error {e.response.status_code}: {e.response.text}"
            )
            if await confirm_send(error_msg):
                await message.channel.send(error_msg)
            else:
                logger.info("Error message sending cancelled by user")
        except httpx.ReadError as e:
            logger.error(
                f"Connection read error for prompt '{prompt}': {str(e)}", exc_info=True
            )
            error_msg = (
                "The connection to the server was interrupted. "
                "The server may have crashed or closed the connection unexpectedly. "
                "Please check if the server is running."
            )
            if await confirm_send(error_msg):
                await message.channel.send(error_msg)
            else:
                logger.info("Error message sending cancelled by user")
        except httpx.ConnectError as e:
            logger.error(
                f"Connection error for prompt '{prompt}': {str(e)}", exc_info=True
            )
            error_msg = (
                "Could not connect to the server. "
                "Please make sure the server is running on http://localhost:5000"
            )
            if await confirm_send(error_msg):
                await message.channel.send(error_msg)
            else:
                logger.info("Error message sending cancelled by user")
        except httpx.TimeoutException as e:
            logger.error(
                f"Request timeout for prompt '{prompt}': {str(e)}", exc_info=True
            )
            error_msg = (
                "The request timed out after 120 seconds. "
                "The server may be taking too long to respond."
            )
            if await confirm_send(error_msg):
                await message.channel.send(error_msg)
            else:
                logger.info("Error message sending cancelled by user")
        except httpx.RequestError as e:
            logger.error(
                f"Request error for prompt '{prompt}': {str(e)}", exc_info=True
            )
            error_msg = (
                f"An error occurred while connecting to the server: {type(e).__name__}"
            )
            if await confirm_send(error_msg):
                await message.channel.send(error_msg)
            else:
                logger.info("Error message sending cancelled by user")
        except Exception as e:
            logger.error(
                f"Unexpected error processing prompt '{prompt}': {str(e)}",
                exc_info=True,
            )
            error_msg = f"An unexpected error occurred: {e}"
            if await confirm_send(error_msg):
                await message.channel.send(error_msg)
            else:
                logger.info("Error message sending cancelled by user")


client.run(os.getenv("DISCORD_BOT_TOKEN"))
