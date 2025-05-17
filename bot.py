import os
import mimetypes
import logging
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from datetime import datetime

# Import configuration from config.py
from config import (
    TELEGRAM_API_ID, 
    TELEGRAM_API_HASH, 
    TELEGRAM_CHANNEL_USERNAME, 
    SESSION_NAME, 
    POSTS_DIR as OUTPUT_DIR, 
    OUTPUT_IMAGE_DIR,
    create_directories,
    validate_config,
    logger
)

# Ensure all required configurations are present
if not validate_config():
    logger.error("Missing required configuration. Please check your .env file.")
    exit(1)

# Create required directories
create_directories()

async def main():
    # Initialize client
    client = TelegramClient(SESSION_NAME, TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()

    logger.info("Bot started. Collecting posts from the channel...")

    # Get channel information
    channel = await client.get_entity(TELEGRAM_CHANNEL_USERNAME)

    # Get all messages from the channel
    async for message in client.iter_messages(channel):
        await save_message_as_md(client, message)

    logger.info(f"All posts saved to {OUTPUT_DIR}/ directory")
    await client.disconnect()

def get_telegram_media_link(channel_username, message_id):
    """
    Creates a direct link to a message with media in Telegram
    """
    return f"https://t.me/{channel_username}/{message_id}?single"

async def save_message_as_md(client, message):
    # Skip service messages
    if not message.text and not message.media:
        return

    # Create filename based on date and message ID
    date_str = message.date.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{OUTPUT_DIR}/{date_str}_msg_{message.id}.md"

    # Prepare Markdown content
    content = ""

    # Add message text
    if message.text:
        content += message.text + "\n\n"

    # Process media (photos, documents)
    if message.media:
        # Create direct link to message with media (for non-images)
        media_link = get_telegram_media_link(TELEGRAM_CHANNEL_USERNAME, message.id)

        if isinstance(message.media, MessageMediaPhoto):
            # Download photo and save locally
            image_path = f"{OUTPUT_IMAGE_DIR}/image_{message.id}.jpg"
            await client.download_media(message, image_path)
            # Relative path for markdown
            relative_path = os.path.relpath(image_path, OUTPUT_DIR)
            content += f"![Photo]({relative_path})\n\n"
            logger.info(f"Saved image: {image_path}")
        elif isinstance(message.media, MessageMediaDocument):
            doc = message.media.document
            if doc.mime_type.startswith('image/'):
                # Determine file extension from MIME type
                extension = mimetypes.guess_extension(doc.mime_type) or '.jpg'
                image_path = f"{OUTPUT_IMAGE_DIR}/image_{message.id}{extension}"
                await client.download_media(message, image_path)
                # Relative path for markdown
                relative_path = os.path.relpath(image_path, OUTPUT_DIR)
                content += f"![Image]({relative_path})\n\n"
                logger.info(f"Saved image: {image_path}")
            else:
                content += f"[Document {doc.mime_type}]({media_link})\n\n"

    # Add metadata
    content += "---\n"
    content += f"- Message ID: {message.id}\n"
    content += f"- Date: {message.date}\n"
    if message.views is not None:
        content += f"- Views: {message.views}\n"

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"Saved: {filename}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
