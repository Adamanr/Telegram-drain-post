# Telegram to Hugo Blog Post Processor

A comprehensive tool for fetching posts from Telegram channels, converting them to Markdown format, and processing them with AI to create Hugo-compatible blog posts.

## 📋 Overview

This project automates the process of creating blog content from Telegram posts:

1. **Fetches posts** from a specified Telegram channel
2. **Converts them** to Markdown format with proper formatting
3. **Processes them through AI** to enhance the content and structure
4. **Prepares them** for Hugo blog publication with appropriate frontmatter

## 🚀 Features

- **Telegram Post Extraction**: Download posts, images, and attachments from Telegram channels
- **AI-Powered Enhancements**: Transform raw posts into well-structured blog content
- **Hugo Integration**: Generate proper frontmatter with tags, summary, and metadata
- **Batch Processing**: Efficient processing of multiple posts with configurable parallelism
- **Image Handling**: Proper extraction and reference of images in Hugo format

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- [Telegram API credentials](https://my.telegram.org/apps)
- [Hugging Face account](https://huggingface.co/) and access token
- Hugo blog (optional, for final deployment)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telegram-to-hugo.git
   cd telegram-to-hugo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your credentials
   ```

## ⚙️ Configuration

Create a `.env` file based on the provided `.env.example` template with the following values:

- `TELEGRAM_API_ID` - Your Telegram API ID
- `TELEGRAM_API_HASH` - Your Telegram API Hash
- `TELEGRAM_CHANNEL_USERNAME` - Telegram channel name to extract posts from
- `HF_TOKEN` - Hugging Face API token
- `MODEL_NAME` - AI model to use for processing (default: mistralai/Mistral-7B-Instruct-v0.1)
- Directory settings for post storage and processing

## 📝 Usage

### Fetching Posts from Telegram

```bash
python bot.py
```

This will download posts from the configured Telegram channel and save them as Markdown files.

### Processing Posts with AI

```bash
python ai_conver.py
```

This processes the downloaded posts through an AI model to enhance and structure them for Hugo.

### Alternative Processing Method

```bash
python hugo_convert.py
```

A simpler post processing utility with fewer AI enhancements.

### Hugo Format Conversion

```bash
python hugo_convert.py
```

Ensures posts are properly formatted according to Hugo standards.

## 🧩 Project Components

### bot.py

Connects to Telegram API and downloads posts, converting them to initial Markdown format.

### ai_conver.py

The main AI processing component that enhances post content, generates tags, summaries, and ensures Hugo-compatible formatting.

### hugo_convert.py

Ensures posts follow Hugo conventions with proper frontmatter and formatting.

### config.py

Centralized configuration management with environment variable support.

## 🔄 Workflow

1. **Extract**: Bot downloads Telegram posts to `posts/` directory
2. **Process**: AI converter enhances the content and structure
3. **Convert**: Posts are formatted for Hugo with proper frontmatter
4. **Publish**: Final posts are ready for Hugo blog in `OUTPUT_DIR`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
