"""
Hugo Post Formatter - A utility for preparing blog posts for Hugo static site generator.
Processes markdown files by extracting and enhancing frontmatter, generating tags,
summaries, and formatting content according to Hugo best practices.
"""

import os
import re
import yaml
import logging
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('hugo_formatter')

# Try importing NLTK - we'll handle missing dependencies gracefully
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize

    # Check for required NLTK data
    required_packages = ['punkt', 'stopwords']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            logger.info(f"Downloading required NLTK data: {package}")
            nltk.download(package, quiet=True)

    nltk_available = True
except ImportError:
    logger.warning("NLTK not available. Some features will be limited.")
    nltk_available = False


class HugoPostFormatter:
    """
    A class to format markdown posts for Hugo static site generator.

    Attributes:
        posts_dir (str): Directory containing original posts.
        output_dir (str): Directory where processed posts will be saved.
        stopwords (set): Set of stopwords to ignore when generating tags.
        tag_categories (dict): Predefined categories for tag generation.
        emojis (list): List of emojis to use for post titles.
    """

    def __init__(self, posts_dir='posts', output_dir='processed_posts'):
        """
        Initialize the HugoPostFormatter with directories and resources.

        Args:
            posts_dir (str): Directory containing original posts.
            output_dir (str): Directory where processed posts will be saved.
        """
        self.posts_dir = posts_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load stopwords
        self.stopwords = self._load_stopwords()

        # Predefined categories for tags
        self.tag_categories = {
            'languages': ['python', 'javascript', 'typescript', 'java', 'c#', 'c++', 'go',
                         'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'elixir', 'haskell'],
            'frameworks': ['django', 'flask', 'fastapi', 'react', 'vue', 'angular', 'svelte',
                          'laravel', 'spring', 'express', 'next.js', 'nuxt', 'rails'],
            'databases': ['sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'sqlite',
                         'cassandra', 'dynamodb', 'firebase', 'supabase', 'neo4j'],
            'technologies': ['machine learning', 'ml', 'ai', 'artificial intelligence',
                            'blockchain', 'cloud', 'docker', 'kubernetes', 'devops',
                            'ci/cd', 'microservices', 'serverless'],
            'topics': ['architecture', 'design patterns', 'algorithms', 'data structures',
                      'security', 'testing', 'optimization']
        }

        # Emojis for post titles
        self.emojis = [
            "📝", "💡", "🚀", "⚡", "🔥", "✨", "🔍", "🛠️", "📊", "📈", "🧩", "🧠",
            "🌐", "💻", "⚙️", "📚", "🤖", "🔐", "🧪", "🌟", "💾", "🔧", "📱", "🧲"
        ]

    def _load_stopwords(self):
        """
        Load stopwords from NLTK if available, otherwise use a basic set.

        Returns:
            set: Set of stopwords to ignore when processing text.
        """
        if nltk_available:
            try:
                # Combine English and other stopwords
                return set(stopwords.words('english'))
            except:
                logger.warning("Failed to load NLTK stopwords, using basic set.")

        # Basic set of stopwords if NLTK is not available
        return {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on',
                'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of', 'from'}

    def extract_frontmatter(self, content):
        """
        Extract YAML frontmatter and content from a post.

        Args:
            content (str): The full content of the post.

        Returns:
            tuple: A tuple containing the frontmatter dict and the post content.
        """
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                yaml_text = parts[1].strip()
                post_content = parts[2].strip()
                try:
                    frontmatter = yaml.safe_load(yaml_text)
                    return frontmatter, post_content
                except yaml.YAMLError as e:
                    logger.warning(f"Failed to parse YAML frontmatter: {e}")

        # If frontmatter is missing or has incorrect format
        return {}, content.strip()

    def extract_title(self, content):
        """
        Extract title from post content.

        Args:
            content (str): The content of the post.

        Returns:
            str or None: Extracted title or None if not found.
        """
        # Look for Markdown headings
        title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        if title_match:
            # Clean emojis and markdown formatting
            title = title_match.group(1).strip()
            title = re.sub(r'[😀-🙏]|\*\*|\*|`|_', '', title).strip()
            return title

        # If no Markdown heading found, use the first line
        lines = content.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Remove any Markdown formatting from the first line
            first_line = re.sub(r'#|\*\*|\*|`|_', '', first_line).strip()
            return first_line

        return None

    def extract_tags(self, content, title):
        """
        Extract relevant tags from post content.

        Args:
            content (str): The content of the post.
            title (str): The title of the post.

        Returns:
            list: A list of extracted tags.
        """
        # Clean text from markdown and punctuation
        clean_text = re.sub(r'#|\*\*|\*|`|_|\[|\]|\(|\)|[,.!?;:]', ' ', content.lower())

        if nltk_available:
            # Tokenize text
            words = word_tokenize(clean_text)
        else:
            # Simple word splitting if NLTK is not available
            words = clean_text.split()

        # Remove stopwords and short words
        words = [word for word in words if word not in self.stopwords and len(word) > 3]

        # Find most frequent words
        word_freq = Counter(words)

        # Search for words from predefined categories
        found_tags = []
        for category, keywords in self.tag_categories.items():
            for keyword in keywords:
                if keyword.lower() in clean_text:
                    found_tags.append(keyword.capitalize())
                    break

        # Add most frequent words as tags (if they appear at least 3 times)
        for word, count in word_freq.most_common(5):
            if count >= 3 and word not in [tag.lower() for tag in found_tags]:
                found_tags.append(word.capitalize())

        # Add "Article" as a default tag if no other tags found
        if not found_tags:
            found_tags = ['Article']

        # Add "Book" tag if books are mentioned in the title or text
        if 'book' in clean_text:
            found_tags.append('Books')

        # Remove duplicates and limit number of tags
        return list(set(found_tags))[:5]

    def generate_summary(self, content):
        """
        Generate a short summary from post content.

        Args:
            content (str): The content of the post.

        Returns:
            str: A summary of the post.
        """
        # Remove Markdown formatting
        clean_text = re.sub(r'#|\*\*|\*|`|_', '', content)
        clean_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_text)  # Replace links

        if nltk_available:
            try:
                # Tokenize text into sentences
                sentences = sent_tokenize(clean_text)

                # Take first 1-2 sentences depending on length
                if sentences:
                    if len(sentences[0]) < 80 and len(sentences) > 1:
                        summary = ' '.join(sentences[:2])
                    else:
                        summary = sentences[0]

                    # Truncate too long summaries
                    if len(summary) > 200:
                        summary = summary[:197] + '...'

                    return summary
            except Exception as e:
                logger.warning(f"Failed to generate summary using NLTK: {e}")

        # Fallback option if sentence tokenization failed
        words = clean_text.split()
        if not words:
            return "Article about technology and programming"

        summary = ' '.join(words[:20]) + ('...' if len(words) > 20 else '')
        return summary

    def get_random_emoji(self):
        """
        Get a random emoji for post titles.

        Returns:
            str: A random emoji.
        """
        return random.choice(self.emojis)

    def calculate_reading_time(self, content):
        """
        Calculate approximate reading time in minutes.

        Args:
            content (str): The content of the post.

        Returns:
            int: Estimated reading time in minutes.
        """
        words = len(re.findall(r'\w+', content))
        reading_time = max(1, round(words / 200))  # Average reading speed: 200 words per minute
        return reading_time

    def format_content(self, content, title):
        """
        Format content in Hugo style with emoji and markup.

        Args:
            content (str): The content of the post.
            title (str): The title of the post.

        Returns:
            str: Formatted content.
        """
        # Find all images in the text
        image_matches = list(re.finditer(r'!\[(.*?)\]\((.*?)\)', content))

        # Remove images from text for further processing
        if image_matches:
            for match in reversed(image_matches):
                content = content[:match.start()] + content[match.end():]

        # Clean content from existing first level headings
        content = re.sub(r'^#\s+.*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Remove extra empty lines

        # Add emoji to title
        emoji = self.get_random_emoji()
        main_title = f"# {emoji} **{title}**"

        # Collect images at the beginning of the document
        image_block = ""
        if image_matches:
            image_block = "\n\n"
            for match in image_matches:
                alt_text = match.group(1)
                image_url = match.group(2)
                image_block += f"![{alt_text}]({image_url})\n"

        # Add reading time information
        reading_time = self.calculate_reading_time(content)
        reading_info = f"Reading time: {reading_time} min\n"

        # Check if there's a source link in the text
        link_match = re.search(r'(?:Link|Source):\s*\[(.*?)\]\((.*?)\)', content, re.IGNORECASE)
        link_block = ""
        if link_match:
            link_text = link_match.group(1)
            link_url = link_match.group(2)
            link_block = f"\nSource: [{link_text}]({link_url})\n"
            # Remove the found link from the text
            content = content[:link_match.start()] + content[link_match.end():]

        # Structure level 2 headings
        sections = re.findall(r'^##\s*(.*?)$', content, re.MULTILINE)
        if not sections:
            # If no level 2 headings, add standard sections
            sections_block = "\n\n## 📌 Key Points\n\n"
        else:
            # If there are level 2 headings, don't change them
            sections_block = ""

        # Put everything together
        formatted_content = f"{image_block}{main_title}\n\n{reading_info}{link_block}{sections_block}{content.strip()}"

        # Clean up double line breaks
        formatted_content = re.sub(r'\n\s*\n\s*\n', '\n\n', formatted_content)

        return formatted_content

    def format_post(self, filename):
        """
        Format a single post in Hugo format.

        Args:
            filename (str): The filename of the post to format.

        Returns:
            str: Path to the output file.
        """
        file_path = os.path.join(self.posts_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Unicode decode error with {filename}, trying alternative encoding")
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            raise

        # Extract frontmatter and content
        frontmatter, post_content = self.extract_frontmatter(content)

        # Extract title
        title = frontmatter.get('title') or self.extract_title(post_content) or Path(filename).stem
        if 'title' not in frontmatter:
            frontmatter['title'] = title

        # Extract or generate tags
        if 'tags' not in frontmatter or not frontmatter['tags']:
            frontmatter['tags'] = self.extract_tags(post_content, title)

        # Generate summary if missing
        if 'summary' not in frontmatter or not frontmatter['summary']:
            frontmatter['summary'] = self.generate_summary(post_content)

        # Add date if missing
        if 'date' not in frontmatter:
            frontmatter['dates'] = datetime.now().strftime('%Y-%m-%d')

        # Set draft: false if not specified
        if 'draft' not in frontmatter:
            frontmatter['draft'] = False

        # Format content in Hugo style
        formatted_content = self.format_content(post_content, title)



        # Create new content with frontmatter
        formatted_post = f"---\n{yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)}---\n\n{formatted_content}"

        # Save result
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_post)

        return output_path

    def process_all_posts(self):
        """
        Process all markdown files in the posts directory.

        Returns:
            list: List of processed file paths.
        """
        post_files = [f for f in os.listdir(self.posts_dir) if f.endswith('.md')]

        if not post_files:
            logger.warning(f"No markdown files found in directory {self.posts_dir}.")
            return []

        processed_files = []
        for filename in post_files:
            try:
                output_path = self.format_post(filename)
                processed_files.append(output_path)
                logger.info(f"Processed file: {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")

        return processed_files


def main():
    """Main function to run the Hugo Post Formatter."""
    parser = argparse.ArgumentParser(description='Format markdown posts for Hugo.')
    parser.add_argument('--input', '-i', default='posts',
                        help='Directory containing markdown files (default: posts)')
    parser.add_argument('--output', '-o', default='processed_posts',
                        help='Directory to save processed files (default: processed_posts)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    formatter = HugoPostFormatter(posts_dir=args.input, output_dir=args.output)
    logger.info("Starting Hugo post processing...")

    processed = formatter.process_all_posts()
    logger.info(f"Done! Processed {len(processed)} files.")
    logger.info(f"Results saved to directory: {os.path.abspath(formatter.output_dir)}")


if __name__ == '__main__':
    main()
