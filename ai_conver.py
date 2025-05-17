import os
import time
import re
import yaml
import random
import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
from concurrent.futures import ThreadPoolExecutor
import gc

# Import configuration
from config import (
    HF_TOKEN,
    MODEL_NAME,
    POSTS_DIR,
    HUGO_CONTENT_DIR as OUTPUT_DIR,
    REQUEST_FILE,
    MAX_WORKERS,
    logger,
    create_directories,
    validate_config
)

# Common tags for post generation
COMMON_TAGS = [
    'Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'C#', 'Java', 'Kotlin',
    'Machine Learning', 'Data Science', 'Article', 'Book', 'Review', 'Tutorial',
    'Backend', 'Frontend', 'DevOps', 'Cloud', 'PostgreSQL', 'MongoDB', 'Redis',
    'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Algorithms', 'Architecture',
    'Elixir', 'React', 'Vue', 'Angular', 'Django', 'Flask', 'FastAPI', 'Spring',
    'Linux', 'Windows', 'MacOS', 'iOS', 'Android', 'Microservices', 'Git'
]

def clear_memory():
    """Aggressively clean memory"""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Try to forcibly free CUDA memory
        try:
            import ctypes
            cudart = ctypes.CDLL('libcudart.so')
            cudart.cudaDeviceReset()
        except:
            pass

def setup_device():
    """Set up the optimal device based on available resources"""
    device = torch.device("cpu")  # Default to CPU

    if torch.cuda.is_available():
        try:
            # Get device with most free memory
            free_memory = 0
            selected_device = 0

            for i in range(torch.cuda.device_count()):
                current_free = torch.cuda.mem_get_info(i)[0] / (1024**3)  # Convert to GB
                if current_free > free_memory:
                    free_memory = current_free
                    selected_device = i

            if free_memory >= 2.0:  # Only use GPU if at least 2GB free
                device = torch.device(f"cuda:{selected_device}")
                torch.cuda.set_device(device)
                logger.info(f"Using GPU {selected_device} with {free_memory:.2f}GB free memory")
            else:
                logger.info(f"Insufficient GPU memory ({free_memory:.2f}GB). Using CPU")
        except Exception as e:
            logger.error(f"Error initializing CUDA: {e}")
            device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device

def load_model(device):
    """Load the model with appropriate memory optimizations"""
    logger.info(f"⚡ Loading model {MODEL_NAME}...")
    clear_memory()

    # Check available GPU memory if using CUDA
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device.index).total_memory / (1024**3)
        free_memory = torch.cuda.mem_get_info(device.index)[0] / (1024**3)
        logger.info(f"Total GPU memory: {total_memory:.2f}GB, free: {free_memory:.2f}GB")

        # Use only 30% of free memory to be conservative
        safe_limit = max(1, int(free_memory * 0.3))
        torch.cuda.set_per_process_memory_fraction(0.3, device.index)
        logger.info(f"Setting GPU memory limit to 30% ({safe_limit}GB of {free_memory:.1f}GB)")

        # Memory config for model loading
        max_memory = {device.index: f"{safe_limit}GiB", "cpu": "16GiB"}

        # Configure quantization if enough memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8,
        )

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            trust_remote_code=True,
            max_memory=max_memory
        )
    else:
        # CPU-only loading
        logger.info("Loading model on CPU only - generation will be slow")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    # Optimize with torch.compile if available
    if hasattr(torch, "compile"):
        try:
            logger.info("Applying torch.compile() optimization...")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile unavailable: {e}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_tokens=500, device=None):
    """Generate text with the model"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Determine model device
        model_device = next(model.parameters()).device

        # Tokenize and move to correct device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except RuntimeError as e:
        logger.error(f"Generation error: {e}")
        return ""

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def extract_yaml_and_content(original_content):
    """Extract YAML frontmatter and post content"""
    yaml_match = re.match(r'^---\n(.*?)\n---\n', original_content, re.DOTALL)

    if yaml_match:
        yaml_text = yaml_match.group(1)
        yaml_data = yaml.safe_load(yaml_text) or {}
        content = original_content[len(yaml_match.group(0)):]
        return yaml_data, content

    return {}, original_content

def guess_tags_from_content(content, existing_tags=None):
    """Generate appropriate tags for the content"""
    if existing_tags and isinstance(existing_tags, list) and len(existing_tags) >= 3:
        return existing_tags

    relevant_tags = []
    if existing_tags and isinstance(existing_tags, list):
        relevant_tags.extend(existing_tags)

    # Extract tags from common patterns in the content
    content_lower = content.lower()

    # Add tags based on content keywords
    for tag in COMMON_TAGS:
        if tag.lower() in content_lower and tag not in relevant_tags:
            relevant_tags.append(tag)

    # Add random popular tags if needed
    while len(relevant_tags) < 3:
        random_tag = random.choice(COMMON_TAGS)
        if random_tag not in relevant_tags:
            relevant_tags.append(random_tag)

    return relevant_tags[:5]  # No more than 5 tags

def generate_summary(model, tokenizer, content, device):
    """Generate a brief summary of the article content"""
    prompt = f"""
    Read this article and write a short informative summary in one sentence (up to 15 words).

    Article:
    {content[:3000]}  <!-- Using only the beginning to save tokens -->

    Only generate the summary text without additional comments.
    """

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        summary = generate_text(model, tokenizer, prompt, max_tokens=100, device=device).strip()
        # Clean up possible artifacts
        summary = re.sub(r'Summary:', '', summary, flags=re.IGNORECASE).strip()
        return summary[:150]  # Limit length

    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "An interesting article about technology and development"

def fix_bold_text(text):
    """Fix issues with bold text formatting"""
    if not text:
        return text

    # Step 1: Fix various forms of broken bold text
    # Fix cases like "**word1 **word2" -> "**word1 word2**"
    text = re.sub(r'\*\*([^\*\n]+?)\s+\*\*([^\*\n]+?)\*\*', r'**\1 \2**', text)

    # Fix cases like "word1** **word2" -> "word1 **word2**"
    text = re.sub(r'([^\*\n]+?)\*\*\s+\*\*([^\*\n]+?)', r'\1 **\2', text)

    # Step 2: Fix incorrect spacing
    text = re.sub(r'([^\s])\*\*', r'\1 **', text)  # Add space before **
    text = re.sub(r'\*\*([^\s])', r'** \1', text)  # Add space after **

    # Step 3: Process lines for more complex cases
    lines = []
    open_bold_tag = False  # Flag to track open bold tags

    for line in text.split('\n'):
        # Check if there are ** markers in the line
        bold_markers = line.count('**')

        # If odd number of markers in the line
        if bold_markers % 2 != 0:
            if line.startswith('**') and not open_bold_tag:
                # New bold tag opened
                open_bold_tag = True
            elif line.endswith('**') and open_bold_tag:
                # Existing bold tag closed
                open_bold_tag = False
            elif '**' in line:
                # Marker in the middle - check if tag is open or closed
                open_bold_tag = not open_bold_tag
            else:
                # If no markers but tag is open, close it
                if open_bold_tag:
                    line += '**'
                    open_bold_tag = False

        lines.append(line)

    # Check if there's an unclosed tag at the end
    if open_bold_tag:
        lines[-1] += '**'

    text = '\n'.join(lines)

    # Step 4: Final cleanup
    text = re.sub(r'\*\*\s+\*\*', '', text)  # Remove empty bold sections
    text = re.sub(r'\s+\*\*\s+', ' **', text)  # Normalize spaces around markers
    text = re.sub(r'\*\*\s+([^\*]+?)\s+\*\*', r'**\1**', text)  # Remove extra spaces inside

    # Remove multiple spaces within text
    text = re.sub(r'(\S)\s{2,}(\S)', r'\1 \2', text)

    return text

def fix_headers(text):
    """Fix header formatting to ensure proper top-down hierarchy"""
    if not text:
        return text

    lines = text.split('\n')
    result = []
    prev_level = 0
    first_header = True

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            # Count number of # at start
            level = len(re.match(r'^#+', stripped).group())

            # For first header always use a single #
            if first_header:
                level = 1
                first_header = False
            # For subsequent headers ensure proper hierarchy
            elif level > prev_level + 1:
                level = prev_level + 1

            # Extract header text
            header_text = stripped.lstrip('#').strip()

            # Format header
            formatted_header = '#' * level + ' ' + header_text

            # Add empty line before header
            if i > 0 and result and result[-1].strip():
                result.append('')

            result.append(formatted_header)

            # Add empty line after header
            if i < len(lines) - 1 and lines[i + 1].strip():
                result.append('')

            prev_level = level
        else:
            result.append(line)

    # Remove multiple empty lines
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(result))
    return text

def fix_image_paths(text):
    """Convert relative image paths to absolute URLs for Hugo"""
    if not text:
        return text

    def process_image_match(match):
        alt_text = match.group(1)
        original_path = match.group(2).strip()

        # If path is already in correct format - leave as is
        if original_path.startswith('http://localhost:1313/'):
            return match.group(0)

        # Extract filename from any path type
        filename = os.path.basename(original_path)

        # Remove possible prefixes like "../../blog/static/"
        filename = filename.replace('../../blog/static/', '').replace('./', '')

        # Form new URL (note /static/ in path)
        new_url = f"http://localhost:1313/images/posts/{filename}"

        return f"![{alt_text}]({new_url})"

    # Find all images in text, including multi-line cases
    return re.sub(
        r'!\[([^\]]*)\]\(([^)]*)\)',
        process_image_match,
        text,
        flags=re.MULTILINE
    )

def preserve_yaml_frontmatter(content):
    """Preserve YAML frontmatter at the beginning of the document"""
    # Check for YAML frontmatter (---...---)
    yaml_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(0)
        main_content = content[len(yaml_content):]
        return yaml_content, main_content
    return None, content

def enhance_post_format(model, tokenizer, content, device):
    """Enhance post formatting using AI"""
    logger.info("AI-based formatting enhancement")

    # Save YAML frontmatter if present
    yaml_frontmatter, content_without_yaml = preserve_yaml_frontmatter(content)

    # Pre-process content to fix formatting
    preprocessed_content = content_without_yaml
    preprocessed_content = fix_headers(preprocessed_content)  # First fix headers
    preprocessed_content = fix_bold_text(preprocessed_content)  # Then fix bold text
    preprocessed_content = fix_image_paths(preprocessed_content)  # Finally fix image paths

    # If there's YAML frontmatter, add it back
    if yaml_frontmatter:
        preprocessed_content = yaml_frontmatter + preprocessed_content

    prompt = f"""
    You are a Hugo blog editor.
    Improve this article: preserve all facts and structure, make the style more expressive and interesting.

    Important formatting rules:
    1. Preserve all headers (#, ##, ###) and their hierarchy
    2. Preserve all bold formatting (**text**)
    3. Don't change image paths in ![Text](url) format
    4. Use empty lines between sections for better readability
    5. Use Markdown format with headers, lists, and emojis, but don't overuse emojis
    6. Write "Aqua" at the end of the article

    Don't invent anything new, work only with the existing content.
    Add a rating for book/technology (if it's a review), reading time, key points. Take the date from the date field below ---.

    Here's the article to edit:

    {preprocessed_content}
    """

    try:
        enhanced_content = generate_text(model, tokenizer, prompt, max_tokens=800, device=device).strip()
        # Remove possible prompt artifacts
        enhanced_content = re.sub(r'Here\'s the improved article:|Improved article:', '', enhanced_content, flags=re.IGNORECASE).strip()

        # Post-processing to ensure correct formatting
        enhanced_content = fix_bold_text(enhanced_content)
        enhanced_content = fix_headers(enhanced_content)
        enhanced_content = fix_image_paths(enhanced_content)

        return enhanced_content
    except Exception as e:
        logger.error(f"Error enhancing formatting: {e}")
        return content

def extract_title_from_content(content):
    """Extract title from content if not specified in YAML"""
    # Look for title in markdown format
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()

    # Look for first line with text
    lines = content.strip().split('\n')
    for line in lines:
        if line.strip() and not line.startswith('!') and not line.startswith('['):
            return line.strip()[:60]  # Limit length

    return "New Article"  # If title not found

def process_post(filename, model, tokenizer, device):
    """Process a single post with image handling and date transfer"""
    try:
        start_time = time.time()
        file_path = os.path.join(POSTS_DIR, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        yaml_data, post_content = extract_yaml_and_content(original_content)

        if len(post_content.split()) < 50:
            logger.warning(f"{filename} is too short, skipping")
            return "skipped"

        if not yaml_data.get('title'):
            yaml_data['title'] = extract_title_from_content(post_content).replace('*', '')

        image_match = re.search(r'!\[.*?\]\((.*?)\)', post_content)
        image_tag = ""
        if image_match:
            image_path = image_match.group(1)
            image_tag = fix_image_paths(f"![Image]({image_path})\n\n")
            post_content = post_content.replace(image_match.group(0), "").strip()

        date_match = re.search(r'- Date: \s*([0-9\-:+\s]+)', post_content)
        if date_match:
            raw_date = date_match.group(1).strip()
            try:
                date_obj = datetime.datetime.strptime(raw_date.split('+')[0].strip(), '%Y-%m-%d %H:%M:%S')
                yaml_data['date'] = date_obj.strftime('%Y-%m-%d')
                post_content = re.sub(r'---\n- Message ID:.*?Views: \d+', '', post_content, flags=re.DOTALL).strip()
            except Exception as e:
                logger.error(f"Error processing date: {e}")
                if not yaml_data.get('date'):
                    yaml_data['date'] = time.strftime('%Y-%m-%d')

        if not yaml_data.get('date'):
            yaml_data['date'] = time.strftime('%Y-%m-%d')

        yaml_data['tags'] = guess_tags_from_content(post_content, yaml_data.get('tags'))

        if not yaml_data.get('summary'):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yaml_data['summary'] = generate_summary(model, tokenizer, post_content, device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        yaml_data['draft'] = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        enhanced_content = enhance_post_format(model, tokenizer, post_content, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create new frontmatter
        new_frontmatter = yaml.dump(yaml_data, allow_unicode=True, default_flow_style=False)

        # Create final post (add image right after frontmatter)
        final_post = f"---\n{new_frontmatter}---\n\n{image_tag}{enhanced_content}"

        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_post)

        elapsed_time = time.time() - start_time
        logger.info(f"{filename} processed in {elapsed_time:.2f} seconds")
        return "success"

    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return f"error: {str(e)}"

def safe_process_post(filename, model, tokenizer, device):
    """Safely process post with memory error handling"""
    try:
        return process_post(filename, model, tokenizer, device)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "memory_error"

def warmup_model(model, tokenizer):
    """Warm up the model for more stable operation"""
    logger.info("🔥 Starting model warmup...")

    # Aggressive memory cleanup before warmup
    clear_memory()

    try:
        # Check that the model is already correctly distributed across devices
        model_device = next(model.parameters()).device
        logger.info(f"Model device: {model_device}")

        # Very small test input
        input_str = "Test"
        inputs = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=4)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # Test run with minimal parameters
        with torch.no_grad():
            try:
                logger.info(f"Warming up on device: {model_device}")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                logger.info("✅ Warmup successful")
                return model_device

            except Exception as e:
                logger.warning(f"Error during warmup: {e}")
                if "device" in str(e).lower() or "memory" in str(e).lower():
                    logger.info("Trying on CPU...")
                    # Move the entire model to CPU
                    model.to("cpu")
                    clear_memory()

                    # New run on CPU
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    logger.info("✅ Warmup on CPU successful")
                    return torch.device("cpu")
                else:
                    raise

    except Exception as e:
        logger.warning(f"Error warming up model: {e}")
        # In case of any error switch to CPU
        model.to("cpu")
        return torch.device("cpu")

    finally:
        clear_memory()

def process_all_posts(model, tokenizer, device):
    """Process all posts in the directory"""
    post_files = [f for f in os.listdir(POSTS_DIR) if f.endswith(('.md', '.markdown'))]

    if not post_files:
        logger.error(f"No Markdown files found in {POSTS_DIR}!")
        return

    logger.info(f"Found {len(post_files)} files to process")

    workers = min(MAX_WORKERS, len(post_files))
    logger.info(f"Starting processing with {workers} workers")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for filename in post_files:
            futures.append(executor.submit(safe_process_post, filename, model, tokenizer, device))

        with tqdm(total=len(futures), desc="Processing posts") as pbar:
            for future in futures:
                try:
                    future.result()
                except torch.cuda.OutOfMemoryError:
                    logger.warning("Not enough memory for processing, skipping file")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                pbar.update(1)

def main():
    """Main function to run the Hugo Blog Enhancer"""
    logger.info("🚀 Hugo Blog Enhancer")
    logger.info("=" * 50)

    # Ensure configuration is valid
    if not validate_config():
        logger.error("Missing required configuration. Please check your .env file.")
        exit(1)

    # Log in to Hugging Face
    try:
        login(token=HF_TOKEN)
        logger.info("Successfully logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face: {e}")
        exit(1)

    # Create necessary directories
    create_directories()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory ensured: {OUTPUT_DIR}")

    # Set up optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    # Setup device
    device = setup_device()

    # Load model and tokenizer
    model, tokenizer = load_model(device)

    # Warm up model
    logger.info("⚡ Warming up model...")
    warmup_model(model, tokenizer)

    # Time execution
    start_time = time.time()

    # Process all posts
    process_all_posts(model, tokenizer, device)

    # Log total time
    elapsed = time.time() - start_time
    logger.info(f"✅ Processing completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

if __name__ == '__main__':
    main()
