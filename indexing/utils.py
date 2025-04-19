import hashlib
from pathlib import Path
import os
import tempfile
import time
from typing import List

# Ensure pdf2image and poppler are installed
# pip install pdf2image
# On Debian/Ubuntu: sudo apt-get install poppler-utils
# On macOS: brew install poppler
try:
    from pdf2image import convert_from_path
except ImportError:
    print("Warning: pdf2image not found. PDF processing will not work.")
    print("Install it via 'pip install pdf2image'")
    print("You also need poppler installed system-wide.")
    convert_from_path = None

# Ensure vllm, torch, and Pillow are installed
# pip install vllm torch Pillow
try:
    from vllm import LLM, SamplingParams
    from PIL import Image
except ImportError:
    print("Warning: vllm, torch, or Pillow not found. Docling extraction will not work.")
    print("Install them via 'pip install vllm torch Pillow'")
    LLM = None
    SamplingParams = None
    Image = None


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of file"""
    print(f"Calculating hash for {file_path}")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_docling_from_pdf(
    pdf_path: Path, model_path: str = "ds4sd/SmolDocling-256M-preview"
) -> str:
    """
    Extracts Docling format text from a PDF file using the SmolDocling model.

    Converts the PDF to temporary images, runs SmolDocling on each image,
    and concatenates the resulting Docling text.

    Args:
        pdf_path: Path to the input PDF file.
        model_path: Path or identifier of the SmolDocling model to use.

    Returns:
        A string containing the concatenated Docling text for all pages.

    Raises:
        FileNotFoundError: If the pdf_path does not exist.
        ImportError: If required libraries (pdf2image, vllm, Pillow) are not installed.
        RuntimeError: If PDF conversion fails or model processing fails.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if convert_from_path is None:
        raise ImportError("pdf2image is required but not installed or poppler is missing.")
    if LLM is None or SamplingParams is None or Image is None:
        raise ImportError("vllm, torch, and Pillow are required but not installed.")

    print(f"Processing PDF: {pdf_path}")
    start_time = time.time()

    # Initialize LLM (consider initializing outside if calling frequently)
    print(f"Initializing LLM model: {model_path}...")
    # Make sure GPU memory is sufficient, this model can be large
    llm = LLM(model=model_path, limit_mm_per_prompt={"image": 1}, enforce_eager=True) # Added enforce_eager for potential stability
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    chat_template = "<|im_start|>User:<image>Convert page to Docling.<end_of_utterance>\nAssistant:"
    print("LLM initialized.")

    all_doctags: List[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Converting PDF to images in temporary directory: {temp_path}")
        try:
            # Use thread_count=1 for potentially more stable conversion with complex PDFs
            images = convert_from_path(pdf_path, output_folder=temp_path, fmt='png', output_file='page', thread_count=1)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")

        if not images:
            print("Warning: No images generated from PDF.")
            return ""

        print(f"Generated {len(images)} page images.")

        # Ensure images are processed in page order using the paths returned by convert_from_path
        # Sorting filenames might be unreliable if page numbers aren't zero-padded
        image_paths = [Path(img.filename) for img in images] # Get paths from image objects

        for idx, img_path in enumerate(image_paths, 1):
            print(f"Processing page {idx}/{len(image_paths)}: {img_path.name}...")
            try:
                # Open image using the path
                with Image.open(img_path) as image:
                    image = image.convert("RGB") # Ensure correct mode
                    llm_input = {"prompt": chat_template, "multi_modal_data": {"image": image}}
                    # Note: llm.generate expects a list of inputs
                    output = llm.generate([llm_input], sampling_params=sampling_params)[0]
                    doctags = output.outputs[0].text
                    all_doctags.append(doctags)
                    print(f"Page {idx} processed.")

            except Exception as e:
                print(f"Error processing page {idx} ({img_path.name}): {e}")
                # Clean up potentially opened image object if error occurs during processing
                # Consider more robust error handling or logging here
                # For now, let's skip the page and continue
                continue # Or raise RuntimeError(f"Failed processing page {idx}: {e}")
            finally:
                # Attempt to remove the temporary image file after processing or error
                try:
                    img_path.unlink()
                except OSError as unlink_error:
                    print(f"Warning: Could not remove temporary image {img_path}: {unlink_error}")


    concatenated_doctags = "\n\n".join(all_doctags) # Join with double newline as page separator

    end_time = time.time()
    print(f"Finished processing {pdf_path}. Total time: {end_time - start_time:.2f} sec")

    # Explicitly clean up LLM object if possible/needed, depends on vLLM specifics
    # del llm
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    return concatenated_doctags

