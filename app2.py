import streamlit as st
import os
import base64
import io
import unicodedata
import tempfile
import numpy as np
from PIL import Image
import requests
from pathlib import Path
import subprocess
import cv2
import google.generativeai as genai
from fpdf import FPDF
import sys
import platform

# Set page configuration
st.set_page_config(page_title="Document Text Extractor and Summarizer", layout="wide")

# Function to check if a command is available
def is_command_available(command):
    try:
        if platform.system() == "Windows":
            # On Windows, use where to check if command exists
            subprocess.check_output(f"where {command}", shell=True)
        else:
            # On Unix-like systems, use which
            subprocess.check_output(["which", command])
        return True
    except subprocess.CalledProcessError:
        return False

# Essential imports check - we need these to continue
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    st.error("pytesseract package is missing. Please install it with: pip install pytesseract")
    st.error("Also ensure Tesseract OCR is installed on your system.")
    TESSERACT_AVAILABLE = False

# Check if tesseract is installed
if TESSERACT_AVAILABLE and not is_command_available("tesseract"):
    TESSERACT_AVAILABLE = False

# Configure Tesseract path - critically important for Windows users
if TESSERACT_AVAILABLE and os.name == 'nt':  # Windows
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ]
    
    found_tesseract = False
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            found_tesseract = True
            break
    
    if not found_tesseract:
        st.warning("Could not find Tesseract OCR automatically. If OCR fails, please install Tesseract or set the path manually.")

# Check for PDF processing capabilities
PDF_SUPPORT = False
try:
    import pdf2image
    # Check if poppler is installed
    if platform.system() == "Windows":
        # On Windows, we need to check differently as poppler binaries might be in different locations
        PDF_SUPPORT = True  # pdf2image will report more specific errors
    else:
        # For Linux/Mac
        PDF_SUPPORT = is_command_available("pdftoppm") and is_command_available("pdfinfo")
        if not PDF_SUPPORT:
            st.warning("Poppler utilities (pdftoppm, pdfinfo) not found. PDF processing might fail.")
            st.info("Installation instructions:")
            st.code("Ubuntu: sudo apt-get install -y poppler-utils")
            st.code("Mac: brew install poppler")
            st.code("Windows: conda install -c conda-forge poppler")
except ImportError:
    st.warning("pdf2image package is missing. PDF processing will not be available.")
    st.info("Install with: pip install pdf2image")
    PDF_SUPPORT = False

# Check for docx support
DOCX_SUPPORT = False
try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    st.warning("python-docx package is missing. Word document processing will not be available.")
    st.info("Install with: pip install python-docx")

# Check for pptx support
PPTX_SUPPORT = False
try:
    from pptx import Presentation
    PPTX_SUPPORT = True
except ImportError:
    st.warning("python-pptx package is missing. PowerPoint processing will not be available.")
    st.info("Install with: pip install python-pptx")

# Initialize Google Gemini AI for summarization
GEMINI_AVAILABLE = False
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or "AIzaSyAdl3L-bN1HmBKExX9--dp2FzHKWE9vNns")
    
    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    
    GEMINI_AVAILABLE = True
except Exception as e:
    st.sidebar.warning(f"Gemini API not properly configured: {str(e)}")
    st.sidebar.info("Set your Gemini API key as an environment variable: GEMINI_API_KEY")
    st.sidebar.info("Summary generation will use basic text extraction only.")

# Display sidebar with instructions
with st.sidebar:
    st.title("Instructions")
    st.markdown("""
    ## How to use
    1. Upload a document (image, PDF, PPT, Word doc)
    2. Click "Process Document" to extract text
    3. Edit the extracted text if needed
    4. Click "Generate Summary" to create a summary
    5. Export to PDF if desired
    
    ## Supported File Types
    - Images: JPG, PNG, BMP, TIFF
    - Documents: PDF, DOCX, DOC, PPTX, PPT
    
    ## Required Software
    - Tesseract OCR (for image and PDF text extraction)
    """)
    
    with st.expander("System Status"):
        st.write("**OCR System Status:**")
        st.write(f"- Tesseract OCR: {'✅ Available' if TESSERACT_AVAILABLE else '❌ Not available'}")
        st.write(f"- PDF Processing: {'✅ Available' if PDF_SUPPORT else '❌ Not available'}")
        st.write(f"- Word (DOCX): {'✅ Available' if DOCX_SUPPORT else '❌ Not available'}")
        st.write(f"- PowerPoint (PPTX): {'✅ Available' if PPTX_SUPPORT else '❌ Not available'}")
        st.write(f"- Gemini AI: {'✅ Available' if GEMINI_AVAILABLE else '❌ Not available'}")
        
        if not TESSERACT_AVAILABLE:
            st.markdown("""
            ### Tesseract Installation
            - Windows: https://github.com/UB-Mannheim/tesseract/wiki
            - Mac: `brew install tesseract`
            - Ubuntu: `sudo apt install tesseract-ocr`
            """)
            
        if not PDF_SUPPORT:
            st.markdown("""
            ### Poppler Installation
            - Ubuntu: `sudo apt-get install -y poppler-utils`
            - Mac: `brew install poppler`
            - Windows: `conda install -c conda-forge poppler`
            """)

# Utility Functions for processing documents

def preprocess_image_for_ocr(img):
    """Process image to improve OCR results"""
    try:
        # Convert PIL Image to OpenCV format
        img_cv = np.array(img)
        
        # Check if image is RGB and convert BGR for OpenCV
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Convert back to PIL image for Tesseract
        processed = Image.fromarray(thresh)
        return processed
    except Exception as e:
        st.warning(f"Image preprocessing failed: {e}. Using original image.")
        return img

def extract_text_with_tesseract(image_path):
    """Extract text from an image using Tesseract OCR"""
    if not TESSERACT_AVAILABLE:
        return ""
        
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Try multiple preprocessing methods for better results
        extraction_methods = [
            {"method": "original", "func": lambda x: x},
            {"method": "preprocessed", "func": preprocess_image_for_ocr},
            {"method": "grayscale", "func": lambda x: x.convert('L')},
        ]
        
        best_text = ""
        best_method = ""
        
        # Try each method and keep the best result
        for method in extraction_methods:
            processed_img = method["func"](img)
            
            # Try multiple OCR configurations
            configs = [
                "--psm 1",  # Auto page segmentation with OSD
                "--psm 3",  # Fully automatic page segmentation, but no OSD
                "--psm 4",  # Assume a single column of text of variable sizes
                "--psm 6",  # Assume a single uniform block of text
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                        best_method = f"{method['method']} with {config}"
                except Exception:
                    continue
        
        if best_text.strip():
            st.info(f"Best OCR result achieved with: {best_method}")
            return best_text
        else:
            st.warning("OCR failed to extract meaningful text.")
            return ""
    except Exception as e:
        st.error(f"Tesseract OCR error: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using OCR"""
    if not PDF_SUPPORT:
        st.error("PDF processing is not available.")
        return ""
    
    try:
        # Convert PDF to images
        st.info("Converting PDF to images for OCR processing...")
        
        # Try to convert PDF to images
        try:
            images = pdf2image.convert_from_path(pdf_path)
        except Exception as e:
            error_msg = str(e).lower()
            if "poppler" in error_msg:
                return ""
            else:
                st.error(f"PDF conversion error: {str(e)}")
                return ""
        
        if not images:
            st.error("Failed to convert PDF to images.")
            return ""
        
        # Extract text from each page
        full_text = ""
        progress_bar = st.progress(0)
        st.info(f"Processing {len(images)} pages with Tesseract OCR...")
        
        for i, image in enumerate(images):
            # Save image temporarily
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            image_path = temp_img.name
            image.save(image_path, 'PNG')
            temp_img.close()
            
            # Extract text
            page_text = extract_text_with_tesseract(image_path)
            
            # Validate if we got any meaningful text
            if len(page_text.strip()) < 5:  # Arbitrary threshold for "meaningful" text
                st.warning(f"Page {i+1}: Little or no text extracted")
                
                # Try image preprocessing for this page
                if GEMINI_AVAILABLE:
                    st.info(f"Attempting Gemini AI for page {i+1}...")
                    try:
                        with open(image_path, "rb") as img_file:
                            image_bytes = img_file.read()
                        
                        prompt = "Extract all text from this image, preserving layout and formatting as much as possible."
                        response = gemini_model.generate_content([
                            prompt,
                            {"mime_type": "image/png", "data": image_bytes}
                        ])
                        gemini_text = response.text
                        if len(gemini_text.strip()) > len(page_text.strip()):
                            page_text = gemini_text
                            
                    except Exception as e:
                        st.warning(f"Gemini extraction failed: {str(e)}")
            
            full_text += f"\n\n--- Page {i+1} ---\n\n{page_text}"
            
            # Update progress
            progress_bar.progress((i + 1) / len(images))
            
            # Clean up
            os.unlink(image_path)
        
        return full_text
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from a Word document"""
    if not DOCX_SUPPORT:
        st.error("Word document processing is not available.")
        return ""
    
    try:
        doc = Document(docx_path)
        
        # Extract text from paragraphs
        text = []
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text.append(" | ".join(row_text))
        
        # Check if we got meaningful text
        combined_text = "\n\n".join(text)
        if len(combined_text.strip()) < 10:  # Arbitrary threshold
            st.warning("Little or no text extracted from Word document.")
        
        return combined_text
    except Exception as e:
        st.error(f"Word document processing error: {str(e)}")
        return ""

def extract_text_from_pptx(pptx_path):
    """Extract text from a PowerPoint presentation"""
    if not PPTX_SUPPORT:
        st.error("PowerPoint processing is not available.")
        return ""
    
    try:
        prs = Presentation(pptx_path)
        
        text = []
        for i, slide in enumerate(prs.slides):
            slide_text = [f"--- Slide {i+1} ---"]
            
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)
            
            if len(slide_text) > 1:  # Only add if there's text content
                text.append("\n".join(slide_text))
        
        # Check if we got meaningful text
        combined_text = "\n\n".join(text)
        if len(combined_text.strip()) < 10:  # Arbitrary threshold
            st.warning("Little or no text extracted from PowerPoint presentation.")
        
        return combined_text
    except Exception as e:
        st.error(f"PowerPoint processing error: {str(e)}")
        return ""

def extract_text_from_document(file_path, file_extension):
    """Extract text from various document types, with error handling and fallbacks"""
    file_extension = file_extension.lower()
    
    # For image files, directly use Tesseract OCR
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        st.info(f"Extracting text from image using Tesseract OCR...")
        text = extract_text_with_tesseract(file_path)
        
        # If OCR failed and Gemini is available, try it as a backup
        if len(text.strip()) < 10 and GEMINI_AVAILABLE:
            
            try:
                with open(file_path, "rb") as img_file:
                    image_bytes = img_file.read()
                
                prompt = "Extract all text from this image, preserving layout and formatting as much as possible."
                response = gemini_model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ])
                gemini_text = response.text
                if len(gemini_text.strip()) > len(text.strip()):
                    text = gemini_text
                    
            except Exception as e:
                st.warning(f"Gemini extraction failed: {str(e)}")
        
        return text
        
    # For PDF files
    elif file_extension == '.pdf':
        st.info("Processing PDF document...")
        text = extract_text_from_pdf(file_path)
        
        # Check if we got meaningful text
        if len(text.strip()) < 10:
            
            
            # Try PyPDF2 as a fallback for text extraction
            try:
                import PyPDF2
                
                
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pypdf_text = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        pypdf_text += f"\n\n--- Page {page_num+1} ---\n\n"
                        pypdf_text += page.extract_text() or ""
                
                if len(pypdf_text.strip()) > len(text.strip()):
                    text = pypdf_text
                    
            except Exception as e:
                st.warning(f"PyPDF2 extraction failed: {str(e)}")
                
        return text
        
    # For Word documents
    elif file_extension in ['.docx', '.doc']:
        st.info("Processing Word document...")
        return extract_text_from_docx(file_path)
        
    # For PowerPoint files
    elif file_extension in ['.pptx', '.ppt']:
        st.info("Processing PowerPoint presentation...")
        return extract_text_from_pptx(file_path)
        
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return ""

def generate_summary_with_gemini(text):
    """Generate a summary of the text using Google Gemini AI"""
    if not GEMINI_AVAILABLE:
        st.warning("Gemini AI is not available. Generating a basic summary...")
        return generate_basic_summary(text)
    
    try:
        if not text or len(text.strip()) < 50:
            return "Not enough text to generate a meaningful summary."
        
        prompt = f"""
        Please summarize the following text concisely:
        
        {text[:50000]}  # Limit to avoid token limits
        
        Provide a clear and concise summary that captures the main points.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Summary generation error: {str(e)}")
        return generate_basic_summary(text)

def generate_basic_summary(text):
    """Generate a very basic summary when Gemini is not available"""
    if not text or len(text.strip()) < 50:
        return "Not enough text to generate a meaningful summary."
    
    # Basic summarization logic - extract key sentences
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    
    if len(sentences) <= 5:
        return ". ".join(sentences) + "."
    
    # Take first 2 sentences, middle sentence, last 2 sentences
    summary_sentences = sentences[:2] + [sentences[len(sentences)//2]] + sentences[-2:]
    return ". ".join(summary_sentences) + "."

def download_file(url, target_path):
    """Download a file from a URL with better error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code != 200:
            st.error(f"Download failed with status code: {response.status_code}")
            return False
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return True
    except requests.exceptions.MissingSchema:
        st.error("Invalid URL. Please make sure the URL includes http:// or https://")
        return False
    except requests.exceptions.ConnectionError:
        st.error("Connection error. Please check your internet connection and the URL.")
        return False
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server took too long to respond.")
        return False
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return False

def create_download_pdf(extracted_text, summary):
    """Create a downloadable PDF with extracted text and summary"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Set up fonts
        pdf.set_font("Arial", "B", 16)
        pdf.cell(190, 10, "Document Text Extraction Report", ln=True, align="C")
        pdf.ln(5)
        
        # Add extraction date
        pdf.set_font("Arial", "", 10)
        from datetime import datetime
        pdf.cell(190, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)
        
        # Add summary section if available
        if summary:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(190, 10, "Summary:", ln=True)
            pdf.set_font("Arial", "", 10)
            
            # Handle summary text - clean it and break into chunks
            summary_text = summary.encode('latin-1', 'replace').decode('latin-1')
            chunk_size = 5000
            for i in range(0, len(summary_text), chunk_size):
                summary_chunk = summary_text[i:i+chunk_size]
                pdf.multi_cell(190, 8, summary_chunk)
            
            pdf.ln(5)
        
        # Add extracted text section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(190, 10, "Extracted Text:", ln=True)
        pdf.set_font("Arial", "", 10)
        
        # Handle long text - break into chunks and clean non-latin characters
        text = extracted_text.encode('latin-1', 'replace').decode('latin-1')
        chunk_size = 5000  # Adjust based on PDF library limitations
        
        for i in range(0, len(text), chunk_size):
            text_chunk = text[i:i+chunk_size]
            pdf.multi_cell(190, 8, text_chunk)
        
        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        st.error(f"PDF creation error: {str(e)}")
        return None

def create_download_link(pdf_bytes, filename):
    """Create a download link for the PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">Download PDF Report</a>'
    return href

# Main application interface
st.title("Document Text Extractor and Summarizer")
st.write("Extract text from documents using OCR and generate summaries")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload File", "From URL"])

with tab1:
    # File upload section
    uploaded_file = st.file_uploader("Upload a document", type=["jpg", "jpeg", "png", "pdf", "docx", "doc", "pptx", "ppt"])

with tab2:
    # URL input
    url_input = st.text_input("Enter a document URL:", placeholder="https://example.com/document.pdf")

# Initialize session state for storing extraction results
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = ""

# Process button
if st.button("Process Document", type="primary"):
    if not uploaded_file and not url_input:
        st.error("Please either upload a document or provide a URL")
    else:
        try:
            # Reset state
            st.session_state.extracted_text = ""
            st.session_state.summary = ""
            
            # Process uploaded file
            if uploaded_file:
                # Get file extension
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                file_name = uploaded_file.name
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    file_path = tmp_file.name
                
                st.success(f"File uploaded: {file_name}")
                st.session_state.current_file_name = file_name
            
            # Process URL
            elif url_input:
                # Extract file extension from URL
                file_extension = os.path.splitext(url_input.split('?')[0])[1].lower()
                if not file_extension:
                    st.error("Cannot determine file type from URL. Please ensure URL ends with a file extension.")
                    st.stop()
                
                file_name = os.path.basename(url_input.split('?')[0])
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    file_path = tmp_file.name
                
                # Download the file
                st.info(f"Downloading file from URL...")
                if not download_file(url_input, file_path):
                    os.unlink(file_path)
                    st.stop()
                
                st.success("File downloaded successfully")
                st.session_state.current_file_name = file_name
            
            # Validate file extension
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.pdf', '.docx', '.doc', '.pptx', '.ppt']
            if file_extension not in supported_extensions:
                st.error(f"Unsupported file type: {file_extension}")
                os.unlink(file_path)
                st.stop()
            
            # Display preview for image files
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    img = Image.open(file_path)
                    st.image(img, caption="Document Preview", width=400)
                except Exception as e:
                    st.warning(f"Could not preview image: {str(e)}")
            
            # Extract text from the document
            with st.spinner("Extracting text from document..."):
                extracted_text = extract_text_from_document(file_path, file_extension)
                
                # Clean up the temporary file
                os.unlink(file_path)
                
                if not extracted_text or len(extracted_text.strip()) < 10:
                    st.warning("Little or no text was extracted from the document.")
                    if GEMINI_AVAILABLE and file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
                        
                        try:
                            with open(file_path, "rb") as img_file:
                                image_bytes = img_file.read()
                            
                            prompt = "Extract all text from this image, preserving layout as much as possible."
                            response = gemini_model.generate_content([
                                prompt,
                                {"mime_type": "image/jpeg", "data": image_bytes}
                            ])
                            extracted_text = response.text
                            st.success("Text extracted using Gemini AI")
                        except Exception as e:
                            st.error(f"Gemini extraction failed: {str(e)}")
                else:
                    st.success("Text extracted successfully!")
                
                st.session_state.extracted_text = extracted_text
            
            # Display and allow editing of extracted text
            if st.session_state.extracted_text:
                st.subheader("Extracted Text")
                st.text_area(
                    "You can edit the text if needed:", 
                    value=st.session_state.extracted_text,
                    height=300,
                    key="edited_text"
                )
                
                # Generate summary button
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary_with_gemini(st.session_state.edited_text)
                        st.session_state.summary = summary
                    
                    if st.session_state.summary:
                        st.subheader("Summary")
                        st.write(st.session_state.summary)
                        
                        # Export to PDF option
                        if st.button("Export to PDF"):
                            with st.spinner("Creating PDF..."):
                                pdf_bytes = create_download_pdf(st.session_state.edited_text, st.session_state.summary)
                                if pdf_bytes:
                                    st.markdown(
                                        create_download_link(pdf_bytes, f"{st.session_state.current_file_name}_report"),
                                        unsafe_allow_html=True
                                    )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Display existing extraction results if available
elif st.session_state.extracted_text:
    st.subheader("Previously Extracted Text")
    st.text_area(
        "You can edit the text if needed:", 
        value=st.session_state.extracted_text,
        height=300,
        key="edited_text"
    )
    
    # Generate or display summary
    if st.session_state.summary:
        st.subheader("Summary")
        st.write(st.session_state.summary)
        
        # Export to PDF option
        if st.button("Export to PDF"):
            with st.spinner("Creating PDF..."):
                pdf_bytes = create_download_pdf(st.session_state.edited_text, st.session_state.summary)
                if pdf_bytes:
                    st.markdown(
                        create_download_link(pdf_bytes, f"{st.session_state.current_file_name}_report"),
                        unsafe_allow_html=True
                    )
    else:
        # Generate summary button
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = generate_summary_with_gemini(st.session_state.edited_text)
                st.session_state.summary = summary
            
            if st.session_state.summary:
                st.subheader("Summary")
                st.write(st.session_state.summary)

# Add expander for troubleshooting
with st.expander("Troubleshooting"):
    st.markdown("""
    ### Common Issues and Solutions
    
    #### Little or No Text Extracted
    1. **For Images:** Make sure the image is clear and text is readable. Try preprocessing the image (increasing contrast, etc.) before uploading.
    2. **For PDFs:** Ensure that Poppler is installed correctly. Try extracting text from individual pages separately.
    3. **For Scanned Documents:** Some heavily formatted or low-quality scanned documents may not extract well with OCR.
    
    #### Installation Issues
    1. **Tesseract OCR:** Make sure Tesseract is installed and in your PATH.
       - Windows: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
       - Mac: `brew install tesseract`
       - Linux: `sudo apt install tesseract-ocr`
    
    2. **Poppler for PDF Support:**
       - Windows: `conda install -c conda-forge poppler`
       - Mac: `brew install poppler`
       - Linux: `sudo apt-get install -y poppler-utils`
       
    3. **Python Packages:**
       ```
       pip install streamlit pytesseract pdf2image python-docx python-pptx Pillow opencv-python-headless google-generativeai fpdf
       ```
    """)

# Add feature to try different OCR engines when available
with st.expander("Advanced Options"):
    st.markdown("""
    ### OCR Engine Selection
    
    When text extraction fails, the application automatically tries:
    1. Multiple Tesseract page segmentation modes
    2. Different image preprocessing techniques
    3. Gemini AI as a fallback for image-based documents
    
    ### Improving Results
    
    For better OCR results:
    - Use high-resolution images
    - Ensure good contrast between text and background
    - For PDFs, consider extracting individual pages if the full document fails
    """)