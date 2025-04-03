import os
import logging
from PyPDF2 import PdfReader
import docx2txt

logger = logging.getLogger(__name__)

def process_document(file_path):
    """
    Process a document file and extract its text content.
    Supports PDF, DOCX, and TXT files.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Extracted text content
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return process_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return process_docx(file_path)
        elif file_extension == '.txt':
            return process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def process_pdf(file_path):
    """
    Extract text from a PDF file
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        # Clean up excessive whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def process_docx(file_path):
    """
    Extract text from a DOCX file
    
    Args:
        file_path (str): Path to the DOCX file
        
    Returns:
        str: Extracted text content
    """
    try:
        text = docx2txt.process(file_path)
        # Clean up excessive whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        raise

def process_txt(file_path):
    """
    Extract text from a TXT file
    
    Args:
        file_path (str): Path to the TXT file
        
    Returns:
        str: Extracted text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Clean up excessive whitespace
        text = ' '.join(text.split())
        
        return text
    except UnicodeDecodeError:
        # Try with another encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            text = file.read()
        # Clean up excessive whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logger.error(f"Error processing TXT: {str(e)}")
        raise

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split a document into overlapping chunks for processing
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    
    if len(text) <= chunk_size:
        return [text]
    
    # Split into sentences to avoid cutting mid-sentence
    sentences = text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        # Add period back to sentence
        if not sentence.endswith('.'):
            sentence += '.'
        
        # If adding this sentence would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(current_chunk)
            
            # Start new chunk with overlap from previous chunk
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
