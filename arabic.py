from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import openai
import os
import streamlit as st
import hashlib
import tempfile
import json
import re
from typing import List
# Import required libraries for visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

# Chart generator functionality is built into the main app

load_dotenv()

# Configuration
api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=api_key)

# PostgreSQL connection parameters
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "pdf_chatbot"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

# File upload limits - 1GB max file size
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB in bytes

# CHANGED: AraE5 model configuration for Arabic documents
ARABIC_MODEL_NAME = "intfloat/multilingual-e5-large"  # AraE5-compatible model
EMBEDDING_DIMENSIONS = 1024  # CHANGED: AraE5 model dimensions
USE_ARAE5_CUSTOM = True  # Flag to use custom AraE5 implementation

# Streamlit UI setup with increased upload limit
st.set_page_config(page_title="Arabic PDF Chatbot with AraE5 - Up to 1GB", layout="wide")

# Initialize session state
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "arabic_tokenizer" not in st.session_state:
    st.session_state.arabic_tokenizer = None
if "arabic_model" not in st.session_state:
    st.session_state.arabic_model = None
if "selected_document_ids" not in st.session_state:
    st.session_state.selected_document_ids = []
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "single"  # "single" or "multi"

# CHANGED: New function to initialize AraE5 model
def init_arae5_model():
    """Initialize AraE5 model using transformers library."""
    try:
        # Load tokenizer and model for AraE5
        tokenizer = AutoTokenizer.from_pretrained(ARABIC_MODEL_NAME)
        model = AutoModel.from_pretrained(ARABIC_MODEL_NAME)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading AraE5 model: {e}")
        return None, None, None

# CHANGED: New function for AraE5 embeddings
def get_arae5_embeddings(texts, tokenizer, model, device, prefix="query: "):
    """Generate embeddings using AraE5 model with proper prefixing."""
    embeddings = []
    
    # CHANGED: Add Arabic-specific prefix for better performance
    prefixed_texts = [f"{prefix}{text}" for text in texts]
    
    with torch.no_grad():
        for text in prefixed_texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             padding=True, max_length=512).to(device)
            
            # Get embeddings
            outputs = model(**inputs)
            
            # Mean pooling
            embeddings_tensor = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Apply attention mask and mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings_tensor.size()).float()
            embeddings_tensor = embeddings_tensor * input_mask_expanded
            pooled = torch.sum(embeddings_tensor, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize
            pooled = F.normalize(pooled, p=2, dim=1)
            
            embeddings.append(pooled.cpu().numpy()[0])
    
    return embeddings

def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        st.error(f"Database connection error: {e}")
        return None

def init_database():
    """Initialize database tables and pgvector extension."""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table with file size tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) UNIQUE NOT NULL,
                    file_size BIGINT NOT NULL,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # CHANGED: Updated to 1024 dimensions for AraE5
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding vector(1024),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            return True
    except psycopg2.Error as e:
        st.error(f"Database initialization error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def calculate_file_hash(uploaded_file):
    """Calculate SHA-256 hash of uploaded file."""
    uploaded_file.seek(0)
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.sha256(file_content).hexdigest()

def check_file_size(uploaded_file):
    """Check if file size is within limits."""
    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE:
        return False, f"File size ({file_size / (1024*1024*1024):.2f} GB) exceeds maximum limit of 1GB"
    return True, f"File size: {file_size / (1024*1024):.2f} MB"

def document_exists(file_hash):
    """Check if document with given hash already exists in database."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename FROM documents WHERE file_hash = %s;", (file_hash,))
            result = cur.fetchone()
            return result[0] if result else None
    except psycopg2.Error as e:
        st.error(f"Error checking document existence: {e}")
        return None
    finally:
        conn.close()

# CHANGED: Updated for Arabic text processing with better chunking
def load_and_chunk_pdf(uploaded_file, chunk_size=1200, chunk_overlap=300):
    """
    Loads a PDF file from Streamlit's file uploader, extracts text, and splits it into manageable chunks.
    CHANGED: Optimized for Arabic text with larger chunks and better separators.
    
    Parameters:
    - uploaded_file (BytesIO): The uploaded PDF file.
    - chunk_size (int): The maximum number of characters per chunk (increased for Arabic).
    - chunk_overlap (int): Overlap between chunks to preserve context (increased for Arabic).
    
    Returns:
    - List of text chunks extracted from the PDF.
    """
    try:
        # Create a temporary file for large PDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # CHANGED: Arabic-optimized text splitter with Arabic punctuation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", "؟", "!", "،", "؛", " ", ""],  # Arabic punctuation added
            keep_separator=True
        )
        chunks = text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        
        # Cleanup temporary file
        os.unlink(temp_file_path)
        
        print(f"Arabic PDF processed: {len(texts)} chunks created")
        return texts
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

# CHANGED: Complete rewrite for AraE5 embedding generation
def generate_embeddings(texts, model_name=ARABIC_MODEL_NAME):
    """Generate embeddings for Arabic text chunks using AraE5 model with progress tracking."""
    
    # Initialize AraE5 model components
    tokenizer, model, device = init_arae5_model()
    if not tokenizer or not model:
        st.error("Failed to initialize AraE5 model")
        return None, []
    
    # Process in batches for large files
    batch_size = 8  # CHANGED: Smaller batches for AraE5 processing
    embeddings = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # CHANGED: Use custom AraE5 embedding function
        batch_embeddings = get_arae5_embeddings(
            batch, tokenizer, model, device, prefix="passage: "
        )
        embeddings.extend(batch_embeddings)
        
        # Update progress
        progress = min((i + batch_size) / len(texts), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing AraE5 embeddings: {i + len(batch)}/{len(texts)} chunks")
    
    progress_bar.empty()
    status_text.empty()
    
    print(f"Generated {len(embeddings)} embeddings with AraE5 model")
    
    # CHANGED: Return a wrapper class that mimics sentence-transformers interface
    class AraE5Wrapper:
        def __init__(self, tokenizer, model, device):
            self.tokenizer = tokenizer
            self.model = model
            self.device = device
        
        def encode(self, texts, convert_to_tensor=False, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return get_arae5_embeddings(texts, self.tokenizer, self.model, self.device, prefix="query: ")
    
    wrapper = AraE5Wrapper(tokenizer, model, device)
    return wrapper, embeddings

def store_document_and_chunks(filename, file_hash, file_size, texts, embeddings):
    """Store document and its chunks with embeddings in PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            # Insert document with file size
            cur.execute("""
                INSERT INTO documents (filename, file_hash, file_size) 
                VALUES (%s, %s, %s) RETURNING id;
            """, (filename, file_hash, file_size))
            document_id = cur.fetchone()[0]
            
            # Insert chunks with embeddings in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    embedding_list = embedding.tolist()
                    cur.execute("""
                        INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s);
                    """, (document_id, text, i + j, embedding_list))
                
                conn.commit()  # Commit in batches
            
            print(f"Stored Arabic document with ID {document_id} and {len(texts)} chunks")
            return document_id
    except psycopg2.Error as e:
        st.error(f"Error storing document: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def search_similar_chunks_multi_docs(query, document_ids: List[int], embedding_model, top_k=7):
    """Search for similar chunks across multiple documents using pgvector cosine similarity."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        # CHANGED: Generate query embedding using AraE5 wrapper
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]
        query_embedding_list = query_embedding.tolist()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Search across multiple documents
            placeholders = ','.join(['%s'] * len(document_ids))
            
            cur.execute(f"""
                SELECT dc.chunk_text, 
                       dc.chunk_index,
                       d.filename,
                       dc.embedding <=> %s::vector as similarity_score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.document_id IN ({placeholders})
                AND dc.embedding <=> %s::vector < 0.8
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s;
            """, [query_embedding_list] + document_ids + [query_embedding_list, query_embedding_list, top_k])
            
            results = cur.fetchall()
            
            if results:
                print(f"Found {len(results)} relevant chunks across {len(document_ids)} documents using AraE5:")
                chunks_with_source = []
                for i, result in enumerate(results):
                    print(f"Chunk {i+1} from {result['filename']}: Score {result['similarity_score']:.3f}")
                    # Add source information to chunk
                    chunk_with_source = f"[Source: {result['filename']}]\n{result['chunk_text']}"
                    chunks_with_source.append(chunk_with_source)
                
                return chunks_with_source
            else:
                print("No chunks found within similarity threshold, trying broader search...")
                # Fallback: get top chunks without threshold
                cur.execute(f"""
                    SELECT dc.chunk_text, 
                           dc.chunk_index,
                           d.filename,
                           dc.embedding <=> %s::vector as similarity_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.document_id IN ({placeholders})
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s;
                """, [query_embedding_list] + document_ids + [query_embedding_list, top_k])
                
                fallback_results = cur.fetchall()
                if fallback_results:
                    chunks_with_source = []
                    for result in fallback_results:
                        chunk_with_source = f"[Source: {result['filename']}]\n{result['chunk_text']}"
                        chunks_with_source.append(chunk_with_source)
                    return chunks_with_source
                else:
                    return ["No relevant results found."]
                    
    except psycopg2.Error as e:
        st.error(f"Error searching chunks: {e}")
        return ["Error occurred during search."]
    finally:
        conn.close()

def search_similar_chunks_single_doc(query, document_id, embedding_model, top_k=5):
    """Search for similar chunks in a single document using pgvector cosine similarity."""
    return search_similar_chunks_multi_docs(query, [document_id], embedding_model, top_k)

def detect_visualization_request(query):
    """Detect if user is requesting visualizations - CHANGED: Added Arabic keywords."""
    visualization_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display', 'bar chart', 
        'line chart', 'pie chart', 'histogram', 'scatter plot', 'trend', 
        'distribution', 'comparison', 'analyze', 'create chart', 'make graph',
        # Arabic keywords added
        'رسم', 'مخطط', 'جدول', 'عرض', 'تصور', 'تحليل', 'إحصائية', 'بيانات'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in visualization_keywords)

def detect_table_request(query):
    """Detect if user is requesting tables - CHANGED: Added Arabic keywords."""
    table_keywords = [
        'table', 'list', 'summary', 'breakdown', 'details', 'data', 
        'show me', 'display', 'extract', 'find', 'get',
        # Arabic keywords added
        'جدول', 'قائمة', 'ملخص', 'تفاصيل', 'بيانات', 'استخراج', 'عرض'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in table_keywords)

# CHANGED: Enhanced system prompt for Arabic language support
def generate_response(query, retrieved_chunks, is_multi_doc=False):
    """Generate response using OpenAI API with retrieved chunks in JSON format - Enhanced for Arabic."""
    if not retrieved_chunks or retrieved_chunks == ["No relevant results found."] or retrieved_chunks == ["Error occurred during search."]:
        return json.dumps([{"type": "text", "content": "لم يتم العثور على معلومات ذات صلة في المستند(ات) للإجابة على سؤالك. / No relevant information found in the document(s) to answer your question."}])

    try:
        # Combine chunks into context
        context = "\n\n".join(retrieved_chunks)
        
        # Enhanced system prompt for JSON formatted responses with Arabic support
        multi_doc_instruction = ""
        if is_multi_doc:
            multi_doc_instruction = """
IMPORTANT: You are searching across MULTIPLE documents (Arabic/English). When answering:
- Mention which document(s) contain the relevant information
- If information comes from different documents, clearly indicate the sources
- Synthesize information from multiple sources when applicable
- If there are contradictions between documents, mention them
- Support both Arabic and English responses based on the query language
"""
        
        # CHANGED: Updated system prompt with Arabic language support
        system_prompt = f"""You are an intelligent PDF document assistant specialized in Arabic and English content. Your job is to analyze PDF content and provide helpful responses in a specific JSON format using AraE5 embeddings for superior Arabic understanding.

{multi_doc_instruction}

LANGUAGE HANDLING:
- Detect query language (Arabic/English) and respond in the same language
- For Arabic queries, provide Arabic responses with English technical terms when needed
- For English queries, provide English responses
- Always maintain technical accuracy regardless of language

RESPONSE TYPE DETERMINATION:

1. VISUALIZATION GENERATION (when user requests charts, graphs, plots, or visual analysis):
- Generate matplotlib/seaborn code for data visualization
- Extract relevant data from the PDF content for plotting
- Create meaningful charts based on available information
- Add Arabic labels if query is in Arabic

2. TABLE GENERATION (when user requests tables, lists, summaries, or structured data):
- Extract and structure relevant information from PDF content
- Create tables with meaningful data organization
- Format data in a clear, readable structure
- Use Arabic headers if query is in Arabic

3. TEXT RESPONSES (for explanations, insights, and general queries):
- Provide clear, comprehensive answers in the query language
- Include insights and analysis of the PDF content
- Reference specific document sources when applicable

MANDATORY JSON RESPONSE FORMAT (ABSOLUTE REQUIREMENT):
ENTIRE response must be valid JSON array - NO exceptions:

[
    {{"type": "text", "content": "Detailed explanatory text in query language, insights, context, assumptions, or business implications based on the PDF content"}},
    {{"type": "img", "content": "import matplotlib.pyplot as plt\\nimport pandas as pd\\nimport numpy as np\\n\\n# Complete executable visualization code based on PDF data\\n# Extract relevant data from context and create meaningful visualizations\\nplt.figure(figsize=(12, 8))\\n# Use Arabic fonts if needed: plt.rcParams['font.family'] = ['Tahoma', 'DejaVu Sans']\\n# Visualization implementation\\nplt.title('Chart Title Based on PDF Content')\\nplt.xlabel('X Label')\\nplt.ylabel('Y Label')\\nplt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\\nplt.tight_layout(pad=3.0)\\nplt.show()"}},
    {{"type": "table", "content": "import pandas as pd\\nimport json\\n\\n# Extract and structure data from PDF content\\ndata = [\\n    {{'Item': 'Example 1', 'Value': 'Data from PDF'}},\\n    {{'Item': 'Example 2', 'Value': 'More PDF data'}}\\n]\\nresult_df = pd.DataFrame(data)\\nprint(result_df.to_json(orient='records', date_format='iso'))"}},
    {{"type": "text", "content": "Additional insights, conclusions, or summary in query language based on the PDF analysis"}}
]

JSON FORMAT RULES (STRICTLY ENFORCED):
- Valid JSON only - no text outside structure
- NO markdown blocks (```python) - code goes directly in "content"
- Every content piece requires JSON object with "type" and "content"
- Line breaks: \\n, Quotes: \\"
- Multiple images/tables allowed and encouraged
- "img" type: complete runnable Python visualization code
- "table" type: code outputting JSON via print(result_df.to_json(orient='records', date_format='iso'))
- "text" type: ONLY direct findings and business insights - NO code explanations
- Extract actual data from PDF content for visualizations and tables
- Support Arabic text rendering in visualizations when needed

Remember: You are using AraE5 embeddings for superior Arabic text understanding. Extract maximum value from Arabic/English PDF content and present it in the requested format."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"""Based on the following PDF document excerpts (processed with AraE5 embeddings), please answer this question: "{query}"

Document Content:
{context}

Please provide a comprehensive response in the required JSON format. Extract relevant data from the PDF content to create meaningful visualizations or tables if requested. Respond in the same language as the query."""
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        response_content = response.choices[0].message.content
        
        # Parse and validate JSON response
        try:
            # Extract JSON from code blocks if present
            if "```json" in response_content:
                json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1)
            elif "```" in response_content:
                json_match = re.search(r'```\n(.*?)\n```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1)
            
            # Validate JSON
            parsed_response = json.loads(response_content)
            return json.dumps(parsed_response, ensure_ascii=False)
        except json.JSONDecodeError:
            # Fallback to simple text response
            return json.dumps([{"type": "text", "content": response_content}])
            
    except Exception as e:
        return json.dumps([{"type": "text", "content": f"Error generating response: {e}"}])

def parse_json_response(json_response):
    """Parse JSON response and separate different content types."""
    try:
        response_data = json.loads(json_response)
        text_parts = []
        image_parts = []
        table_parts = []
        
        for item in response_data:
            if item.get("type") == "text":
                text_parts.append(item.get("content", ""))
            elif item.get("type") == "img":
                image_parts.append(item.get("content", ""))
            elif item.get("type") == "table":
                table_parts.append(item.get("content", ""))
        
        return text_parts, image_parts, table_parts
    except json.JSONDecodeError:
        return [json_response], [], []

# CHANGED: Enhanced visualization with Arabic font support
def execute_visualization_code(code):
    """Execute visualization code and capture the plot with Arabic font support."""
    try:
        # CHANGED: Set up Arabic font support
        plt.rcParams['font.family'] = ['Tahoma', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create a local environment for code execution
        local_env = {
            'plt': plt,
            'pd': pd,
            'np': np,
            'matplotlib': matplotlib
        }
        exec(code, {"__builtins__": {}}, local_env)
        return True
    except Exception as e:
        st.error(f"Error executing visualization code: {e}")
        return False

def execute_table_code(code):
    """Execute table code and capture the output."""
    try:
        import io
        import contextlib
        import ast
        
        # First, try to parse the code to check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as syntax_error:
            st.error(f"Syntax error in generated table code: {syntax_error}")
            return None
        
        # Create a safe execution environment with all necessary modules
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                '__import__': __import__,  # Allow imports
            },
            'pd': pd,
            'pandas': pd,
            'json': json,
            'np': np,
            'numpy': np
        }
        
        # Capture print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            exec(code, safe_globals)
        
        table_output = captured_output.getvalue().strip()
        if table_output:
            try:
                return json.loads(table_output)
            except json.JSONDecodeError as json_error:
                st.error(f"Invalid JSON output from table code: {json_error}")
                # Try to display the raw output as a fallback
                st.text("Raw output:")
                st.text(table_output)
                return None
        return None
    except Exception as e:
        st.error(f"Error executing table code: {e}")
        # For debugging, show the problematic code
        with st.expander("View problematic code"):
            st.code(code, language="python")
        return None
    
def get_uploaded_documents():
    """Get list of uploaded documents from database with file sizes."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, filename, file_size, upload_timestamp 
                FROM documents 
                ORDER BY upload_timestamp DESC;
            """)
            return cur.fetchall()
    except psycopg2.Error as e:
        st.error(f"Error fetching documents: {e}")
        return []
    finally:
        conn.close()

def delete_document(document_id):
    """Delete document and its chunks from database."""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s;", (document_id,))
            conn.commit()
            return True
    except psycopg2.Error as e:
        st.error(f"Error deleting document: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.2f} GB"

# Initialize database on startup
if init_database():
    # CHANGED: Updated success message for AraE5
    st.success("Database initialized successfully! (Using AraE5 embeddings for Arabic)", icon="✅")
else:
    st.error("Failed to initialize database. Please check your PostgreSQL connection.", icon="❌")

# CHANGED: Initialize AraE5 embedding model
if st.session_state.embedding_model is None:
    with st.spinner("Loading AraE5 embedding model for Arabic documents..."):
        try:
            # Try to load as sentence transformer first
            st.session_state.embedding_model = SentenceTransformer(ARABIC_MODEL_NAME)
        except Exception as e:
            st.warning(f"Could not load as sentence transformer: {e}")
            st.info("Initializing custom AraE5 implementation...")
            tokenizer, model, device = init_arae5_model()
            if tokenizer and model:
                # Create wrapper
                class AraE5Wrapper:
                    def __init__(self, tokenizer, model, device):
                        self.tokenizer = tokenizer
                        self.model = model
                        self.device = device
                    
                    def encode(self, texts, convert_to_tensor=False, **kwargs):
                        if isinstance(texts, str):
                            texts = [texts]
                        return get_arae5_embeddings(texts, self.tokenizer, self.model, self.device, prefix="query: ")
                
                st.session_state.embedding_model = AraE5Wrapper(tokenizer, model, device)
                st.success("AraE5 model loaded successfully!")
            else:
                st.error("Failed to initialize AraE5 model")

# Sidebar for file upload and document management
with st.sidebar:
    st.header("📄 إدارة المستندات / Document Management")
    
    # CHANGED: File uploader with Arabic/English help text
    uploaded_file = st.file_uploader(
        "Upload a PDF / رفع ملف PDF", 
        type=["pdf"], 
        key="file_uploader",
        help="Maximum file size: 1GB / الحد الأقصى لحجم الملف: 1 جيجابايت"
    )
    
    if uploaded_file is not None:
        # Check file size
        size_ok, size_msg = check_file_size(uploaded_file)
        
        if not size_ok:
            st.error(size_msg)
        else:
            file_hash = calculate_file_hash(uploaded_file)
            existing_doc_id = document_exists(file_hash)
            
            if existing_doc_id:
                st.info(f"Document already exists in database! / المستند موجود بالفعل في قاعدة البيانات!")
                if existing_doc_id not in st.session_state.selected_document_ids:
                    st.session_state.selected_document_ids = [existing_doc_id]
            else:
                with st.spinner("Processing Arabic PDF with AraE5... / معالجة ملف PDF العربي باستخدام AraE5..."):
                    texts = load_and_chunk_pdf(uploaded_file)
                    if texts:
                        embedding_model, embeddings = generate_embeddings(texts)
                        document_id = store_document_and_chunks(
                            uploaded_file.name, file_hash, uploaded_file.size, texts, embeddings
                        )
                        if document_id:
                            st.session_state.selected_document_ids = [document_id]
                            st.success(f"✅ {uploaded_file.name} processed successfully with AraE5! / تمت معالجة الملف بنجاح!")
                        else:
                            st.error("Failed to store document in database. / فشل في حفظ المستند في قاعدة البيانات.")
    
    st.divider()
    
    # CHANGED: Chat mode selection with Arabic text
    st.subheader("🎯 وضع الاستعلام / Query Mode")
    chat_mode = st.radio(
        "Choose how to search / اختر طريقة البحث:",
        ["Single Document / مستند واحد", "Multiple Documents / مستندات متعددة"],
        index=0 if st.session_state.chat_mode == "single" else 1
    )
    st.session_state.chat_mode = "single" if "Single" in chat_mode else "multi"
    
    # CHANGED: Document selector with Arabic text
    st.subheader("📚 المستندات المتاحة / Available Documents")
    documents = get_uploaded_documents()
    
    if documents:
        if st.session_state.chat_mode == "single":
            # Single document selection
            doc_options = {f"{doc['filename']} ({format_file_size(doc['file_size'])})": doc['id'] 
                          for doc in documents}
            
            selected_doc = st.selectbox("Select a document to chat with / اختر مستندًا للدردشة معه:", 
                                      options=list(doc_options.keys()),
                                      index=0)
            
            if selected_doc:
                selected_doc_id = doc_options[selected_doc]
                st.session_state.selected_document_ids = [selected_doc_id]
        else:
            # Multiple document selection
            st.write("Select multiple documents to search across / اختر مستندات متعددة للبحث:")
            selected_docs = []
            
            for doc in documents:
                doc_name = f"{doc['filename']} ({format_file_size(doc['file_size'])})"
                if st.checkbox(doc_name, value=doc['id'] in st.session_state.selected_document_ids, key=f"doc_{doc['id']}"):
                    selected_docs.append(doc['id'])
            
            st.session_state.selected_document_ids = selected_docs
            
            if selected_docs:
                st.success(f"Selected {len(selected_docs)} document(s) for multi-document search / تم اختيار {len(selected_docs)} مستند للبحث")
        
        # CHANGED: Delete document functionality with Arabic text
        st.divider()
        st.subheader("🗑️ حذف المستندات / Delete Documents")
        doc_to_delete = st.selectbox(
            "Select document to delete / اختر مستندًا للحذف:",
            options=[""] + [f"{doc['filename']} ({format_file_size(doc['file_size'])})" for doc in documents],
            key="delete_selector"
        )
        
        if doc_to_delete and st.button("Delete Selected Document / حذف المستند المحدد", type="secondary"):
            doc_id_to_delete = next(doc['id'] for doc in documents if f"{doc['filename']} ({format_file_size(doc['file_size'])})" == doc_to_delete)
            if delete_document(doc_id_to_delete):
                st.success("Document deleted successfully! / تم حذف المستند بنجاح!")
                st.rerun()
    else:
        st.info("No documents uploaded yet. / لم يتم رفع أي مستندات بعد.")
    
    # CHANGED: Chart generation info with Arabic examples
    st.divider()
    st.subheader("📊 التحليل المتقدم / Enhanced Analysis")
    st.info("""Ask for / اطلب:
• 'Create a chart of sales data' / 'أنشئ مخططًا لبيانات المبيعات'
• 'Show me a table of financial info' / 'أرني جدولاً للمعلومات المالية'  
• 'Visualize the trends' / 'تصور الاتجاهات'
• 'Extract key metrics in a table' / 'استخرج المقاييس الرئيسية في جدول'""")

# CHANGED: Main chat interface with Arabic support
st.title("💬 تحدث مع ملفات PDF / Chat with your PDF(s)")
if st.session_state.chat_mode == "multi":
    st.caption("🚀 Multi-document RAG with AraE5 Embeddings / بحث متعدد المستندات باستخدام تضمينات AraE5")
else:
    st.caption("🚀 Single-document RAG with AraE5 Embeddings / بحث مستند واحد باستخدام تضمينات AraE5")

if "messages" not in st.session_state:
    # CHANGED: Updated initial message with Arabic support
    st.session_state["messages"] = [
        {"role": "assistant", "content": """Upload PDF documents (up to 1GB each) and start asking questions in Arabic or English! 
        
ارفع مستندات PDF (حتى 1 جيجابايت لكل منها) وابدأ في طرح الأسئلة باللغة العربية أو الإنجليزية!

I'm now using the powerful AraE5 embedding model for superior Arabic text understanding. I can provide detailed analysis, create visualizations, generate tables, and extract insights from your documents.

أستخدم الآن نموذج تضمينات AraE5 القوي لفهم أفضل للنصوص العربية. يمكنني تقديم تحليل مفصل وإنشاء المخططات والجداول واستخراج الرؤى من مستنداتك."""}
    ]

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
        # Display chart if it exists in the message
        if "chart" in msg and msg["chart"] is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# CHANGED: Chat input with Arabic placeholder
if prompt := st.chat_input("Ask a question in Arabic or English / اطرح سؤالاً باللغة العربية أو الإنجليزية..."):
    if not st.session_state.selected_document_ids:
        st.warning("Please upload and select document(s) first. / يرجى رفع واختيار المستند(ات) أولاً.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Search for relevant chunks
        is_multi_doc = len(st.session_state.selected_document_ids) > 1
        # CHANGED: Search message with Arabic text
        search_msg = f"Searching across {len(st.session_state.selected_document_ids)} document(s) with AraE5... / البحث في {len(st.session_state.selected_document_ids)} مستند باستخدام AraE5..." if is_multi_doc else "Searching document with AraE5... / البحث في المستند باستخدام AraE5..."
        
        with st.spinner(search_msg):
            if is_multi_doc:
                retrieved_chunks = search_similar_chunks_multi_docs(
                    prompt, 
                    st.session_state.selected_document_ids, 
                    st.session_state.embedding_model,
                    top_k=10  # More chunks for multi-doc search
                )
            else:
                retrieved_chunks = search_similar_chunks_single_doc(
                    prompt, 
                    st.session_state.selected_document_ids[0], 
                    st.session_state.embedding_model
                )
        
        print("=== AraE5 SEARCH RESULTS ===")
        print(f"Query: {prompt}")
        print(f"Mode: {'Multi-doc' if is_multi_doc else 'Single-doc'}")
        print(f"Documents: {st.session_state.selected_document_ids}")
        print(f"Retrieved chunks: {len(retrieved_chunks)}")
        print("============================")
        
        # Generate JSON formatted response
        with st.spinner("Generating comprehensive analysis... / إنشاء تحليل شامل..."):
            json_response = generate_response(prompt, retrieved_chunks, is_multi_doc)
            
            # Parse JSON response
            text_parts, image_parts, table_parts = parse_json_response(json_response)
            
            # Display response parts
            response_content = ""
            
            # Parse and execute the JSON response
            try:
                response_data = json.loads(json_response)
                
                for item in response_data:
                    item_type = item.get("type", "")
                    content = item.get("content", "")
                    
                    if item_type == "text":
                        st.chat_message("assistant").write(content)
                        response_content += content + "\n\n"
                    
                    elif item_type == "img":
                        try:
                            st.chat_message("assistant").write("📊 Generating visualization... / إنشاء مخطط بصري...")
                            # Execute visualization code
                            exec(content)
                            st.pyplot(plt.gcf())
                            plt.close()
                            response_content += "[Visualization Generated / تم إنشاء المخطط]\n\n"
                        except Exception as e:
                            st.chat_message("assistant").write(f"Error generating visualization: {e}")
                            response_content += f"Error generating visualization: {e}\n\n"
                    
                    elif item_type == "table":
                        try:
                            st.chat_message("assistant").write("📋 Generating table... / إنشاء جدول...")
                            table_data = execute_table_code(content)
                            if table_data:
                                df_display = pd.DataFrame(table_data)
                                st.dataframe(df_display, use_container_width=True)
                                response_content += "[Table Generated / تم إنشاء الجدول]\n\n"
                            else:
                                st.chat_message("assistant").write("No table data generated. / لم يتم إنشاء بيانات الجدول.")
                                response_content += "No table data generated.\n\n"
                        except Exception as e:
                            st.chat_message("assistant").write(f"Error generating table: {e}")
                            response_content += f"Error generating table: {e}\n\n"
                
                # Store the response in session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_content.strip() if response_content.strip() else "Analysis completed. / تم إكمال التحليل."
                })
                
            except json.JSONDecodeError:
                # Fallback for non-JSON responses
                st.chat_message("assistant").write(json_response)
                st.session_state.messages.append({"role": "assistant", "content": json_response})

# CHANGED: Display current document info with Arabic text
if st.session_state.selected_document_ids:
    with st.expander("📋 معلومات الاختيار الحالي / Current Selection Info"):
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    placeholders = ','.join(['%s'] * len(st.session_state.selected_document_ids))
                    cur.execute(f"""
                        SELECT d.filename, d.file_size, d.upload_timestamp, COUNT(c.id) as chunk_count
                        FROM documents d
                        LEFT JOIN document_chunks c ON d.id = c.document_id
                        WHERE d.id IN ({placeholders})
                        GROUP BY d.id, d.filename, d.file_size, d.upload_timestamp
                        ORDER BY d.upload_timestamp DESC;
                    """, st.session_state.selected_document_ids)
                    doc_infos = cur.fetchall()
                    
                    if doc_infos:
                        total_chunks = sum(doc['chunk_count'] for doc in doc_infos)
                        total_size = sum(doc['file_size'] for doc in doc_infos)
                        
                        st.write(f"**Embedding Model / نموذج التضمين:** AraE5 (1024 dimensions)")
                        st.write(f"**Mode / الوضع:** {'Multi-document / متعدد المستندات' if len(doc_infos) > 1 else 'Single-document / مستند واحد'}")
                        st.write(f"**Total Documents / إجمالي المستندات:** {len(doc_infos)}")
                        st.write(f"**Total Size / الحجم الإجمالي:** {format_file_size(total_size)}")
                        st.write(f"**Total Chunks / إجمالي القطع:** {total_chunks}")
                        
                        st.write("**Documents / المستندات:**")
                        for doc in doc_infos:
                            st.write(f"- {doc['filename']} ({format_file_size(doc['file_size'])}) - {doc['chunk_count']} chunks")
            except psycopg2.Error as e:
                st.error(f"Error fetching document info: {e}")
            finally:
                conn.close()

# CHANGED: Add footer with Arabic examples
st.markdown("---")
with st.expander("💡 أمثلة التحليل المتقدم / Enhanced Analysis Examples"):
    st.markdown("""
    **Try these enhanced requests with AraE5 / جرب هذه الطلبات المتقدمة مع AraE5:**
    
    📊 **Data Visualization / تصور البيانات:** 
    - "Create a chart showing the financial trends" / "أنشئ مخططًا يوضح الاتجاهات المالية"
    - "Visualize the sales performance data" / "تصور بيانات أداء المبيعات"
    - "Generate a graph of quarterly results" / "أنشئ رسمًا بيانيًا للنتائج الفصلية"
    
    📋 **Structured Tables / الجداول المنظمة:** 
    - "Extract the financial data into a table" / "استخرج البيانات المالية في جدول"
    - "Show me a summary table of key metrics" / "أرني جدول ملخص للمقاييس الرئيسية"
    - "Create a comparison table" / "أنشئ جدول مقارنة"
    
    📈 **Comprehensive Analysis / التحليل الشامل:** 
    - "Analyze the market trends and show them visually" / "حلل اتجاهات السوق وأظهرها بصريًا"
    - "Extract insights and create charts" / "استخرج الرؤى وأنشئ المخططات"
    - "Provide complete analysis with tables" / "قدم تحليلاً كاملاً مع الجداول"
    
    💡 **Smart Insights / الرؤى الذكية:** 
    - "What are the key findings?" / "ما هي النتائج الرئيسية؟"
    - "Summarize the important metrics" / "لخص المقاييس المهمة"
    - "Show me the most important data points" / "أرني أهم نقاط البيانات"
    
    **Enhanced Features with AraE5 / الميزات المحسنة مع AraE5:**
    - Superior Arabic text understanding / فهم متفوق للنصوص العربية
    - Optimized for Arabic semantic search / محسن للبحث الدلالي العربي
    - 1024-dimensional embeddings for better accuracy / تضمينات 1024 بُعد لدقة أفضل
    - Bilingual support (Arabic & English) / دعم ثنائي اللغة
    - Arabic-aware text chunking / تقسيم النص المدرك للعربية
    - Enhanced Arabic punctuation handling / معالجة محسنة لعلامات الترقيم العربية
    - Arabic font support in visualizations / دعم الخطوط العربية في المخططات
    """)
    
# CHANGED: Add requirements note for AraE5
st.markdown("---")
st.info("""
**📦 Additional Requirements for AraE5 / متطلبات إضافية لـ AraE5:**

Install these packages / قم بتثبيت هذه الحزم:
```bash
pip install torch transformers sentence-transformers
```

**🔧 Model Details / تفاصيل النموذج:**
- Model: intfloat/multilingual-e5-large (AraE5 compatible)
- Dimensions: 1024 (updated from 768)
- Optimized for Arabic semantic understanding
- Supports both Arabic and English queries
""")