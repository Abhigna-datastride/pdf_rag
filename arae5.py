from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
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

# Arabic E5 model configuration - Using gimmeursocks/ara-e5-base
ARAE5_MODEL_NAME = "gimmeursocks/ara-e5-base"
EMBEDDING_DIMENSIONS = 768  # Standard for E5 base models
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fallback models in case primary model fails
FALLBACK_MODELS = [
    {"name": "gimmeursocks/ara-e5-small", "dimensions": 384},
    {"name": "intfloat/multilingual-e5-base", "dimensions": 768},
    {"name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "dimensions": 768}
]

# Streamlit UI setup with increased upload limit
st.set_page_config(page_title="Arabic PDF Chatbot with Ara-E5-Base - Up to 1GB", layout="wide")

# Initialize session state
if "arae5_tokenizer" not in st.session_state:
    st.session_state.arae5_tokenizer = None
if "arae5_model" not in st.session_state:
    st.session_state.arae5_model = None
if "model_info" not in st.session_state:
    st.session_state.model_info = {"name": ARAE5_MODEL_NAME, "dimensions": EMBEDDING_DIMENSIONS}
if "selected_document_ids" not in st.session_state:
    st.session_state.selected_document_ids = []
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "single"  # "single" or "multi"

def init_arae5_model():
    """Initialize the Ara-E5-Base model with fallback options."""
    models_to_try = [{"name": ARAE5_MODEL_NAME, "dimensions": EMBEDDING_DIMENSIONS}] + FALLBACK_MODELS
    
    for model_config in models_to_try:
        try:
            st.info(f"Loading Arabic model: {model_config['name']}...")
            
            # Load the Ara-E5 model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
            model = AutoModel.from_pretrained(
                model_config['name'],
                torch_dtype=torch.float32,  # Ensure compatibility
                trust_remote_code=True  # Allow custom model code if needed
            )
            
            # Move model to appropriate device
            model = model.to(DEVICE)
            model.eval()
            
            # Update model info
            model_info = {
                "name": model_config['name'],
                "dimensions": model_config['dimensions']
            }
            
            st.success(f"✅ Ara-E5 model loaded successfully: {model_config['name']} on {DEVICE}!")
            return tokenizer, model, model_info
            
        except Exception as e:
            st.warning(f"⚠️ Failed to load {model_config['name']}: {str(e)}")
            continue
    
    # If all models fail
    st.error("❌ Failed to load any Arabic embedding model. Please check your internet connection.")
    return None, None, None

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get sentence embeddings from transformer output."""
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_arae5_embeddings(texts, tokenizer, model, max_length=512):
    """Generate embeddings using the Ara-E5 model with proper preprocessing."""
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            # Preprocess text for E5 models (add query prefix for better performance)
            if not text.startswith("query: ") and not text.startswith("passage: "):
                processed_text = f"passage: {text}"
            else:
                processed_text = text
            
            # Tokenize the text
            encoded_input = tokenizer(
                processed_text, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors='pt'
            ).to(DEVICE)
            
            # Compute token embeddings
            model_output = model(**encoded_input)
            
            # Perform mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            embeddings.append(sentence_embeddings.cpu().numpy()[0])
    
    return embeddings

def get_query_embedding(query, tokenizer, model):
    """Generate embedding for query with proper E5 preprocessing."""
    query_text = f"query: {query}"
    return get_arae5_embeddings([query_text], tokenizer, model)[0]

class AraE5EmbeddingModel:
    """Wrapper class for Ara-E5 model to provide a consistent interface."""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def encode(self, texts, convert_to_tensor=False, **kwargs):
        """Encode texts using Ara-E5 model."""
        embeddings = get_arae5_embeddings(texts, self.tokenizer, self.model)
        
        if convert_to_tensor:
            return torch.tensor(embeddings)
        return embeddings
    
    def encode_query(self, query):
        """Encode query with proper E5 preprocessing."""
        return get_query_embedding(query, self.tokenizer, self.model)

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
            
            # Create documents table with file size tracking and model info
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) UNIQUE NOT NULL,
                    file_size BIGINT NOT NULL,
                    embedding_model VARCHAR(255) NOT NULL DEFAULT 'unknown',
                    embedding_dimensions INTEGER NOT NULL DEFAULT 768,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create document chunks table with flexible dimensions
            embedding_dim = st.session_state.model_info.get("dimensions", EMBEDDING_DIMENSIONS)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding vector({embedding_dim}),
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

def load_and_chunk_pdf(uploaded_file, chunk_size=1200, chunk_overlap=300):
    """Load PDF and chunk it optimally for Arabic text processing."""
    try:
        # Create a temporary file for large PDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # Arabic-optimized text splitter with comprehensive separators
        arabic_separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ".",     # English period
            "!",     # English exclamation
            "?",     # English question mark
            "؟",     # Arabic question mark
            "!",     # Arabic exclamation
            "،",     # Arabic comma
            "؛",     # Arabic semicolon
            ":",     # Colon
            "؛",     # Arabic semicolon (duplicate check)
            " ",     # Space
            ""       # Character level fallback
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=arabic_separators,
            keep_separator=True
        )
        
        chunks = text_splitter.split_documents(documents)
        texts = [chunk.page_content.strip() for chunk in chunks if chunk.page_content.strip()]
        
        # Cleanup temporary file
        os.unlink(temp_file_path)
        
        print(f"Arabic PDF processed with Ara-E5 optimization: {len(texts)} chunks created")
        return texts
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def generate_embeddings_with_arae5(texts):
    """Generate embeddings for Arabic text chunks using Ara-E5 model with progress tracking."""
    
    if not st.session_state.arae5_tokenizer or not st.session_state.arae5_model:
        st.error("Ara-E5 model not initialized")
        return None, []
    
    tokenizer = st.session_state.arae5_tokenizer
    model = st.session_state.arae5_model
    
    # Process in batches for efficiency
    batch_size = 4  # Conservative batch size for stability
    embeddings = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = get_arae5_embeddings(batch, tokenizer, model)
            embeddings.extend(batch_embeddings)
            
            # Update progress
            progress = min((i + batch_size) / len(texts), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing Ara-E5 embeddings: {i + len(batch)}/{len(texts)} chunks")
        
        progress_bar.empty()
        status_text.empty()
        
        print(f"Generated {len(embeddings)} embeddings with Ara-E5 model")
        
        # Return the Ara-E5 wrapper model
        arae5_wrapper = AraE5EmbeddingModel(tokenizer, model)
        return arae5_wrapper, embeddings
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error generating embeddings: {e}")
        return None, []

def store_document_and_chunks(filename, file_hash, file_size, texts, embeddings):
    """Store document and its chunks with embeddings in PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            model_info = st.session_state.model_info
            
            # Insert document with model information
            cur.execute("""
                INSERT INTO documents (filename, file_hash, file_size, embedding_model, embedding_dimensions) 
                VALUES (%s, %s, %s, %s, %s) RETURNING id;
            """, (filename, file_hash, file_size, model_info['name'], model_info['dimensions']))
            document_id = cur.fetchone()[0]
            
            # Insert chunks with embeddings in batches
            batch_size = 50  # Smaller batches for large embeddings
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                    cur.execute("""
                        INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s);
                    """, (document_id, text, i + j, embedding_list))
                
                conn.commit()  # Commit in batches for reliability
            
            print(f"Stored Arabic document with ID {document_id} and {len(texts)} chunks using Ara-E5")
            return document_id
    except psycopg2.Error as e:
        st.error(f"Error storing document: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def search_similar_chunks_multi_docs(query, document_ids: List[int], embedding_model, top_k=7):
    """Search for similar chunks across multiple documents using Ara-E5 embeddings."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        # Generate query embedding using Ara-E5 with proper query preprocessing
        query_embedding = embedding_model.encode_query(query)
        query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Search across multiple documents
            placeholders = ','.join(['%s'] * len(document_ids))
            
            cur.execute(f"""
                SELECT dc.chunk_text, 
                       dc.chunk_index,
                       d.filename,
                       d.embedding_model,
                       dc.embedding <=> %s::vector as similarity_score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.document_id IN ({placeholders})
                AND dc.embedding <=> %s::vector < 0.85
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s;
            """, [query_embedding_list] + document_ids + [query_embedding_list, query_embedding_list, top_k])
            
            results = cur.fetchall()
            
            if results:
                print(f"Found {len(results)} relevant chunks across {len(document_ids)} documents using Ara-E5:")
                chunks_with_source = []
                for i, result in enumerate(results):
                    print(f"Chunk {i+1} from {result['filename']}: Score {result['similarity_score']:.3f}")
                    # Add source information to chunk
                    chunk_with_source = f"[Source: {result['filename']} | Model: {result['embedding_model']}]\n{result['chunk_text']}"
                    chunks_with_source.append(chunk_with_source)
                
                return chunks_with_source
            else:
                print("No chunks found within similarity threshold, trying broader search...")
                # Fallback: get top chunks without threshold
                cur.execute(f"""
                    SELECT dc.chunk_text, 
                           dc.chunk_index,
                           d.filename,
                           d.embedding_model,
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
                        chunk_with_source = f"[Source: {result['filename']} | Model: {result['embedding_model']}]\n{result['chunk_text']}"
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
    """Search for similar chunks in a single document using Ara-E5."""
    return search_similar_chunks_multi_docs(query, [document_id], embedding_model, top_k)

def generate_response(query, retrieved_chunks, is_multi_doc=False):
    """Generate response using OpenAI API with retrieved chunks - Enhanced for Arabic with Ara-E5."""
    if not retrieved_chunks or retrieved_chunks == ["No relevant results found."] or retrieved_chunks == ["Error occurred during search."]:
        return json.dumps([{"type": "text", "content": "لم يتم العثور على معلومات ذات صلة في المستند(ات) للإجابة على سؤالك. / No relevant information found in the document(s) to answer your question."}])

    try:
        # Combine chunks into context
        context = "\n\n".join(retrieved_chunks)
        
        # Enhanced system prompt for Arabic support with Ara-E5
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
        
        system_prompt = f"""You are an intelligent PDF document assistant specialized in Arabic and English content using the Ara-E5-Base model for superior Arabic text understanding. Your job is to analyze PDF content and provide helpful responses in a specific JSON format.

{multi_doc_instruction}

MODEL CONTEXT:
- Using gimmeursocks/ara-e5-base model for Arabic text embeddings
- This model provides enhanced Arabic semantic understanding and context preservation
- E5 models use query/passage preprocessing for optimal similarity matching
- 768-dimensional embeddings optimized for Arabic language processing

LANGUAGE HANDLING:
- Detect query language (Arabic/English) and respond in the same language
- For Arabic queries, provide comprehensive Arabic responses with technical terms when needed
- For English queries, provide detailed English responses
- Always maintain technical accuracy regardless of language
- Leverage Ara-E5's superior Arabic semantic understanding

RESPONSE TYPE DETERMINATION:

1. VISUALIZATION GENERATION (when user requests charts, graphs, plots, or visual analysis):
- Generate matplotlib code for data visualization
- Extract relevant data from the PDF content for plotting
- Create meaningful charts based on available information
- Add Arabic labels and support Arabic fonts when query is in Arabic
- Use proper Arabic font configuration: plt.rcParams['font.family'] = ['Tahoma', 'Arial Unicode MS', 'DejaVu Sans']

2. TABLE GENERATION (when user requests tables, lists, summaries, or structured data):
- Extract and structure relevant information from PDF content
- Create comprehensive tables with meaningful data organization
- Format data in a clear, readable structure
- Use Arabic headers and content when query is in Arabic

3. TEXT RESPONSES (for explanations, insights, and general queries):
- Provide clear, comprehensive answers in the query language
- Include detailed insights and analysis of the PDF content
- Reference specific document sources when applicable
- Leverage Ara-E5's superior Arabic understanding for nuanced responses

MANDATORY JSON RESPONSE FORMAT (ABSOLUTE REQUIREMENT):
ENTIRE response must be valid JSON array - NO exceptions:

[
    {{"type": "text", "content": "Detailed explanatory text in query language, insights, context, assumptions, or business implications based on the PDF content. Leverage Ara-E5's enhanced Arabic understanding."}},
    {{"type": "img", "content": "import matplotlib.pyplot as plt\\nimport pandas as pd\\nimport numpy as np\\n\\n# Configure Arabic font support\\nplt.rcParams['font.family'] = ['Tahoma', 'Arial Unicode MS', 'DejaVu Sans']\\nplt.rcParams['axes.unicode_minus'] = False\\n\\n# Complete executable visualization code based on PDF data\\nplt.figure(figsize=(12, 8))\\n# Visualization implementation with Arabic support\\nplt.title('Chart Title Based on PDF Content')\\nplt.xlabel('X Label')\\nplt.ylabel('Y Label')\\nplt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\\nplt.tight_layout(pad=3.0)\\nplt.show()"}},
    {{"type": "table", "content": "import pandas as pd\\nimport json\\n\\n# Extract and structure data from PDF content\\ndata = [\\n    {{'Item': 'Example 1', 'Value': 'Data from PDF'}},\\n    {{'Item': 'Example 2', 'Value': 'More PDF data'}}\\n]\\nresult_df = pd.DataFrame(data)\\nprint(result_df.to_json(orient='records', date_format='iso'))"}},
    {{"type": "text", "content": "Additional insights, conclusions, or summary in query language based on the Ara-E5-powered PDF analysis"}}
]

JSON FORMAT RULES (STRICTLY ENFORCED):
- Valid JSON only - no text outside structure
- NO markdown blocks (```python) - code goes directly in "content"
- Every content piece requires JSON object with "type" and "content"
- Line breaks: \\n, Quotes: \\"
- Multiple images/tables allowed and encouraged
- "img" type: complete runnable Python visualization code with Arabic font support
- "table" type: code outputting JSON via print(result_df.to_json(orient='records', date_format='iso'))
- "text" type: ONLY direct findings and business insights - NO code explanations
- Extract actual data from PDF content for visualizations and tables
- Support Arabic text rendering in visualizations when needed

Remember: You are using the gimmeursocks/ara-e5-base model for superior Arabic text understanding and semantic search. Extract maximum value from Arabic/English PDF content and present it in the requested format."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"""Based on the following PDF document excerpts (processed with Ara-E5-Base embeddings from gimmeursocks/ara-e5-base model), please answer this question: "{query}"

Document Content:
{context}

Please provide a comprehensive response in the required JSON format. Extract relevant data from the PDF content to create meaningful visualizations or tables if requested. Respond in the same language as the query. Leverage the superior Arabic understanding provided by Ara-E5-Base."""
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

def execute_visualization_code(code):
    """Execute visualization code and capture the plot with Arabic font support."""
    try:
        # Set up Arabic font support for Ara-E5-powered visualizations
        plt.rcParams['font.family'] = ['Tahoma', 'Arial Unicode MS', 'DejaVu Sans']
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
        
        # Check for syntax errors first
        try:
            ast.parse(code)
        except SyntaxError as syntax_error:
            st.error(f"Syntax error in generated table code: {syntax_error}")
            return None
        
        # Create a safe execution environment
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
                '__import__': __import__,
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
                st.text("Raw output:")
                st.text(table_output)
                return None
        return None
    except Exception as e:
        st.error(f"Error executing table code: {e}")
        with st.expander("View problematic code"):
            st.code(code, language="python")
        return None
    
def get_uploaded_documents():
    """Get list of uploaded documents from database with file sizes and model info."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, filename, file_size, embedding_model, embedding_dimensions, upload_timestamp 
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
    st.success("Database initialized successfully! (Using Ara-E5-Base)")
else:
    st.error("Failed to initialize database. Please check your PostgreSQL connection.")

# Initialize Ara-E5 embedding model
if st.session_state.arae5_tokenizer is None or st.session_state.arae5_model is None:
    with st.spinner("Loading Ara-E5-Base model (gimmeursocks/ara-e5-base) for Arabic documents..."):
        tokenizer, model, model_info = init_arae5_model()
        if tokenizer and model and model_info:
            st.session_state.arae5_tokenizer = tokenizer
            st.session_state.arae5_model = model
            st.session_state.model_info = model_info
            st.success(f"Ara-E5 model loaded successfully: {model_info['name']}")
        else:
            st.error("Failed to initialize Ara-E5 model. Please check your internet connection and model availability.")

# Sidebar for file upload and document management
with st.sidebar:
    st.header("إدارة المستندات / Document Management")
    
    # Display current model info
    if st.session_state.model_info:
        model_info = st.session_state.model_info
        st.info(f"🤖 Model: {model_info['name']}\n📊 Dimensions: {model_info['dimensions']}")
    
    # File uploader with Arabic/English help text
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
                st.info("Document already exists in database! / المستند موجود بالفعل في قاعدة البيانات!")
                if existing_doc_id not in st.session_state.selected_document_ids:
                    st.session_state.selected_document_ids = [existing_doc_id]
            else:
                # Check if Ara-E5 model is loaded
                if st.session_state.arae5_tokenizer is None or st.session_state.arae5_model is None:
                    st.error("Ara-E5 model not loaded. Please refresh the page.")
                else:
                    with st.spinner("Processing Arabic PDF with Ara-E5... / معالجة ملف PDF العربي باستخدام Ara-E5..."):
                        texts = load_and_chunk_pdf(uploaded_file)
                        if texts:
                            embedding_model, embeddings = generate_embeddings_with_arae5(texts)
                            if embedding_model and embeddings:
                                document_id = store_document_and_chunks(
                                    uploaded_file.name, file_hash, uploaded_file.size, texts, embeddings
                                )
                                if document_id:
                                    st.session_state.selected_document_ids = [document_id]
                                    st.success(f"{uploaded_file.name} processed successfully with Ara-E5! / تمت معالجة الملف بنجاح!")
                                else:
                                    st.error("Failed to store document in database. / فشل في حفظ المستند في قاعدة البيانات.")
                            else:
                                st.error("Failed to generate embeddings with Ara-E5.")
    
    st.divider()
    
    # Chat mode selection with Arabic text
    st.subheader("وضع الاستعلام / Query Mode")
    chat_mode = st.radio(
        "Choose how to search / اختر طريقة البحث:",
        ["Single Document / مستند واحد", "Multiple Documents / مستندات متعددة"],
        index=0 if st.session_state.chat_mode == "single" else 1
    )
    st.session_state.chat_mode = "single" if "Single" in chat_mode else "multi"
    
    # Document selector with Arabic text
    st.subheader("المستندات المتاحة / Available Documents")
    documents = get_uploaded_documents()
    
    if documents:
        if st.session_state.chat_mode == "single":
            # Single document selection
            doc_options = {f"{doc['filename']} ({format_file_size(doc['file_size'])}) - {doc['embedding_model']}": doc['id'] 
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
                doc_name = f"{doc['filename']} ({format_file_size(doc['file_size'])}) - {doc['embedding_model']}"
                if st.checkbox(doc_name, value=doc['id'] in st.session_state.selected_document_ids, key=f"doc_{doc['id']}"):
                    selected_docs.append(doc['id'])
            
            st.session_state.selected_document_ids = selected_docs
            
            if selected_docs:
                st.success(f"Selected {len(selected_docs)} document(s) for multi-document search / تم اختيار {len(selected_docs)} مستند للبحث")
        
        # Delete document functionality with Arabic text
        st.divider()
        st.subheader("حذف المستندات / Delete Documents")
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
    
    # Chart generation info with Arabic examples
    st.divider()
    st.subheader("التحليل المتقدم / Enhanced Analysis")
    st.info("""Ask for / اطلب:
• 'Create a chart of sales data' / 'أنشئ مخططًا لبيانات المبيعات'
• 'Show me a table of financial info' / 'أرني جدولًا للمعلومات المالية'  
• 'Visualize the trends' / 'تصور الاتجاهات'
• 'Extract key metrics in a table' / 'استخرج المقاييس الرئيسية في جدول'""")

# Main chat interface with Arabic support
st.title("تحدث مع ملفات PDF / Chat with your PDF(s)")
if st.session_state.chat_mode == "multi":
    st.caption("Multi-document RAG with Ara-E5-Base (gimmeursocks/ara-e5-base)")
else:
    st.caption("Single-document RAG with Ara-E5-Base (gimmeursocks/ara-e5-base)")

if "messages" not in st.session_state:
    # Updated initial message with Ara-E5-Base model info
    st.session_state["messages"] = [
        {"role": "assistant", "content": """Upload PDF documents (up to 1GB each) and start asking questions in Arabic or English! 

ارفع مستندات PDF (حتى 1 جيجابايت لكل منها) وابدأ في طرح الأسئلة باللغة العربية أو الإنجليزية!

I'm now using the Ara-E5-Base (gimmeursocks/ara-e5-base) model specifically designed for Arabic text understanding. This model provides:
- Superior Arabic semantic search capabilities
- Better understanding of Arabic context and nuances
- Optimized 768-dimensional embeddings for Arabic text
- Enhanced multilingual support with E5 architecture
- Query/passage preprocessing for optimal similarity matching

أستخدم الآن نموذج Ara-E5-Base (gimmeursocks/ara-e5-base) المصمم خصيصًا لفهم النصوص العربية. يوفر هذا النموذج:
- قدرات بحث دلالي متفوقة للعربية
- فهم أفضل للسياق والفروق الدقيقة العربية
- تضمينات محسنة بـ 768 بعد للنصوص العربية
- دعم محسن متعدد اللغات مع هندسة E5
- معالجة مسبقة للاستعلامات والمقاطع للحصول على أفضل مطابقة تشابه"""}
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

# Chat input with Arabic placeholder
if prompt := st.chat_input("Ask a question in Arabic or English / اطرح سؤالًا باللغة العربية أو الإنجليزية..."):
    if not st.session_state.selected_document_ids:
        st.warning("Please upload and select document(s) first. / يرجى رفع واختيار المستند(ات) أولاً.")
    elif st.session_state.arae5_tokenizer is None or st.session_state.arae5_model is None:
        st.error("Ara-E5 model not loaded. Please refresh the page.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Create Ara-E5 wrapper for search
        arae5_wrapper = AraE5EmbeddingModel(st.session_state.arae5_tokenizer, st.session_state.arae5_model)
        
        # Search for relevant chunks
        is_multi_doc = len(st.session_state.selected_document_ids) > 1
        search_msg = f"Searching across {len(st.session_state.selected_document_ids)} document(s) with Ara-E5... / البحث في {len(st.session_state.selected_document_ids)} مستند باستخدام Ara-E5..." if is_multi_doc else "Searching document with Ara-E5... / البحث في المستند باستخدام Ara-E5..."
        
        with st.spinner(search_msg):
            if is_multi_doc:
                retrieved_chunks = search_similar_chunks_multi_docs(
                    prompt, 
                    st.session_state.selected_document_ids, 
                    arae5_wrapper,
                    top_k=10  # More chunks for multi-doc search
                )
            else:
                retrieved_chunks = search_similar_chunks_single_doc(
                    prompt, 
                    st.session_state.selected_document_ids[0], 
                    arae5_wrapper
                )
        
        print("=== ARA-E5-BASE SEARCH RESULTS ===")
        print(f"Model: {st.session_state.model_info['name']}")
        print(f"Query: {prompt}")
        print(f"Mode: {'Multi-doc' if is_multi_doc else 'Single-doc'}")
        print(f"Documents: {st.session_state.selected_document_ids}")
        print(f"Retrieved chunks: {len(retrieved_chunks)}")
        print("==================================")
        
        # Generate JSON formatted response
        with st.spinner("Generating comprehensive analysis with Ara-E5... / إنشاء تحليل شامل باستخدام Ara-E5..."):
            json_response = generate_response(prompt, retrieved_chunks, is_multi_doc)
            
            # Parse and execute the JSON response
            try:
                response_data = json.loads(json_response)
                response_content = ""
                
                for item in response_data:
                    item_type = item.get("type", "")
                    content = item.get("content", "")
                    
                    if item_type == "text":
                        st.chat_message("assistant").write(content)
                        response_content += content + "\n\n"
                    
                    elif item_type == "img":
                        try:
                            st.chat_message("assistant").write("Generating visualization with Ara-E5 insights... / إنشاء مخطط بصري مع رؤى Ara-E5...")
                            # Execute visualization code
                            exec(content)
                            st.pyplot(plt.gcf())
                            plt.close()
                            response_content += "[Visualization Generated with Ara-E5 / تم إنشاء المخطط باستخدام Ara-E5]\n\n"
                        except Exception as e:
                            st.chat_message("assistant").write(f"Error generating visualization: {e}")
                            response_content += f"Error generating visualization: {e}\n\n"
                    
                    elif item_type == "table":
                        try:
                            st.chat_message("assistant").write("Generating table with Ara-E5 insights... / إنشاء جدول مع رؤى Ara-E5...")
                            table_data = execute_table_code(content)
                            if table_data:
                                df_display = pd.DataFrame(table_data)
                                st.dataframe(df_display, use_container_width=True)
                                response_content += "[Table Generated with Ara-E5 / تم إنشاء الجدول باستخدام Ara-E5]\n\n"
                            else:
                                st.chat_message("assistant").write("No table data generated. / لم يتم إنشاء بيانات الجدول.")
                                response_content += "No table data generated.\n\n"
                        except Exception as e:
                            st.chat_message("assistant").write(f"Error generating table: {e}")
                            response_content += f"Error generating table: {e}\n\n"
                
                # Store the response in session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_content.strip() if response_content.strip() else "Analysis completed with Ara-E5. / تم إكمال التحليل باستخدام Ara-E5."
                })
                
            except json.JSONDecodeError:
                # Fallback for non-JSON responses
                st.chat_message("assistant").write(json_response)
                st.session_state.messages.append({"role": "assistant", "content": json_response})

# Display current document info with Arabic text
if st.session_state.selected_document_ids:
    with st.expander("معلومات الاختيار الحالي / Current Selection Info"):
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    placeholders = ','.join(['%s'] * len(st.session_state.selected_document_ids))
                    cur.execute(f"""
                        SELECT d.filename, d.file_size, d.embedding_model, d.embedding_dimensions, 
                               d.upload_timestamp, COUNT(c.id) as chunk_count
                        FROM documents d
                        LEFT JOIN document_chunks c ON d.id = c.document_id
                        WHERE d.id IN ({placeholders})
                        GROUP BY d.id, d.filename, d.file_size, d.embedding_model, d.embedding_dimensions, d.upload_timestamp
                        ORDER BY d.upload_timestamp DESC;
                    """, st.session_state.selected_document_ids)
                    doc_infos = cur.fetchall()
                    
                    if doc_infos:
                        total_chunks = sum(doc['chunk_count'] for doc in doc_infos)
                        total_size = sum(doc['file_size'] for doc in doc_infos)
                        
                        st.write(f"**Embedding Model / نموذج التضمين:** {st.session_state.model_info['name']}")
                        st.write(f"**Dimensions / الأبعاد:** {st.session_state.model_info['dimensions']}")
                        st.write(f"**Device / الجهاز:** {DEVICE}")
                        st.write(f"**Mode / الوضع:** {'Multi-document / متعدد المستندات' if len(doc_infos) > 1 else 'Single-document / مستند واحد'}")
                        st.write(f"**Total Documents / إجمالي المستندات:** {len(doc_infos)}")
                        st.write(f"**Total Size / الحجم الإجمالي:** {format_file_size(total_size)}")
                        st.write(f"**Total Chunks / إجمالي القطع:** {total_chunks}")
                        
                        st.write("**Documents / المستندات:**")
                        for doc in doc_infos:
                            st.write(f"- {doc['filename']} ({format_file_size(doc['file_size'])}) - {doc['chunk_count']} chunks - Model: {doc['embedding_model']}")
            except psycopg2.Error as e:
                st.error(f"Error fetching document info: {e}")
            finally:
                conn.close()

# Add footer with Ara-E5-Base-specific examples
st.markdown("---")
with st.expander("أمثلة التحليل المتقدم مع Ara-E5-Base / Enhanced Analysis Examples with Ara-E5-Base"):
    st.markdown("""
    **Try these enhanced requests with Ara-E5-Base / جرب هذه الطلبات المتقدمة مع Ara-E5-Base:**
    
    **Data Visualization / تصور البيانات:** 
    - "أنشئ مخططًا يوضح الاتجاهات المالية" / "Create a chart showing the financial trends"
    - "تصور بيانات أداء المبيعات" / "Visualize the sales performance data"
    - "أنشئ رسمًا بيانيًا للنتائج الفصلية" / "Generate a graph of quarterly results"
    
    **Structured Tables / الجداول المنظمة:** 
    - "استخرج البيانات المالية في جدول" / "Extract the financial data into a table"
    - "أرني جدول ملخص للمقاييس الرئيسية" / "Show me a summary table of key metrics"
    - "أنشئ جدول مقارنة" / "Create a comparison table"
    
    **Comprehensive Analysis / التحليل الشامل:** 
    - "حلل اتجاهات السوق وأظهرها بصريًا" / "Analyze the market trends and show them visually"
    - "استخرج الرؤى وأنشئ المخططات" / "Extract insights and create charts"
    - "قدم تحليلًا كاملًا مع الجداول" / "Provide complete analysis with tables"
    
    **Smart Insights / الرؤى الذكية:** 
    - "ما هي النتائج الرئيسية؟" / "What are the key findings?"
    - "لخص المقاييس المهمة" / "Summarize the important metrics"
    - "أرني أهم نقاط البيانات" / "Show me the most important data points"
    
    **Ara-E5-Base Model Advantages / مزايا نموذج Ara-E5-Base:**
    - **Specialized Arabic Understanding** / فهم متخصص للعربية: Built specifically for Arabic text processing
    - **E5 Architecture Benefits** / مزايا هندسة E5: Advanced query/passage preprocessing for optimal matching
    - **768-dimensional Embeddings** / تضمينات 768 بعد: High-precision vector representations
    - **Arabic Context Preservation** / الحفاظ على السياق العربي: Maintains meaning across text chunks
    - **Optimized Performance** / أداء محسن: Faster and more accurate Arabic text processing
    - **Semantic Search Excellence** / تميز البحث الدلالي: Superior understanding of Arabic nuances and context
    """)
    
# Add requirements and model info
st.markdown("---")
st.info(f"""
**Ara-E5-Base Model Requirements / متطلبات نموذج Ara-E5-Base:**

**Current Model:** `{st.session_state.model_info.get('name', 'gimmeursocks/ara-e5-base') if st.session_state.model_info else 'gimmeursocks/ara-e5-base'}`
**Dimensions:** {st.session_state.model_info.get('dimensions', EMBEDDING_DIMENSIONS) if st.session_state.model_info else EMBEDDING_DIMENSIONS}
**Device:** {DEVICE}

Install these packages / قم بتثبيت هذه الحزم:
```bash
pip install torch transformers accelerate
pip install langchain langchain-community
pip install streamlit psycopg2-binary python-dotenv
pip install matplotlib pandas numpy openai
```

**Model Features / ميزات النموذج:**
- Specialized Arabic text embedding model by gimmeursocks
- E5 architecture with query/passage preprocessing
- 768-dimensional embeddings for high precision
- Superior performance on Arabic NLP tasks
- Optimized for Arabic semantic understanding and retrieval

**Environment Variables Required / متغيرات البيئة المطلوبة:**
```env
OPENAI_API_KEY=your_openai_api_key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pdf_chatbot
DB_USER=postgres
DB_PASSWORD=your_db_password
```

**Database Setup / إعداد قاعدة البيانات:**
1. Install PostgreSQL with pgvector extension
2. Create database named 'pdf_chatbot'
3. Ensure pgvector extension is available
4. The app will automatically create required tables with flexible dimensions

**Usage Tips / نصائح الاستخدام:**
- Use Arabic queries for best results with Arabic documents
- The model automatically applies E5 preprocessing (query: / passage:)
- Supports both single and multi-document search
- Optimized chunking strategy for Arabic text preservation
""")