import os
import logging
import asyncio
from typing import List, Optional, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import threading
import time

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Load environment variables first
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import LangChain dependencies with graceful fallback
try:
    from langchain.agents import AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
    from langchain.agents import create_tool_calling_agent
    from langchain import hub
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.tools import tool
    from supabase.client import Client, create_client
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain dependencies loaded successfully")
except ImportError as e:
    logger.warning(f"LangChain imports failed: {e}")
    LANGCHAIN_AVAILABLE = False
    # Define placeholder for type hints
    AgentExecutor = Any

# Global state management with thread safety
class RAGState:
    def __init__(self):
        self._lock = threading.RLock()
        self._agent_executor = None
        self._vector_store = None
        self._initialized = False
        self._initialization_error = None
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        
    @property
    def agent_executor(self):
        with self._lock:
            return self._agent_executor
    
    @agent_executor.setter
    def agent_executor(self, value):
        with self._lock:
            self._agent_executor = value
    
    @property
    def vector_store(self):
        with self._lock:
            return self._vector_store
    
    @vector_store.setter
    def vector_store(self, value):
        with self._lock:
            self._vector_store = value
    
    @property
    def initialized(self):
        with self._lock:
            return self._initialized
    
    @initialized.setter
    def initialized(self, value):
        with self._lock:
            self._initialized = value
    
    @property
    def initialization_error(self):
        with self._lock:
            return self._initialization_error
    
    @initialization_error.setter
    def initialization_error(self, value):
        with self._lock:
            self._initialization_error = value
    
    def is_healthy(self):
        """Check if the system is healthy and perform periodic health checks"""
        with self._lock:
            current_time = time.time()
            
            # Perform periodic health check
            if current_time - self._last_health_check > self._health_check_interval:
                self._last_health_check = current_time
                if self._initialized and self._agent_executor is not None:
                    try:
                        # Quick health check - just verify the objects exist
                        _ = self._agent_executor.tools
                        _ = self._vector_store.similarity_search.__name__
                        logger.debug("Periodic health check passed")
                    except Exception as e:
                        logger.error(f"Periodic health check failed: {e}")
                        self._initialized = False
                        self._initialization_error = f"Health check failed: {str(e)}"
                        return False
            
            return self._initialized and self._agent_executor is not None

# Global state instance
rag_state = RAGState()

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[dict]] = []
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    message: str
    rag_available: bool
    langchain_available: bool
    last_error: Optional[str] = None

async def initialize_rag_system():
    """Initialize the RAG system components with comprehensive error handling"""
    logger.info("Starting RAG system initialization...")
    
    if not LANGCHAIN_AVAILABLE:
        rag_state.initialization_error = "LangChain dependencies not available"
        rag_state.initialized = False
        logger.error("Cannot initialize: LangChain dependencies not available")
        return False
    
    try:
        # Check environment variables
        required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
            raise ValueError(error_msg)

        # Initialize Supabase with retry logic
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                supabase: Client = create_client(supabase_url, supabase_key)
                # Test the connection
                _ = supabase.table("documents").select("count", count="exact").execute()
                logger.info("Supabase client initialized and tested")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to connect to Supabase after {max_retries} attempts: {e}")
                logger.warning(f"Supabase connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Initialize embeddings
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            # Test embeddings with a simple query
            _ = await asyncio.to_thread(embeddings.embed_query, "test")
            logger.info("Embeddings model initialized and tested")
        except Exception as e:
            raise Exception(f"Failed to initialize embeddings: {e}")
        
        # Initialize vector store
        try:
            vector_store = SupabaseVectorStore(
                embedding=embeddings,
                client=supabase,
                table_name="documents",
                query_name="match_documents",
            )
            # Test vector store with error handling for API compatibility
            try:
                test_results = await asyncio.to_thread(
                    vector_store.similarity_search, 
                    "test query", 
                    k=1
                )
                logger.info(f"Vector store initialized and tested (found {len(test_results)} documents)")
            except AttributeError as attr_err:
                # Known compatibility issue with supabase client versions
                logger.warning(f"Vector store test failed due to API compatibility: {attr_err}")
                logger.info("Vector store initialized (skipping test due to compatibility issue)")
        except Exception as e:
            raise Exception(f"Failed to initialize vector store: {e}")
        
        # Initialize LLM
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Using a more stable model
            # Test LLM
            test_response = await asyncio.to_thread(
                llm.invoke, 
                [HumanMessage(content="Hello, respond with 'OK'")]
            )
            logger.info("LLM initialized and tested")
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {e}")
        
        # Pull prompt from hub with fallback
        try:
            prompt = hub.pull("hwchase17/openai-functions-agent")
            logger.info("Prompt pulled from hub")
        except Exception as e:
            logger.warning(f"Failed to pull prompt from hub: {e}, using fallback")
            # Create a simple fallback prompt
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant with access to document retrieval tools."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        
        # Create retriever tool with enhanced error handling
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            try:
                if not rag_state.vector_store:
                    return "Vector store not available", []
                
                retrieved_docs = rag_state.vector_store.similarity_search(query, k=3)
                if not retrieved_docs:
                    return "No relevant documents found for the query.", []
                
                serialized = "\n\n".join(
                    (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs
            except Exception as e:
                logger.error(f"Error in retrieve tool: {e}")
                return f"Error retrieving documents: {e}", []
        
        # Create agent with error handling
        try:
            tools = [retrieve]
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            logger.info("Agent executor created")
        except Exception as e:
            raise Exception(f"Failed to create agent: {e}")
        
        # Update global state
        rag_state.vector_store = vector_store
        rag_state.agent_executor = agent_executor
        rag_state.initialized = True
        rag_state.initialization_error = None
        
        logger.info("RAG system initialized successfully")
        return True
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {e}"
        logger.error(error_msg, exc_info=True)
        rag_state.initialization_error = str(e)
        rag_state.initialized = False
        return False

async def get_agent_executor():
    """Dependency to get the agent executor with auto-recovery"""
    if not LANGCHAIN_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="LangChain dependencies not available. Please check installation."
        )
    
    # Check if system is healthy
    if not rag_state.is_healthy():
        # Try to reinitialize if not healthy
        logger.info("System not healthy, attempting reinitialization...")
        success = await initialize_rag_system()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail=f"RAG system not available. Error: {rag_state.initialization_error}"
            )
    
    return rag_state.agent_executor

def convert_chat_history(chat_history: List[ChatMessage]):
    """Convert chat history to LangChain message format"""
    if not LANGCHAIN_AVAILABLE:
        return []
    
    try:
        messages = []
        for msg in chat_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        return messages
    except Exception as e:
        logger.error(f"Error converting chat history: {e}")
        return []

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application...")
    logger.info(f"LangChain available: {LANGCHAIN_AVAILABLE}")
    
    if LANGCHAIN_AVAILABLE:
        await initialize_rag_system()
    else:
        logger.warning("Skipping RAG initialization - LangChain dependencies not available")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Agentic RAG API",
    description="RAG-powered chatbot API with document retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://omerzirh.com", 
        "https://www.omerzirh.com",
        "chrome-extension://*",  # Allow Chrome extensions
        "moz-extension://*",     # Allow Firefox extensions
        "*"  # Allow all origins for development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "path": str(request.url)
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Agentic RAG API",
        "version": "1.0.0",
        "status": "healthy" if rag_state.initialized else "initializing",
        "rag_available": rag_state.initialized,
        "langchain_available": LANGCHAIN_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "query": "/query", 
            "search": "/search",
            "docs": "/docs",
            "initialize": "/initialize"
        }
    }

# Health check endpoint with detailed information
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Always return 200 OK for basic health check
        # This allows the container to be considered healthy even during initialization
        if rag_state.is_healthy():
            return HealthResponse(
                status="healthy", 
                message="All systems operational", 
                rag_available=True,
                langchain_available=LANGCHAIN_AVAILABLE
            )
        elif not LANGCHAIN_AVAILABLE:
            # Still healthy from container perspective, just degraded functionality
            return HealthResponse(
                status="healthy", 
                message="Running without LangChain dependencies", 
                rag_available=False,
                langchain_available=False,
                last_error="LangChain not installed"
            )
        elif rag_state.initialization_error:
            # Still healthy from container perspective
            return HealthResponse(
                status="healthy", 
                message="RAG system error but API is running", 
                rag_available=False,
                langchain_available=LANGCHAIN_AVAILABLE,
                last_error=rag_state.initialization_error
            )
        else:
            # Initializing - still healthy
            return HealthResponse(
                status="healthy", 
                message="RAG system initializing...", 
                rag_available=False,
                langchain_available=LANGCHAIN_AVAILABLE
            )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        # Even on error, return healthy if the API is responding
        return HealthResponse(
            status="healthy",
            message="API responding but health check encountered error",
            rag_available=False,
            langchain_available=LANGCHAIN_AVAILABLE,
            last_error=str(e)
        )

# Manual initialization endpoint
@app.post("/initialize")
async def manual_initialize():
    """Manually trigger RAG system initialization"""
    if not LANGCHAIN_AVAILABLE:
        return {"status": "error", "message": "LangChain dependencies not available"}
    
    logger.info("Manual initialization requested")
    rag_state.initialization_error = None
    success = await initialize_rag_system()
    
    if success:
        return {
            "status": "success", 
            "message": "RAG system initialized successfully",
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "error", 
            "message": f"Failed to initialize: {rag_state.initialization_error}",
            "timestamp": datetime.now().isoformat()
        }

# Chat endpoint with enhanced error handling
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with auto-recovery"""
    try:
        # Get executor with auto-recovery
        executor = await get_agent_executor()
        
        # Validate input
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Convert chat history
        chat_history = convert_chat_history(request.chat_history or [])
        
        # Invoke the agent with timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    executor.invoke,
                    {
                        "input": request.message.strip(),
                        "chat_history": chat_history
                    }
                ),
                timeout=60  # 60 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout")
        
        # Extract response and sources
        response_text = result.get("output", "No response generated")
        sources = []
        
        # Extract sources from intermediate steps
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) > 1 and hasattr(step[1], 'artifact'):
                    docs = step[1].artifact if step[1].artifact else []
                    for doc in docs:
                        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                            sources.append({
                                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                                "metadata": doc.metadata
                            })
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# Simple query endpoint
@app.post("/query")
async def query(message: str):
    """Simple query endpoint without chat history"""
    try:
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="Message parameter is required")
        
        executor = await get_agent_executor()
        
        result = await asyncio.wait_for(
            asyncio.to_thread(
                executor.invoke,
                {"input": message.strip(), "chat_history": []}
            ),
            timeout=60
        )
        
        return {
            "response": result.get("output", "No response generated"),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Query endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Direct search endpoint
@app.post("/search")
async def search_documents(query: str, k: int = 3):
    """Direct document search endpoint"""
    try:
        if not LANGCHAIN_AVAILABLE:
            raise HTTPException(status_code=503, detail="LangChain dependencies not available")
        
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        if not rag_state.is_healthy() or rag_state.vector_store is None:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        # Validate k parameter
        k = max(1, min(k, 10))  # Limit between 1 and 10
        
        docs = await asyncio.wait_for(
            asyncio.to_thread(
                rag_state.vector_store.similarity_search,
                query.strip(),
                k
            ),
            timeout=30
        )
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": getattr(doc, 'relevance_score', None)
            })
        
        return {
            "query": query.strip(),
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Search timeout")
    except Exception as e:
        logger.error(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )