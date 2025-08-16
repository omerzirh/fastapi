import os
import logging
from typing import List, Optional, Any, Union
from datetime import datetime
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
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

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="RAG-powered chatbot API with document retrieval",
    version="1.0.0"
)

origins = [
    "https://omerzirh.com",       # production SPA
    "http://localhost:5173/",      # local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # <— no "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],         # optional: let the browser read custom headers
)


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

# Global variables for the RAG system
agent_executor = None
vector_store = None
rag_initialized = False
initialization_error = None

def initialize_rag_system():
    """Initialize the RAG system components"""
    global agent_executor, vector_store, rag_initialized, initialization_error
    
    if not LANGCHAIN_AVAILABLE:
        initialization_error = "LangChain dependencies not available"
        rag_initialized = False
        return False
    
    try:
        # Check environment variables
        required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

        logger.info("Starting RAG system initialization...")

        # Initialize Supabase
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("Embeddings model initialized")
        
        # Initialize vector store
        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
        )
        logger.info("Vector store initialized")
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        logger.info("LLM initialized")
        
        # Pull prompt from hub
        prompt = hub.pull("hwchase17/openai-functions-agent")
        logger.info("Prompt pulled from hub")
        
        # Create retriever tool
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            try:
                retrieved_docs = vector_store.similarity_search(query, k=3)
                serialized = "\n\n".join(
                    (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs
            except Exception as e:
                logger.error(f"Error in retrieve tool: {e}")
                return f"Error retrieving documents: {e}", []
        
        # Create agent
        tools = [retrieve]
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        rag_initialized = True
        initialization_error = None
        logger.info("RAG system initialized successfully")
        return True
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {e}"
        logger.error(error_msg)
        initialization_error = str(e)
        rag_initialized = False
        return False

def get_agent_executor():
    """Dependency to get the agent executor"""
    if not LANGCHAIN_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="LangChain dependencies not available. Please check installation."
        )
    
    if not rag_initialized or agent_executor is None:
        raise HTTPException(
            status_code=503, 
            detail=f"RAG system not available. Error: {initialization_error or 'System not initialized'}"
        )
    return agent_executor

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

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Agentic RAG API",
        "version": "1.0.0",
        "status": "healthy" if rag_initialized else "initializing",
        "rag_available": rag_initialized,
        "langchain_available": LANGCHAIN_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "query": "/query", 
            "search": "/search",
            "docs": "/docs",
            "initialize": "/initialize"
        }
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if rag_initialized:
        return HealthResponse(
            status="healthy", 
            message="Service is running", 
            rag_available=True,
            langchain_available=LANGCHAIN_AVAILABLE
        )
    elif not LANGCHAIN_AVAILABLE:
        return HealthResponse(
            status="degraded", 
            message="LangChain dependencies not available", 
            rag_available=False,
            langchain_available=False
        )
    elif initialization_error:
        return HealthResponse(
            status="degraded", 
            message=f"RAG system error: {initialization_error}", 
            rag_available=False,
            langchain_available=LANGCHAIN_AVAILABLE
        )
    else:
        return HealthResponse(
            status="starting", 
            message="RAG system initializing...", 
            rag_available=False,
            langchain_available=LANGCHAIN_AVAILABLE
        )

# Manual initialization endpoint
@app.post("/initialize")
async def manual_initialize():
    """Manually trigger RAG system initialization"""
    global initialization_error
    
    if not LANGCHAIN_AVAILABLE:
        return {"status": "error", "message": "LangChain dependencies not available"}
    
    initialization_error = None
    success = initialize_rag_system()
    if success:
        return {"status": "success", "message": "RAG system initialized successfully"}
    else:
        return {"status": "error", "message": f"Failed to initialize: {initialization_error}"}

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    executor: AgentExecutor = Depends(get_agent_executor)
):
    """Main chat endpoint"""
    try:
        # Convert chat history
        chat_history = convert_chat_history(request.chat_history)
        
        # Invoke the agent
        result = executor.invoke({
            "input": request.message,
            "chat_history": chat_history
        })
        
        # Extract response and sources (if available)
        response_text = result["output"]
        sources = []
        
        # Try to extract sources from intermediate steps if available
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) > 1 and hasattr(step[1], 'artifact'):
                    # Extract document sources
                    docs = step.artifact
                    for doc in docs:
                        sources.append({
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata
                        })
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# Simple query endpoint (without chat history)
@app.post("/query")
async def query(
    message: str,
    executor: AgentExecutor = Depends(get_agent_executor)
):
    """Simple query endpoint without chat history"""
    try:
        result = executor.invoke({"input": message, "chat_history": []})
        
        return {
            "response": result["output"],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Search endpoint (direct vector search)
@app.post("/search")
async def search_documents(query: str, k: int = 3):
    """Direct document search endpoint"""
    try:
        if not LANGCHAIN_AVAILABLE:
            raise HTTPException(status_code=503, detail="LangChain dependencies not available")
        
        if not rag_initialized or vector_store is None:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        docs = vector_store.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": getattr(doc, 'relevance_score', None)
            })
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

# Initialize RAG system on startup (but don't fail if it doesn't work)
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    logger.info("Starting application...")
    logger.info(f"LangChain available: {LANGCHAIN_AVAILABLE}")
    
    # Try to initialize RAG system, but don't fail startup if it doesn't work
    if LANGCHAIN_AVAILABLE:
        initialize_rag_system()
    else:
        logger.warning("Skipping RAG initialization - LangChain dependencies not available")
    
    logger.info("Application startup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
