# main.py
import os
import logging
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Langchain imports
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

# Supabase imports
from supabase.client import Client, create_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="RAG-powered chatbot API with document retrieval",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Global variables for the RAG system
agent_executor = None
vector_store = None

def initialize_rag_system():
    """Initialize the RAG system components"""
    global agent_executor, vector_store
    
    try:
        # Check environment variables
        required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

        # Initialize Supabase
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize vector store
        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
        )
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Pull prompt from hub
        prompt = hub.pull("hwchase17/openai-functions-agent")
        
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
        
        logger.info("RAG system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise e

def get_agent_executor():
    """Dependency to get the agent executor"""
    if agent_executor is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    return agent_executor

def convert_chat_history(chat_history: List[ChatMessage]):
    """Convert chat history to LangChain message format"""
    messages = []
    for msg in chat_history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    return messages

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    try:
        initialize_rag_system()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise e

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test if agent is working
        if agent_executor is None:
            return HealthResponse(status="unhealthy", message="RAG system not initialized")
        
        return HealthResponse(status="healthy", message="Service is running")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", message=str(e))

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
                    docs = step[1].artifact
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
        if vector_store is None:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
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

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Agentic RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "query": "/query",
            "search": "/search",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)