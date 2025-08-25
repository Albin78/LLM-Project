from pydantic import BaseModel, ConfigDict, field_validator, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Request, status
from datetime import datetime
from typing_extensions import Optional, Dict, Any, Annotated
from src.RAG.run_rag_pipeline import RAGPipeline
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from src.inference_repo.inference import order_response
from pathlib import Path
import asyncio
import logging
import json
import re
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_pipeline = None

class RAG:
    """Loading RAG pipeline"""

    def __init__(self):
        self.rag = None
        self.loaded = False

    async def load_rag(self):
        """Initializing the RAG"""

        try:
            logger.info("============RAG pipeline Loading================")
            self.rag = RAGPipeline(vector_store_path=None,
                                temperature=0.8, top_p=None,
                                top_k=50)

            saved_dir = Path("src/RAG")
            if saved_dir.exists() or any(saved_dir.iterdir()):
                logger.info("Found already saved Vectore store, loading it instead of building->->->->->->->->->->->->")
                self.rag.load_vector_store(path=str(saved_dir))
            else:
                logger.info("No saved vectore store available, building knowledge base->->->->->->->->->->->->")
                self.rag.build_knowledge(document_path=r"src\data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf",
                            save_path=r"src\RAG")

            self.loaded = True
            logger.info("============RAG Pipeline Loaded successfully=============")

        except Exception as e:
            logger.error(f"Error while loading RAG Pipeline: {str(e)}")
            raise e

    async def cleanup(self):
        """Cleanup Resources"""

        try:
            logger.info("Cleanup Started.........")
            self.loaded = False

            logger.info("Cleanup Completed successfully>>>>>>")

        except Exception as e:
            logger.error(f"Error while clean up: {str(e)}")

    async def query(self, query: str, config: "RAGConfig") -> Dict[str, Any]:
        """Query format which is passed to the pipeline"""

        if not self.load_rag:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        try:
            await asyncio.sleep(1)
            result = self.rag.query_format(question=query)
            return result
        
        except Exception as e:
            logger.info(f"Error during query processing: {str(e)}")
            raise HTTPException(status_code=500, 
                                detail=f"Error while processing query: {str(e)}")
    
@asynccontextmanager
async def lifepsan(app: FastAPI):
    """Manages Application lifespans tasks"""

    logger.info("Application Starting Up ->->->->->->")

    global rag_pipeline
    rag_pipeline = RAG()
    await rag_pipeline.load_rag()
    logger.info("RAG Pipeline loaded successfully")

    yield

    logger.info("Shutting down starting .......")
    if rag_pipeline:
        await rag_pipeline.cleanup()
    
    logger.info("Shutting down completed......")

app = FastAPI(
    title="app", 
    version="2.0",
    description="A FastAPI server integrating RAG pipeline with custom LLM",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifepsan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class RAGConfig(BaseModel):
    """Configuration for the RAG pipeline"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True, 
        validate_assignment=True
    )

    max_new_tokens: int = Field(default=300, 
                        description="Maximum tokens to generate",
                        ge=10, le=500)
    
    temperature: float = Field(default=0.7,
                        description="Temperature value",
                        ge=0.1, le=1.0)
    
    top_p: Optional[float] = Field(default=0.9,
                             description="Top P value")

    top_k: Optional[int] = Field(default=40,
                            description="fetch top k tokens")

    stream: bool = Field(default=False,
                        description="Stream the response"
                        ) 
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float):
        if v < 0.0 and v > 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")

        return v
    
    @field_validator("max_new_tokens")
    @classmethod
    def validate_tokens(cls, v):
        if v < 2:
            raise ValueError("At least 2 max tokens needed")
        
        if v > 400:
            raise ValueError("Max new tokens can't exceed 400")

        return v
    
class QueryRequest(BaseModel):
    """Request to model integrated RAG"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )

    query: str = Field(..., description="User query")
    config: RAGConfig = Field(default_factory=RAGConfig, 
                        description="Configuration for RAG Pipeline"
                        )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query can't be empty")
        
        v = re.sub(r"\s+", " ", v.strip())

        if len(v) <= 2:
            raise ValueError("Please provide more than 2 words in query")
        
        return v
    
class Response(BaseModel):
    """Manages response"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    response: str = Field(..., description="Response for the query")
    metadata: Dict[str, Any] = Field(default_factory=Dict, description="Response Metadata")
    processing_time: Any = Field(..., description="Processing time for response")

    @field_validator("response")
    @classmethod
    def validate_response(cls, v):
        if not v.strip():
            raise ValueError("Empty Response..")
        
        return v

class HealthStatus(BaseModel):
    """Health status response"""

    status: str = Field(default=..., description="Status Service")
    pipeline: bool = Field(default=..., description="Pipeline status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")

    @field_validator("status")
    def validate_status(cls, v):
        validate_statuses = {"Healthy", "Unhealthy", "Initializing", "Maintanance"}
        if v not in validate_statuses:
            raise ValueError("Status must be one among:", validate_statuses)
        
        return v

class ErrorResponse(BaseModel):
    """Error Response model"""

    model_config = ConfigDict(validate_assignment=True)

    error: str = Field(..., description="Error message")
    error_code: int = Field(..., description="Error code message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

async def get_rag_pipeline() -> RAG:
    """Dependency to get RAG Pipeline"""
    global rag_pipeline
    if not rag_pipeline or not rag_pipeline.loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="RAG Pipeline not loaded")
    
    return rag_pipeline

@app.get(
    "/health",
    response_model=HealthStatus,
    tags=["Health"],
    summary="Health check",
    description="Check the health status of RAG pipeline API"
)
async def health_check():
    """Heath check of pipeline"""
    global rag_pipeline

    pipeline_status = rag_pipeline.loaded
    health_status = "Healthy" if pipeline_status else "Initializing"

    return HealthStatus(status=health_status,
                        pipeline=pipeline_status)

@app.post(
    "/query",
    response_model=Response,
    tags=["Response"],
    summary="Query Response",
    description="The response after processing query",
    responses={
        200: {"description": "Successfull query processing"},
        400: {"description": "Invalid request parameters"},
        503: {"description": "Service Unavailable"}
    }
)
async def query_process(
    request: QueryRequest,
    pipeline: Annotated[RAG, Depends(get_rag_pipeline)]
    ) -> Response:
    """Process query"""
    logger.info(f"Processing query: {request.query}")

    if request.config.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use /stream endpoint for streaming response"
        )
    
    try:
        answer = await pipeline.query(query=request.query, 
                                      config=request.config)
        result = " ".join(source['content'] for source in answer['sources'])

        return Response(
            response=order_response(result),
            metadata= answer["sources"][0]["metadata"],
            processing_time=answer['processing_time']
        )
    
    except HTTPException as e:
        logger.error(f"Unexpected error during query processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during query processing"
            )

@app.post(
    "/stream",
    response_model=Response,
     tags=["Response"],
     summary="Stream query response",
     description="Process the query response in streamable way",
     responses={
        200: {"description": "Stream response successfull"},
        400: {"description": "Invalid request parameters"},
        503: {"description": "Service Unavailable"}
     }
)
async def stream_query(
    request: QueryRequest,
    pipeline: Annotated[RAG, Depends(get_rag_pipeline)]
    ):
    """Response in streamable format"""

    logger.info(f"Streamable query {request.query}")

    async def generate_stream():
        try:
            response = await pipeline.query(query=request.query,
                                            config=request.config)
            
            if isinstance(response, dict):
                answer_text = " ".join(source['content'] for source in response['sources'])
                answer_text = order_response(answer_text)
                processing_time = response["processing_time"]
            else:
                answer_text = str(response)
            
            chunk_size = 50
            for i in range(0, len(answer_text), chunk_size):
                chunk = answer_text[i:i+chunk_size]
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.05)
            
            yield f"data: {json.dumps({'time': processing_time, 'done': False})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"data: {json.dumps({'Error': str(e), 'done': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get(
    "/pipeline/info",
    tags=["Pipeline"],
    summary="Pipeline information",
    description="Get information about the loaded RAG Pipeline"
)
async def pipeline_info():
    """RAG pipeline information"""
    global rag_pipeline

    return {
        "Pipeline_type": "Custom RAG Pipeline",
        "Status": "Loaded" if rag_pipeline.loaded else "Loaded Failed",
        "Capabilities": [
            "document retrieval",
            "text generation",
            "similarity search",
            "Streaming response"
        ]
    }

@app.exception_handler(HTTPException)
async def http_execption(exception: HTTPException):
    """Global Exception handler for HTTPException"""
    return ErrorResponse(
        error=exception.detail,
        error_code=exception.status_code
    ).model_dump()

@app.exception_handler(Exception)
async def general_exception(request: Request, exception: Exception):
    """Global Exception handler for general exceptions"""
    logger.info(f"Unexpected Error: {str(exception)}", exc_info=True)
    return ErrorResponse(
        error="Internal Server Error",
        error_code="500"
    ).model_dump()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.fastapi_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )