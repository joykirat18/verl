#!/usr/bin/env python3
"""
FastAPI model host for serving the Qwen model and providing model outputs.
This service is designed to work with the redundant token eviction algorithm.
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import uvicorn
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
os.environ['HF_HOME'] = '/nas-ssd2/joykirat/.cache/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Initialize FastAPI app
app = FastAPI(
    title="Model Host API",
    description="API for serving the Qwen model with attention outputs",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

class ModelInput(BaseModel):
    """Input model for text generation requests."""
    text: str
    return_attention: Optional[bool] = True

class ModelOutput(BaseModel):
    """Output model for text generation responses."""
    generated_text: str
    input_ids: list
    attention_weights: Optional[list] = None
    model_output: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_name: str

@app.on_event("startup")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((Exception,))
)
async def startup_event():
    """Initialize the model and tokenizer on startup."""
    global model, tokenizer, device
    
    try:
        logger.info("Loading model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B", 
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        device = next(model.parameters()).device
        logger.info(f"Model loaded successfully on device: {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health information."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        model_name="Qwen/Qwen3-4B"
    )

@app.get("/health", response_model=HealthResponse)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((Exception,))
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        model_name="Qwen/Qwen3-4B"
    )



@app.post("/model_output")
@retry(
    stop=stop_after_attempt(3),                          # Reduced from 5 to 3 attempts
    wait=wait_exponential(multiplier=2, min=2, max=30),  # Wait: 2s, 4s, 8s, 16s, 30s
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
async def get_model_output(request: ModelInput):
    """Get raw model output with attention weights for the eviction algorithm."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text, 
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model output with attention weights
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Prepare response for eviction algorithm
        # Convert tensors to numpy arrays for JSON serialization
        response_data = {
            "input_ids": inputs['input_ids'][0].cpu().numpy().tolist(),
            "attention_weights": [],
        }
        
        # Convert attention tensors to numpy arrays
        if hasattr(outputs, 'attentions'):
            for layer_attentions in outputs.attentions:
                layer_array = layer_attentions.cpu().numpy()
                response_data["attention_weights"].append(layer_array.tolist())
        
        return response_data
        
    except Exception as e:
        logger.error(f"Model output error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model output: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "model_host:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 