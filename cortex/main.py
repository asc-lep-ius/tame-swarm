from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-8B"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TAME Cortex: Agential Swarm Node")

logger.info(f"Loading {MODEL_ID}... (Simulating 'Gestational' Phase)")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

logger.info("Cortex Online. Ready for Homeostatic Regulation.")



class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Future TAME inputs:
    # steering_vector: list[float] = None
    # active_inference_mode: bool = False

@app.get("/health")
def health_check():
    return {"status": "alive", "gpu": torch.cuda.get_device_name(0)}

@app.post("/generate")
async def generate(req: PromptRequest):
    """
    Standard generation endpoint. 
    In the TAME architecture, this is the raw 'reflex' of the organism.
    """
    try:
        # 1. Tokenize
        messages = [{"role": "user", "content": req.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 2. Generate (The Thought Process)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_k=50,
            top_p=0.95
        )
        
        # 3. Decode
        # We slice [inputs.input_ids.shape[1]:] to remove the prompt from the output
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return {
            "response": response, 
            "usage": {"input_tokens": inputs.input_ids.shape[1], "output_tokens": len(generated_ids)}
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))