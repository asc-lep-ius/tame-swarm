from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
# Qwen 1.5-1.8B is an excellent, smart, and tiny model that fits easily in 8GB VRAM
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat" 

app = FastAPI(title="TAME Cortex: Agential Swarm Node")

print(f"Loading {MODEL_ID}... (Simulating 'Gestational' Phase)")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load Model with Low Memory Footprint (fp16)
# We use float16 to ensure we have VRAM left over for the "Steering Vectors" later
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="auto", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)

print("Cortex Online. Ready for Homeostatic Regulation.")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
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
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))