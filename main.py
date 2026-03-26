import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Jewelment API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict to 'jewelment.com' in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API Keys securely from environment
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_API_TOKEN")

class GenerateRequest(BaseModel):
    prompt: str
    image_b64: Optional[str] = None
    mime_type: Optional[str] = None

@app.post("/api/generate/gemini")
async def generate_gemini(req: GenerateRequest):
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="Gemini key not configured")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_KEY}"
    
    parts = []
    if req.image_b64 and req.mime_type:
        parts.append({"inline_data": {"mime_type": req.mime_type, "data": req.image_b64}})
    parts.append({"text": req.prompt + "\n\nGenerate a photorealistic jewellery image."})
    
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"], "temperature": 0.7}
    }
    
    async with httpx.AsyncClient() as client:
        res = await client.post(url, json=payload, timeout=60.0)
        
    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail=res.text)
        
    data = res.json()
    try:
        for p in data['candidates'][0]['content']['parts']:
            if p.get('inlineData', {}).get('mimeType', '').startswith('image/'):
                return {"url": f"data:{p['inlineData']['mimeType']};base64,{p['inlineData']['data']}"}
    except KeyError:
        raise HTTPException(status_code=500, detail="Invalid response from Gemini")

@app.post("/api/generate/hf")
async def generate_hf(req: GenerateRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF token not configured")
        
    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": req.prompt,
        "parameters": {"width": 1024, "height": 1024, "num_inference_steps": 28, "guidance_scale": 3.5}
    }
    
    async with httpx.AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload, timeout=120.0)
        
    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail=res.text)
        
    import base64
    b64_img = base64.b64encode(res.content).decode('utf-8')
    return {"url": f"data:image/jpeg;base64,{b64_img}"}
