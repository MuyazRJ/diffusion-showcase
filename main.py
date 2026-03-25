from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import base64, io

app = FastAPI()

class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def frames_to_gif_b64(frames: list) -> str:
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=60, loop=0)
    return base64.b64encode(buf.getvalue()).decode()

def make_placeholder_mnist():
    rng = np.random.default_rng()
    final = rng.integers(20, 200, (28, 28), dtype=np.uint8)
    frames = []
    for i in range(50):
        t = (i / 49) ** 2
        noise = rng.integers(0, 255, (28, 28), dtype=np.uint8)
        blended = (final * t + noise * (1 - t)).astype(np.uint8)
        frame = Image.fromarray(blended, mode="L").resize((280, 280), Image.NEAREST).convert("RGB")
        frames.append(frame)
    return Image.fromarray(final, mode="L").resize((280, 280), Image.NEAREST), frames

def make_placeholder_cifar():
    rng = np.random.default_rng()
    final = rng.integers(30, 225, (32, 32, 3), dtype=np.uint8)
    frames = []
    for i in range(50):
        t = (i / 49) ** 2
        noise = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        blended = (final * t + noise * (1 - t)).astype(np.uint8)
        frame = Image.fromarray(blended).resize((280, 280), Image.NEAREST)
        frames.append(frame)
    return Image.fromarray(final).resize((280, 280), Image.NEAREST), frames

def make_placeholder_cub():
    rng = np.random.default_rng()
    final = rng.integers(30, 225, (256, 256, 3), dtype=np.uint8)
    frames = []
    for i in range(50):
        t = (i / 49) ** 2
        noise = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        blended = (final * t + noise * (1 - t)).astype(np.uint8)
        frames.append(Image.fromarray(blended))
    return Image.fromarray(final), frames

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()

@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.model == "mnist":
        img, frames = make_placeholder_mnist()
    elif req.model == "cifar":
        img, frames = make_placeholder_cifar()
    else:
        img, frames = make_placeholder_cub()
    return {"image": img_to_b64(img), "process": frames_to_gif_b64(frames)}