# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# FastAPI application for serving diffusion model image generation demos.
# Provides a lightweight web interface and API endpoints for generating images
# from multiple trained diffusion models, including MNIST, CIFAR-10, and a
# text-conditioned latent diffusion model for CUB bird image synthesis.
#
# The application loads pretrained models at startup, serves a frontend page,
# accepts generation requests through a REST API, and returns both the final
# generated image and an animated visualisation of the denoising process
# encoded in base64 format for browser display.
#

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from mnist_diffusion_unet.generate_mnist import load_mnist_model, generate_mnist
from cifar_10_diffusion.generate_cifar import load_cifar_model, generate_cifar
from latent_text_diffusion.src.generate_cub import load_cub_model, generate_cub

import numpy as np
from PIL import Image
import base64, io
import torch

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Select GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained diffusion models at application startup
mnist_model = load_mnist_model(ckpt_path="mnist_diffusion_unet/model_dict/mnist_unet.pth", device=device)
cifar_model = load_cifar_model(ckpt_path="cifar_10_diffusion/model_dict/final_ddpm.pt")
cub_model   = load_cub_model()

class GenerateRequest(BaseModel):
    # Request payload for generation API calls
    model: str
    prompt: str = ""
    steps: int = 50

def img_to_b64(img: Image.Image) -> str:
    # Convert a PIL image into a base64 PNG string for JSON transport
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def frames_to_gif_b64(frames: list) -> str:
    # Convert a list of PIL frames into an animated GIF encoded as base64
    buf = io.BytesIO()
    durations = [220] * (len(frames) - 1) + [1000]  # linger 1.5s on final frame
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=durations, loop=0)
    return base64.b64encode(buf.getvalue()).decode()

def make_placeholder_mnist():
    # Create a synthetic MNIST-like placeholder image and denoising sequence
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
    # Create a synthetic CIFAR-like placeholder image and denoising sequence
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
    # Create a synthetic CUB-like placeholder image and denoising sequence
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
    # Serve the main frontend page
    with open("index.html") as f:
        return f.read()

@app.post("/api/generate")
def generate(req: GenerateRequest):
    # Dispatch image generation based on the selected model
    if req.model == "mnist":
        img, frames = generate_mnist(mnist_model, device)
    elif req.model == "cifar":
        class_idx = int(req.prompt) if req.prompt.isdigit() else None
        img, frames, class_name = generate_cifar(cifar_model, class_idx=class_idx)
        return {"image": img_to_b64(img), "process": frames_to_gif_b64(frames), "class_name": class_name}
    else:
        img, frames = generate_cub(cub_model, prompt=req.prompt, steps=req.steps)
    return {"image": img_to_b64(img), "process": frames_to_gif_b64(frames)}