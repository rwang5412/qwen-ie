import os, torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

assert torch.cuda.is_available(), "no CUDA visible — wrong node"

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16
).to("cuda")

img = Image.new("RGB", (512, 512), "white")
out = pipe(image=[img], prompt="make the background blue",
           true_cfg_scale=4.0, negative_prompt=" ", num_inference_steps=4)
out.images[0].save("smoke_out.png")
print("SMOKE PASS ->", os.path.abspath("smoke_out.png"))
