from pipeline import DiT360Pipeline
import torch
import os

device = torch.device("cuda:0")
pipe = DiT360Pipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16).to(device)
pipe.load_lora_weights("Insta360-Research/DiT360-Panorama-Image-Generation")

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output_image")

image = pipe(
    # "This is a panorama. The image shows a medieval castle stands proudly on a hilltop surrounded by autumn forests, with golden light spilling across the landscape.",
    "This is a panorama. A small cottage in the center of a flower field, surrounded by blooming pink, white, and purple wildflowers. A stone path leads to the door. Rolling hills and a blue sky with clouds in the distance, dreamy and soft, illustration style, 360° panoramic image.",
    width=2048,
    height=1024,
    num_inference_steps=28,
    guidance_scale=2.8,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]

save_path = os.path.join(output_dir, "output_image_generate.png")
image.save(save_path)
print("image saved at", os.path.abspath(save_path))