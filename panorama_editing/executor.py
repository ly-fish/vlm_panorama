import os
import torch
from PIL import Image
from panorama_editing.qwen_image_editing.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline 
from io import BytesIO
import requests
from PIL import Image

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", 
    torch_dtype=torch.bfloat16)
print("pipeline loaded")

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "output_image")
os.makedirs(output_dir, exist_ok=True)

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
# image1 = Image.open(BytesIO(requests.get("https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen-Image/edit2511/edit2511input.png").content))
image1 = Image.open("/users/2522553y/liangyue_ws/vlm_panorama/panorama_editing/input_image/robot_dog.png")
prompt = "这是一幅全景图，请在图中加一张桌子，并且放置一个机械臂在桌子上，机器狗放在地上。"
inputs = {
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    save_path = os.path.join(output_dir, "output_image_edit_2511.png")
    output_image.save(save_path)
    print("image saved at", os.path.abspath(save_path))