# 导入所需库
import gradio as gr
import torch
from diffusers import FluxPipeline
from PIL import Image
from io import BytesIO

# 预训练模型和 LoRA 权重路径
model_id = "black-forest-labs/FLUX.1-dev"
lora_dir = "antas/fenglin-flux-lora"

# 加载模型
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(lora_dir, weight_name="fenglin-flux-lora.safetensors")
pipe.enable_model_cpu_offload()
# trigger word
trigger_word = "fenglin"

def generate_image(prompt, seed, num_inference_steps):
    """
    根据用户输入生成图片，并保存为 PNG 格式。
    """
    if seed < -1 or seed == 0:
        raise ValueError("随机种子必须为-1或正整数")

    image = pipe(prompt + ", " + trigger_word,
                 output_type="pil",
                 num_inference_steps=num_inference_steps,
                 generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    # 将图片保存为 PNG 格式并加载到内存中
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    # 将内存中的图片加载为PIL Image对象
    image = Image.open(buffer)

    return image  # 直接返回图片对象

# 创建 Gradio 界面
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="提示文本", placeholder="请输入提示文本"),
        gr.Number(label="随机种子", value=1, precision=0, info="只能输入-1（随机）或正整数，固定的提示文本和固定的正数随机种子会生成同样的图片"),
        gr.Slider(label="推理步数", minimum=1, maximum=100, step=1, value=32)
    ],
    outputs=gr.Image(type="pil", format="png"),  # 使用 PIL Image 类型并指定格式为 png
    title="图像生成器",
    description="请输入提示文本、随机种子及推理步数以生成图片。"
)

# 启动 Gradio 应用
iface.launch(share=True)
