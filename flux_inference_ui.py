# 导入所需库
import gradio as gr
import torch
from diffusers import FluxPipeline

# 预训练模型和 LoRA 权重路径
model_id = "black-forest-labs/FLUX.1-dev"
lora_dir = "antas/fenglin-flux-lora"

# 加载模型
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(lora_dir, weight_name="fenglin-flux-lora.safetensors")
pipe.enable_model_cpu_offload()


def generate_image(prompt, seed, num_inference_steps):
    """
    根据用户输入生成图片。

    :param prompt: 提示文本
    :param seed: 随机种子
    :param num_inference_steps: 推理步数
    :return: 生成的图片
    """
    # 检查随机种子是否合法
    if seed < -1 or seed == 0:
        raise ValueError("随机种子必须为-1或正整数")

    # 生成图片
    image = pipe(prompt, output_type="pil", num_inference_steps=num_inference_steps,
                 generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    return image


# 创建 Gradio 界面
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="提示文本", placeholder="请输入提示文本"),
        gr.Number(label="随机种子", value=1, precision=0,
                  info="只能输入-1（随机）或正整数，固定的提示文本和固定的正数随机种子会生成同样的图片"),
        gr.Slider(label="推理步数", minimum=1, maximum=100, step=1, value=32)
    ],
    outputs=gr.Image(type="pil"),
    title="图像生成器",
    description="请输入提示文本、随机种子及推理步数以生成图片。"
)

# 启动 Gradio 应用
iface.launch(share=True)
