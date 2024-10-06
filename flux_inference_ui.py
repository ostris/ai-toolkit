# 导入所需库
import gradio as gr
import torch
from diffusers import FluxPipeline
from PIL import Image
from io import BytesIO
import argparse
import openai

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='图像生成器参数设置')
parser.add_argument('--model_id', type=str, default="black-forest-labs/FLUX.1-dev",
                    help='预训练模型的ID，默认“black-forest-labs/FLUX.1-dev”，可选“black-forest-labs/FLUX.1-dev”或者“black-forest-labs/FLUX.1-schnell”')
parser.add_argument('--lora_dir', type=str, default="antas/fenglin-flux-lora",
                    help="""Enter lora weight path, either:
                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)""")
parser.add_argument('--trigger_word', type=str, default="fenglin",
                    help='触发词')
parser.add_argument('--use_ai_assist', action='store_true',
                    help='是否启用AI辅助提示词功能，默认不启用')
parser.add_argument('--openai_base_url', type=str, default=None,
                    help='OpenAI API基础URL，以“/v1”结尾')
parser.add_argument('--openai_api_key', type=str, default=None,
                    help='OpenAI API密钥')
parser.add_argument('--model', type=str, default="gpt-4o-mini",
                    help='使用的LLM模型名称')
parser.add_argument('--weight_name', type=str, default="",
                    help='LoRA权重文件名，默认行为是自动判断。')
args = parser.parse_args()

# 从命令行获取参数值
trigger_word = args.trigger_word
use_ai_assist = args.use_ai_assist
openai_base_url = args.openai_base_url
openai_api_key = args.openai_api_key
model_name = args.model

if use_ai_assist:
    openai.api_base = openai_base_url
    openai.api_key = openai_api_key


def refine_prompt_with_ai(prompt):
    """使用OpenAI API润色prompt"""
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content": "Return an English version of text-to-image prompts for Stable Diffusion based on user prompts"},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()


def generate_image(prompt, seed, num_inference_steps):
    """
    根据用户输入生成图片，并保存为 PNG 格式。
    """
    if seed < -1 or seed == 0:
        raise ValueError("随机种子必须为-1或正整数")

    if use_ai_assist:
        prompt = refine_prompt_with_ai(prompt)
    prompt = prompt + ", " + trigger_word
    print(f"{prompt=}")
    image = pipe(prompt,
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


if __name__ == '__main__':
    # 加载模型
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.lora_dir, weight_name=args.weight_name)  # 使用命令行参数传递的权重文件名
    pipe.enable_model_cpu_offload()

    # 创建 Gradio 界面
    iface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(label="提示文本", placeholder="请输入提示文本"),
            gr.Number(label="随机种子", value=1, precision=0,
                      info="只能输入-1（随机）或正整数"),
            gr.Slider(label="推理步数", minimum=1, maximum=100, step=1, value=32)
        ],
        outputs=gr.Image(type="pil", format="png"),  # 使用 PIL Image 类型并指定格式为 png
        title="AI写真定制",
        description="请输入提示文本、随机种子及推理步数以生成图片。"
    )
    # 启动 Gradio 应用
    iface.launch()
