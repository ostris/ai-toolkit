import os
from huggingface_hub import whoami    
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys

# Add the current working directory to the Python path
sys.path.insert(0, os.getcwd())

import gradio as gr
from PIL import Image
import torch
import uuid
import os
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.insert(0, "ai-toolkit")
from toolkit.job import get_job

MAX_IMAGES = 300

def load_captioning(uploaded_files, concept_sentence):
   # Parse files
   uploaded_images = [f for f in uploaded_files if not f.endswith('.txt')]
   txt_files_dict = {os.path.splitext(os.path.basename(f))[0]: f 
                     for f in uploaded_files if f.endswith('.txt')}
   
   # Validate image count
   if len(uploaded_images) <= 1:
       raise gr.Error("请至少上传两张图，太少了不太好，建议5-50张图，不建议超过200张图")
   if len(uploaded_images) > MAX_IMAGES:
       raise gr.Error(f"你传的太多了， {MAX_IMAGES} 张图就可以了，多不如精")

   updates = [gr.update(visible=True)]  # Make captioning area visible
   
   # Process each image
   for i in range(1, MAX_IMAGES + 1):
       visible = i <= len(uploaded_images)
       if visible:
           image_value = uploaded_images[i - 1]
           base_name = os.path.splitext(os.path.basename(image_value))[0]
           caption = ""
           if base_name in txt_files_dict:
               with open(txt_files_dict[base_name], 'r') as f:
                   caption = f.read()
           text_value = caption or ("[trigger]" if concept_sentence else None)
       else:
           image_value = None
           text_value = None

       updates.extend([
           gr.update(visible=visible),  # Row visibility
           gr.update(value=image_value, visible=visible),  # Image
           gr.update(value=text_value, visible=visible)  # Caption
       ])

   # Add sample prompts
   updates.extend([
       gr.update(visible=True),  # Sample area
       gr.update(placeholder=f'A portrait of person in a bustling cafe {concept_sentence}', 
                value=f'A person in a bustling cafe {concept_sentence}'),
       gr.update(placeholder=f"A mountainous landscape in the style of {concept_sentence}"),
       gr.update(placeholder=f"A {concept_sentence} in a mall"),
       gr.update(visible=True)
   ])
   
   return updates

def load_reg_captioning(uploaded_reg_images, concept_sentence):
   updates = []
   if len(uploaded_reg_images) <= 0:
       raise gr.Error("请至少上传一张正则化图片")
   elif len(uploaded_reg_images) > MAX_IMAGES:
       raise gr.Error(f"正则化图片数量超过上限 {MAX_IMAGES}")
       
   updates.append(gr.update(visible=True))
   
   for i in range(MAX_IMAGES):
       visible = i < len(uploaded_reg_images)
       updates.append(gr.update(visible=visible))
       image_value = uploaded_reg_images[i] if visible else None
       updates.append(gr.update(value=image_value, visible=visible))
       caption = "[reg]" if visible else None
       updates.append(gr.update(value=caption, visible=visible))
   
   return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) 

def create_dataset(*inputs):
    print("Creating dataset")
    images = inputs[0]
    destination_folder = str(f"datasets/{uuid.uuid4()}")
    os.makedirs(destination_folder, exist_ok=True)

    jsonl_path = os.path.join(destination_folder, "metadata.jsonl") 
    with open(jsonl_path, "a") as f:
        for image, caption in zip(images, inputs[2:]):
            filename = os.path.basename(shutil.copy(image, destination_folder))
            f.write(json.dumps({
                "file_name": filename,
                "prompt": caption
            }) + "\n")

    # 确保reg_destination_folder总是返回有效路径或None
    reg_destination_folder = None
    if len(inputs) > 1 and inputs[1] and isinstance(inputs[1], list) and inputs[1][0]:
        reg_destination_folder = str(f"datasets/reg_{uuid.uuid4()}")
        os.makedirs(reg_destination_folder, exist_ok=True)
        for reg_image in inputs[1]:
            if os.path.exists(reg_image):
                shutil.copy(reg_image, reg_destination_folder)
                filename = os.path.basename(reg_image)
                txt_path = os.path.join(reg_destination_folder, f"{os.path.splitext(filename)[0]}.txt")
                with open(txt_path, "w") as f:
                    f.write("[reg]")
        # 如果没有成功复制任何文件，返回None
        if not os.listdir(reg_destination_folder):
            shutil.rmtree(reg_destination_folder)
            reg_destination_folder = None

    return destination_folder, reg_destination_folder

def run_captioning(images, concept_sentence, *captions):
    #Load internally to not consume resources for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "/root/ai-toolkit/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("/root/ai-toolkit/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        if concept_sentence:
            caption_text = f"{caption_text} [trigger]"
        captions[i] = caption_text

        yield captions
    model.to("cpu")
    del model
    del processor

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def start_training(
    lora_name,
    concept_sentence,
    resolution,
    steps,
    lr,
    rank,
    #alpha,
    model_to_train,
    low_vram,
    dataset_folder,
    save_safetensors_every_steps,
    max_step_saves,
    sample_1,
    sample_2,
    sample_3,
    use_more_advanced_options,
    more_advanced_options,
    is_reg,
    reg_dataset_folder,
    sample_steps,
    batch_size,
    gradient_accumulation,
    lr_scheduler,
    use_network_kwargs,
    transformer_blocks,
    single_transformer_blocks
):
    resolution_ints = [int(res) for res in resolution]
    
    push_to_hub = False
    if not lora_name:
        raise gr.Error("You forgot to insert your LoRA name! This name has to be unique.")
    try:
        if whoami()["auth"]["accessToken"]["role"] == "write" or "repo.write" in whoami()["auth"]["accessToken"]["fineGrained"]["scoped"][0]["permissions"]:
            gr.Info(f"Starting training locally {whoami()['name']}. Your LoRA will be available locally and in Hugging Face after it finishes.")
        else:
            push_to_hub = False
            gr.Warning("训练开始了！")
    except:
        push_to_hub = False
        gr.Warning("训练开始啦！")
            
    print("Started training")
    slugged_lora_name = slugify(lora_name)

    # Load the default config
    with open("/root/ai-toolkit/config/examples/train_lora_flux_24gb.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update the config with user inputs
    config["config"]["name"] = slugged_lora_name
    config["config"]["process"][0]["model"]["low_vram"] = low_vram
    config["config"]["process"][0]["train"]["skip_first_sample"] = False
    config["config"]["process"][0]["train"]["steps"] = int(steps)
    config["config"]["process"][0]["train"]["lr"] = float(lr)
    config["config"]["process"][0]["train"]["batch_size"] = int(batch_size)
    config["config"]["process"][0]["train"]["gradient_accumulation"] = int(gradient_accumulation)
    config["config"]["process"][0]["train"]["reg_weight"] = 0.8
    config["config"]["process"][0]["train"]["lr_scheduler"] = lr_scheduler
    if lr_scheduler == "polynomial":
        lr_scheduler_params = {"power": 0.2}
    else:
        lr_scheduler_params = {}
    config["config"]["process"][0]["train"]["lr_scheduler_params"] = lr_scheduler_params
    config["config"]["process"][0]["network"]["linear"] = int(rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(rank)
    if use_network_kwargs:
        only_if_contains = []
        for block in transformer_blocks:
            only_if_contains.append(f"transformer.transformer_blocks.{block}")
        for block in single_transformer_blocks:
            if block in ["4", "5", "8", "9", "10", "11", "12", "13", "14", "15", "21", "22", "23", "24", "26", "27", "33", "34"]:
                only_if_contains.append(f"transformer.single_transformer_blocks.{block}.proj_out")
            else:
                only_if_contains.append(f"transformer.single_transformer_blocks.{block}")
        
        config["config"]["process"][0]["network"]["network_kwargs"] = {
            "only_if_contains": only_if_contains
        }
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
    config["config"]["process"][0]["save"]["push_to_hub"] = push_to_hub
    config["config"]["process"][0]["save"]["save_every"] = int(save_safetensors_every_steps)
    config["config"]["process"][0]["save"]["max_step_saves_to_keep"] = int(max_step_saves)
    config["config"]["process"][0]["datasets"][0] = {
        "folder_path": dataset_folder,
        "caption_ext": "txt",
        "caption_dropout_rate": 0.05,
        "shuffle_tokens": False,
        "cache_latents_to_disk": True,
        "resolution": resolution_ints,
        "flip_aug": True,
        "keep_tokens": True
    }
    if is_reg and reg_dataset_folder:
        config["config"]["process"][0]["datasets"].append({
            "folder_path": reg_dataset_folder,
            "is_reg": True,
            "caption_ext": "txt",
            "caption_dropout_rate": 0.05,
            "shuffle_tokens": False,
            "resolution": 768,
            "cache_latents_to_disk": True
        })
    if(push_to_hub):
        try:
            username = whoami()["name"]
        except:
            raise gr.Error("Error trying to retrieve your username. Are you sure you are logged in with Hugging Face?")
        config["config"]["process"][0]["save"]["hf_repo_id"] = f"{username}/{slugged_lora_name}"
        config["config"]["process"][0]["save"]["hf_private"] = True
    if concept_sentence:
        config["config"]["process"][0]["trigger_word"] = concept_sentence
    
    if sample_1 or sample_2 or sample_3:
        config["config"]["process"][0]["train"]["disable_sampling"] = False
        config["config"]["process"][0]["sample"]["sample_every"] = int(sample_steps)
        config["config"]["process"][0]["sample"]["sample_steps"] = 25
        config["config"]["process"][0]["sample"]["prompts"] = []
        if sample_1:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_1)
        if sample_2:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_2)
        if sample_3:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_3)
    else:
        config["config"]["process"][0]["train"]["disable_sampling"] = True
    if(model_to_train == "schnell"):
        config["config"]["process"][0]["model"]["name_or_path"] = "black-forest-labs/FLUX.1-schnell"
        config["config"]["process"][0]["model"]["assistant_lora_path"] = "ostris/FLUX.1-schnell-training-adapter"
        config["config"]["process"][0]["sample"]["sample_steps"] = 4
    if(use_more_advanced_options):
        more_advanced_options_dict = yaml.safe_load(more_advanced_options)
        config["config"]["process"][0] = recursive_update(config["config"]["process"][0], more_advanced_options_dict)
        print(config)
    
    # Save the updated config
    # generate a random name for the config
    random_config_name = str(uuid.uuid4())
    os.makedirs("tmp", exist_ok=True)
    config_path = f"tmp/{random_config_name}-{slugged_lora_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # run the job locally
    job = get_job(config_path)
    job.run()
    job.cleanup()

    return f"Training completed successfully. Model saved as {slugged_lora_name}"

config_yaml = '''
device: cuda:0
model:
  is_flux: true
  quantize: true
network:
  linear: 16 #it will overcome the 'rank' parameter
  linear_alpha: 16 #you can have an alpha different than the ranking if you'd like
  type: lora
sample:
  guidance_scale: 3.5
  height: 1024
  neg: '' #doesn't work for FLUX
  sample_every: 1000
  sample_steps: 28
  sampler: flowmatch
  seed: 42
  walk_seed: true
  width: 1024
save:
  dtype: float16
  hf_private: true
  max_step_saves_to_keep: 4
  push_to_hub: true
  save_every: 10000
train:
  batch_size: 1
  dtype: bf16
  ema_config:
    ema_decay: 0.99
    use_ema: true
  gradient_accumulation_steps: 1
  gradient_checkpointing: true
  noise_scheduler: flowmatch 
  optimizer: adamw8bit #options: prodigy, dadaptation, adamw, adamw8bit, lion, lion8bit
  train_text_encoder: false #probably doesn't work for flux
  train_unet: true
'''

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
h1{font-size: 2em}
h3{margin-top: 0}
#component-1{text-align:center}
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
.tabitem{border: 0px}
.group_padding{padding: .55em}
"""
with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown(
        """# AI-Toolkit LORA 训练界面
### 训练高质量FLUX-LORA 原作者[Ostris' AI Toolkit](https://github.com/ostris/ai-toolkit) 由[PAseer](https://space.bilibili.com/52227183)整理汉化""" 
    )
    with gr.Column() as main_ui:
        with gr.Row():
            lora_name = gr.Textbox(
                label="The name of your LoRA",
                info="This has to be a unique name",
                placeholder="你的LORA想叫什么名字？",
            )
            concept_sentence = gr.Textbox(
                label="Trigger word/sentence",
                info="Trigger word or sentence to be used",
                placeholder="填写你的触发词",
                interactive=True,
            )
        with gr.Group(visible=True) as image_upload:
            with gr.Row():
                images = gr.File(
                    file_types=["image", ".txt"],
                    label="请上传你的训练集",
                    file_count="multiple",
                    interactive=True,
                    visible=True,
                    scale=1,
                )
                with gr.Column(scale=3, visible=False) as captioning_area:
                    with gr.Column():
                        gr.Markdown(
                            """# Custom captioning
<p style="margin-top:0">PAseer我还是建议手动修改一下 (或者干脆摆烂，点击黑色按钮自动生成). [trigger] will represent your concept sentence/trigger word.</p>
""", elem_classes="group_padding")
                        do_captioning = gr.Button("自动生成自然语言标签：Florence-2")
                        output_components = [captioning_area]
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(
                                    label=f"Caption {i}", scale=15, interactive=True
                                )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])

        with gr.Accordion("Advanced options", open=False):
            resolution = gr.CheckboxGroup(["512","768","832","896","1024"], value="1024", label="选择训练分桶分辨率(可多选)")
            steps = gr.Number(label="Steps", value=1000, minimum=1, maximum=10000, step=1)
            save_safetensors_every_steps = gr.Number(label="每多少步保存一次", value=200, minimum=1, maximum=10000, step=50)
            max_step_saves = gr.Number(label="最多保留多少中间模型", value=4, minimum=1, maximum=100, step=1, info="控制保存的checkpoint数量，避免占用过多磁盘空间")
            lr = gr.Number(label="Learning Rate", value=3.6e-4, minimum=1e-6, maximum=1e-3, step=1e-6, info="学习风格建议默认，学习新概念建议5e-4")
            lr_scheduler = gr.Radio(["constant", "cosine", "polynomial"], value="polynomial", label="LR Scheduler")
            rank = gr.Number(label="LoRA Rank", value=16, minimum=4, maximum=256, step=1, info="风格建议32，人物建议8")
            use_network_kwargs = gr.Checkbox(label="启用择层训练", value=False)
            with gr.Group() as network_kwargs_group:
                transformer_blocks = gr.CheckboxGroup(
                    ["0", "2", "5", "12", "15", "18"],
                    label="Double Blocks",
                    info="选择你想训练的double_block层",
                    visible=False
                )
                single_transformer_blocks = gr.CheckboxGroup(
                    ["0", "4", "5", "7", "8", "9", "10", "11", "12", "13", "14", "15", "20", "21", "22", "23", "24", "25", "26", "27", "33", "34", "35"],
                    label="Single Transformer Blocks",
                    info="选择你想训练的single_blocks层",
                    visible=False
                )
                def update_network_kwargs_visibility(use_network_kwargs):
                    return [
                        gr.update(visible=use_network_kwargs),
                        gr.update(visible=use_network_kwargs)
                    ]
                use_network_kwargs.change(
                    update_network_kwargs_visibility,
                    inputs=[use_network_kwargs],
                    outputs=[transformer_blocks, single_transformer_blocks]
                )
            #alpha = gr.Number(label="LoRA Alpha", value=8, minimum=1, maximum=128, step=1)
            batch_size = gr.Number(label="间断并行bs", value=1, minimum=1, maximum=32, step=1, info="就是batchsize，占用显存约23G-32G")
            gradient_accumulation = gr.Number(label="梯度累积gacc", value=2, minimum=1, maximum=32, step=1, info="占用显存约23G-32G")
            sample_steps = gr.Number(label="每多少步出一张样品图？", value=200, minimum=100, maximum=10000, step=50)
            model_to_train = gr.Radio(["dev"], value="dev", label="目前仅支持Dev版本")
            low_vram = gr.Checkbox(label="Low VRAM", value=False)
            is_reg = gr.Checkbox(label="正则化", value=False)
            with gr.Group(visible=False) as reg_image_upload:
                reg_images = gr.File(
                    file_types=["image", ".txt"],
                    label="请上传你的正则化训练集",
                    file_count="multiple",
                    interactive=True,
                    visible=True,
                    scale=1,
                )                
                with gr.Column(scale=3, visible=False) as reg_captioning_area:
                    gr.Markdown("# 正则化数据标注")
                    do_reg_captioning = gr.Button("自动生成正则化数据标注")
                    reg_output_components = [reg_captioning_area]
                    reg_caption_list = []
                    for i in range(1, MAX_IMAGES + 1):
                        locals()[f"reg_captioning_row_{i}"] = gr.Row(visible=False)
                        with locals()[f"reg_captioning_row_{i}"]:
                            locals()[f"reg_image_{i}"] = gr.Image(
                                type="filepath",
                                width=111,
                                height=111,
                                min_width=111,
                                interactive=False,
                                scale=2,
                                show_label=False,
                                show_share_button=False,
                                show_download_button=False,
                            )
                            locals()[f"reg_caption_{i}"] = gr.Textbox(
                                label=f"Caption {i}", scale=15, interactive=True
                            )
                        reg_output_components.append(locals()[f"reg_captioning_row_{i}"])
                        reg_output_components.append(locals()[f"reg_image_{i}"])
                        reg_output_components.append(locals()[f"reg_caption_{i}"])
                        reg_caption_list.append(locals()[f"reg_caption_{i}"])
                                
            with gr.Accordion("Even more advanced options", open=False):
                use_more_advanced_options = gr.Checkbox(label="Use more advanced options", value=False)
                more_advanced_options = gr.Code(config_yaml, language="yaml")

        with gr.Accordion("Sample prompts (optional)", visible=False) as sample:
            gr.Markdown(
                "Include sample prompts to test out your trained model. Don't forget to include your trigger word/sentence (optional)"
            )
            sample_1 = gr.Textbox(label="Test prompt 1")
            sample_2 = gr.Textbox(label="Test prompt 2")
            sample_3 = gr.Textbox(label="Test prompt 3")
        
        output_components.append(sample)
        output_components.append(sample_1)
        output_components.append(sample_2)
        output_components.append(sample_3)
        start = gr.Button("Start training", visible=False)
        output_components.append(start)
        progress_area = gr.Markdown("")

    dataset_folder = gr.State()
    reg_dataset_folder = gr.State()

    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components,
        concurrency_limit=4
    )
    
    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components,
        concurrency_limit=4
    )

    images.clear(
        hide_captioning,
        outputs=[captioning_area, sample, start]
    )
    def create_reg_dataset(reg_images):
        destination_folder = str(f"datasets/reg_{uuid.uuid4()}")
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        for image in reg_images:
            shutil.copy(image, destination_folder)
        
        return destination_folder

    reg_images.upload(fn=create_reg_dataset, inputs=[reg_images], outputs=reg_dataset_folder)
    reg_images.upload(load_reg_captioning, inputs=[reg_images, concept_sentence], outputs=reg_output_components)
    reg_images.clear(lambda: [gr.update(visible=False)] * len(reg_output_components), outputs=reg_output_components)
    do_reg_captioning.click(fn=run_captioning, inputs=[reg_images, concept_sentence] + reg_caption_list, outputs=reg_caption_list)
    
    is_reg.change(lambda x: gr.update(visible=x), inputs=[is_reg], outputs=[reg_image_upload])
    
    start.click(fn=create_dataset, inputs=[images, reg_images] + caption_list, outputs=[dataset_folder, reg_dataset_folder]).then(
        fn=start_training,
        inputs=[
            lora_name,
            concept_sentence,
            resolution,
            steps,
            lr,
            rank,
            #alpha,
            model_to_train,
            low_vram,
            dataset_folder,
            save_safetensors_every_steps,
            max_step_saves,
            sample_1,
            sample_2,
            sample_3,
            use_more_advanced_options,
            more_advanced_options,
            is_reg,
            reg_dataset_folder,
            sample_steps,
            batch_size,
            gradient_accumulation,
            lr_scheduler,
            use_network_kwargs,
            transformer_blocks,
            single_transformer_blocks
        ],
        outputs=progress_area,
    )

    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=12800,
        show_error=True, 
        share=True,
        max_threads=8
    )
