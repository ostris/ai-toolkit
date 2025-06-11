import os
from huggingface_hub import whoami    
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys

# Add the current working directory to the Python path
sys.path.insert(0, os.getcwd())

import gradio as gr
from extensions_built_in.dataset_tools.tools.image_tools import load_image
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

MAX_IMAGES = 150

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    # Update for the captioning_area
    # for _ in range(3):
    updates.append(gr.update(visible=True))
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)
        
        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))
        
        corresponding_caption = False
        if(image_value):
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            print(base_name)
            print(image_value)
            if base_name in txt_files_dict:
                print("entrou")
                with open(txt_files_dict[base_name], 'r', encoding='utf-8') as file:
                    corresponding_caption = file.read()
                    
        # Update value of captioning area
        text_value = corresponding_caption if visible and corresponding_caption else "[trigger]" if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))

    # Update for the sample caption area
    updates.append(gr.update(visible=True))
    # Update prompt samples
    updates.append(gr.update(placeholder=f'A portrait of person in a bustling cafe {concept_sentence}', value=f'A person in a bustling cafe {concept_sentence}'))
    updates.append(gr.update(placeholder=f"A mountainous landscape in the style of {concept_sentence}"))
    updates.append(gr.update(placeholder=f"A {concept_sentence} in a mall"))
    updates.append(gr.update(visible=True))
    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) 

def create_dataset(*inputs):
    print("Creating dataset")
    images = inputs[0]
    destination_folder = str(f"datasets/{uuid.uuid4()}")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
    with open(jsonl_file_path, "a", encoding='utf-8') as jsonl_file:
        for index, image in enumerate(images):
            new_image_path = shutil.copy(image, destination_folder)

            original_caption = inputs[index + 1]
            file_name = os.path.basename(new_image_path)

            data = {"file_name": file_name, "prompt": original_caption}

            jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")

    return destination_folder


def run_captioning(images, concept_sentence, *captions):
    #Load internally to not consume resources for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = load_image(image_path, force_rgb=True)

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
    steps,
    lr,
    rank,
    model_to_train,
    low_vram,
    dataset_folder,
    sample_1,
    sample_2,
    sample_3,
    use_more_advanced_options,
    more_advanced_options,
):
    push_to_hub = True
    if not lora_name:
        raise gr.Error("You forgot to insert your LoRA name! This name has to be unique.")
    try:
        if whoami()["auth"]["accessToken"]["role"] == "write" or "repo.write" in whoami()["auth"]["accessToken"]["fineGrained"]["scoped"][0]["permissions"]:
            gr.Info(f"Starting training locally {whoami()['name']}. Your LoRA will be available locally and in Hugging Face after it finishes.")
        else:
            push_to_hub = False
            gr.Warning("Started training locally. Your LoRa will only be available locally because you didn't login with a `write` token to Hugging Face")
    except:
        push_to_hub = False
        gr.Warning("Started training locally. Your LoRa will only be available locally because you didn't login with a `write` token to Hugging Face")
            
    print("Started training")
    slugged_lora_name = slugify(lora_name)

    # Load the default config
    with open("config/examples/train_lora_flux_24gb.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Update the config with user inputs
    config["config"]["name"] = slugged_lora_name
    config["config"]["process"][0]["model"]["low_vram"] = low_vram
    config["config"]["process"][0]["train"]["skip_first_sample"] = True
    config["config"]["process"][0]["train"]["steps"] = int(steps)
    config["config"]["process"][0]["train"]["lr"] = float(lr)
    config["config"]["process"][0]["network"]["linear"] = int(rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(rank)
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
    config["config"]["process"][0]["save"]["push_to_hub"] = push_to_hub
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
        config["config"]["process"][0]["sample"]["sample_every"] = steps
        config["config"]["process"][0]["sample"]["sample_steps"] = 28
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
    with open(config_path, "w", encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
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
        """# LoRA Ease for FLUX 🧞‍♂️
### Train a high quality FLUX LoRA in a breeze ༄ using [Ostris' AI Toolkit](https://github.com/ostris/ai-toolkit)"""
    )
    with gr.Column() as main_ui:
        with gr.Row():
            lora_name = gr.Textbox(
                label="The name of your LoRA",
                info="This has to be a unique name",
                placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
            )
            concept_sentence = gr.Textbox(
                label="Trigger word/sentence",
                info="Trigger word or sentence to be used",
                placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                interactive=True,
            )
        with gr.Group(visible=True) as image_upload:
            with gr.Row():
                images = gr.File(
                    file_types=["image", ".txt"],
                    label="Upload your images",
                    file_count="multiple",
                    interactive=True,
                    visible=True,
                    scale=1,
                )
                with gr.Column(scale=3, visible=False) as captioning_area:
                    with gr.Column():
                        gr.Markdown(
                            """# Custom captioning
<p style="margin-top:0">You can optionally add a custom caption for each image (or use an AI model for this). [trigger] will represent your concept sentence/trigger word.</p>
""", elem_classes="group_padding")
                        do_captioning = gr.Button("Add AI captions with Florence-2")
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
            steps = gr.Number(label="Steps", value=1000, minimum=1, maximum=10000, step=1)
            lr = gr.Number(label="Learning Rate", value=4e-4, minimum=1e-6, maximum=1e-3, step=1e-6)
            rank = gr.Number(label="LoRA Rank", value=16, minimum=4, maximum=128, step=4)
            model_to_train = gr.Radio(["dev", "schnell"], value="dev", label="Model to train")
            low_vram = gr.Checkbox(label="Low VRAM", value=True)
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

    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    
    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )

    images.clear(
        hide_captioning,
        outputs=[captioning_area, sample, start]
    )
    
    start.click(fn=create_dataset, inputs=[images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            lora_name,
            concept_sentence,
            steps,
            lr,
            rank,
            model_to_train,
            low_vram,
            dataset_folder,
            sample_1,
            sample_2,
            sample_3,
            use_more_advanced_options,
            more_advanced_options
        ],
        outputs=progress_area,
    )

    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)