# Fine-Tuning and Inference with TI LoRAs

## Fine-Tuning

### On AMD MI300X GPUs - dheyo_amd Branch

- Login to the jump server from your VSCode on `193.143.78.200`

- SSH to gpu-60 with Tunneling

```bash
ssh -L 7777:localhost:7777 ubuntu@gpu-60
```

- Navigate to ai-toolkit directory 

```bash
cd /shareddata/dheyo/shivanvitha/ai-toolkit
```

- Activate the virtual environment

```bash 
source aitool/bin/activate
```

- Launch UI 

```bash
cd ui
```

```bash
npm run build_and_start
```

- Open the application on port 7777 on the browser 

```bash 
http://localhost:7777/
```

- Upload dataset(s): Navigate to the `Datasets` link on the left menu and add your images there. Write the captions for each image in the textbox provided beneath each image. [DO NOT FORGET TO PRESS ENTER ONCE YOU FINISH WRITING EVERY SINGLE CAPTION]

- **IMPORTANT: Assign a unique special token to each character and follow it up by the class of the subject. (eg.,: [A] man, [AB] woman)**

- Navigate to the `+ New Job` section on the left menu and choose the below settings:

| Nodel Configuration     | Value           |
|-----------------|--------------------------|
| Training Name   | Give some name based on the character(s) that you are about to train |
| GPU ID      | Choose any idle GPU (among the 8 [`GPU #0`, `GPU #1`,..., `GPU #7` ]) |
| Model Architecture | Qwen/Qwen-Image |
| Save Every      | 300 or 400 |
| Batch Size      | 2 |
| Steps      | At least 4000 |

- Leave the `Embedding Training` option as it is in the training configuration

| Dataset Configuration     | Value           |
|-----------------|--------------------------|
| Dataset 1 - Trigger/Special Token | The special token to be used for the character in the dataset (eg,: [A], [AB], etc..) |
| Dataset 1 - Initializer Concept | A short description of the character (eg,: A 35 year old Indian man with a strong build, brown skin, beard and mustache, A young Indian woman with fair skin, dimples on cheeks, fair skin, slim build)|

Do the same if there are more than one datasets to be trained after clicking on `Add Dataset`

- Sample Prompts: At the bottom of the page you can provide certain validation prompts. Make sure that these have the special trigger followed by the class of the object just like in the training dataset prompts

- Click on `Create Job` after adding all the settings

- Click on the play button at the top right corner

- **Note: If the training fails, ping Shivanvitha Ambati well before 10pm**



----


### On NVIDIA H100 GPUs - dheyo_nvidia Branch

- Login to dheyo01 VSCode on `lh100.dheyo.ai` with password `Gailen804!`

- Navigate to the working directory 

```bash
cd /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit
```

- Activate the virtual environment 

```bash 
source ../../../ai-toolkit/aitool/bin/activate
```

- Launch UI 

```bash
cd ui
```

```bash
npm run build_and_start
```

- Open the application on port 7777 on the browser 

```bash 
lh100.dheyo.ai:7777
```

- The rest of the instructions are same as described in the AMD section

----

## Inference 

- Stay in the same directory (irrespective of which machine) and find the file with the name -- `inference_qwen_image_lora.py`

- The checkpoints are stored in the `output` directory. Find the sub-directory under this with the training name you provided during fine-tuning

- Copy the absolute paths of the below

| Category     | Pattern           |
|-----------------|--------------------------|
| Transformer LoRA Path  | <your_training_name>_<ckpt_number>.safetensors |
| Text Encoder Path      | text_encoder_<your_training_name>_<ckpt_number> |
| Tokenizer Path | tokenizer_<your_training_name>_<ckpt_number> |
| Token Abstraction JSON apth      | There will be a tokens.json file |


- Create a command and launch it

Always look out for idle GPUs and give the device ID accordingly in the environment variable (`HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`) in front of the actual script command

### On AMD:

Single Prompt Inference:

```bash
HIP_VISIBLE_DEVICES=7 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path <your_transformer_lora_path> \
--tokenizer_path <your_tokenizer_path> \
--text_encoder_path <your_text_encoder_path> \
--token_abstraction_json_path tokens.json \
--num_inference_steps 50 \
--output_image_path inferenced_images/<some_file_name>.png \
--instruction "A photo of [A] man in prison, crying, wearing prison outfit with 420 written on his shirt" \
--aspect_ratio "16:9"
```

Multiple Prompt Inference:
```bash
HIP_VISIBLE_DEVICES=7 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path <your_transformer_lora_path> \
--tokenizer_path <your_tokenizer_path> \
--text_encoder_path <your_text_encoder_path> \
--token_abstraction_json_path tokens.json \
--num_inference_steps 50 \
--output_image_path inferenced_images/<some_file_name>.png \
--prompts_path <your_prompts_txt_file>.txt \ \
--aspect_ratio "16:9"
```


### On NVIDIA:

```bash
CUDA_VISIBLE_DEVICES=1 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path <your_transformer_lora_path> \
--tokenizer_path <your_tokenizer_path> \
--text_encoder_path <your_text_encoder_path> \
--token_abstraction_json_path tokens.json \
--num_inference_steps 50 \
--output_image_path inferenced_images/<some_file_name>.png \
--instruction "A photo of [A] man in prison, crying, wearing prison outfit with 420 written on his shirt" \
--aspect_ratio "16:9"
```

Multiple Prompt Inference:
```bash
CUDA_VISIBLE_DEVICES=1 python3 inference_qwen_image_lora.py --model_path "Qwen/Qwen-Image" \
--transformer_lora_path <your_transformer_lora_path> \
--tokenizer_path <your_tokenizer_path> \
--text_encoder_path <your_text_encoder_path> \
--token_abstraction_json_path tokens.json \
--num_inference_steps 50 \
--output_image_path inferenced_images/<some_file_name>.png \
--prompts_path <your_prompts_txt_file>.txt \ \
--aspect_ratio "16:9"
```
