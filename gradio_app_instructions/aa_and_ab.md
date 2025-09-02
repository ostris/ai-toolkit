## 💡 Usage Tips
You can adjust the following key hyperparameters based on your specific use case.
- `Text Guidance Scale`: Controls how strictly the output adheres to the text prompt (Classifier-Free Guidance).
- `Number of Inference steps`: Controls image denoising during image generations. If you are unsure about what to set, leave it as default.
- `Number of images per prompt`: Set this to the number of images you want to generate per prompt (default is 1)


**Suggestions for generation:**
1. Be Specific with Instructions
  - Clearly describe **what you want in the output image** in a clear and concise manner.
  - This checkpoint works best for close up and torso-level shots of Allu Arjun and Alia Bhatt as it was not trained with long or wide range shots
  - **STRICTLY** use the phrase "(([A] man))" and "(([AB] woman))" for Allu Arjun and Alia Bhatt respectively.
  - Example prompt-1: A photo of (([A] man)) and (([AB] woman)) sitting in a restaurant, having coffee
  - Example prompt-2: A photo of (([A] man)) with an angry expression, wearing white shirt covered in blood with dusty road in the background
  - Example prompt-3: A photo of (([AB] woman)) looking scared standing in a court room, with random people in the background

2. Aspect Ratio
  - For obtaining both Allu Arjun and Alia Bhatt in one frame, mostly use `16:9 aspect ratio (width = 1664; height = 928)`

3. Prioritize English
  - The model currently performs best with **English** prompts.