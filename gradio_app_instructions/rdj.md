## 💡 Usage Tips
You can adjust the following key hyperparameters based on your specific use case.
- `Text Guidance Scale`: Controls how strictly the output adheres to the text prompt (Classifier-Free Guidance).
- `Number of Inference steps`: Controls image denoising during image generations. If you are unsure about what to set, leave it as default.
- `Number of images per prompt`: Set this to the number of images you want to generate per prompt (default is 1)


**Suggestions for generation:**
1. Be Specific with Instructions
  - Clearly describe **what you want in the output image** in a clear and concise manner.
  - This checkpoint works best for close up and torso-level shots of Robert Downey Jr. as it was not trained with long or wide range shots
  - **STRICTLY** use the phrase "(([RD] man))" for to trigger Robert Downey Jr. generation.
  - Example prompt-1: A photo of (([RD] man)) sitting in a restaurant, having coffee
  

3. Prioritize English
  - The model currently performs best with **English** prompts.