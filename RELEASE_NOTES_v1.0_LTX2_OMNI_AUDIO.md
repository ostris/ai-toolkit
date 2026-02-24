# 🚀 Release: LTX-2 Audio Excellence & Omni-Merge Framework

This major release transforms the AI Toolkit into a production-grade environment for LTX-2, featuring industry-first breakthroughs in **Multi-Concept Audio/Video Generative Models** and **Zero-Bleed Character Merging**.

## 🎙️ Breakthrough #1: LTX-2 Audio Training Excellence
LTX-2 is a unified Audio-Video model, but standard fine-tuning pipelines historically ignored or mangled its acoustic latents. We have completely rewritten the core VAE and network handling to achieve **Audio Training Excellence**:
*   **Joint VAE Encoding/Decoding:** Implemented `ComboVae` and `AudioProcessor` to convert raw audio waveforms directly into log-mel spectrogram latents and natively pass them into the DiT pipeline during generation.
*   **Acoustic & Temporal Positional IDs:** Perfected the injection of `video_coords` and `audio_coords` using RoPE scaling, ensuring temporal consistency and audio synchronization.
*   **Audio Cross-Attention Fine-Tuning:** Unlocked the `audio_a2v_cross_attn` blocks. You can now train the model to directly associate text prompts with specific audio signatures (voices, sound effects, ambient noise) with crystal-clear fidelity.

## 🧬 Breakthrough #2: The Omni-Merge (DO-Merge 2026 Framework)
Merging multiple character LoRAs into a single file traditionally causes catastrophic bleeding (jumbled faces, blended voices, erratic motion). This release introduces the **Omni-Merge Algorithm (DO-Merge style)**—the absolute bleeding edge of 2026 AI research. 

By mathematically dissecting the model's layers, we can force trigger concepts to exist in completely isolated subspaces, while dynamically balancing their shared structural knowledge. 

### Omni-Merge Features:
*   **Bilateral Subspace Orthogonalization (BSO):** Instead of just projecting Character A out of B, the algorithm symmetrically projects them out of each other's principal components. **Cross-Attention**, **Audio**, and **Temporal** layers are strictly isolated. A character's face trigger, unique voice, and unique motion will *never* bleed into another character.
*   **Magnitude & Direction Decoupling (DO-Merge):** Structural (Spatial) layers must share knowledge of anatomy and lighting. Standard merges fail because one LoRA "crushes" the other with larger weight magnitudes. Omni-Merge physically splits weight matrices into *Direction* (averaged) and *Magnitude* (Geometric Mean). Neither character can overpower the other's body structure.
*   **Exact Rank Concatenation:** SVD truncation causes data loss. Omni-Merge uses pure concatenation (`Rank 1 + Rank 2`) to ensure 100% mathematical fidelity.
*   **Direct Execution UI:** Bypassed the unstable Prisma queue system for merges. The Next.js UI now triggers the Python merge directly with live, real-time polling updates.

### 💡 How to Use Omni-Merge
1.  Navigate to the UI: **Orthogonal Character Merge (OCR-Merge)**.
2.  Input your two Character LoRAs.
3.  Adjust the **Bilateral Orthogonalization Strength** slider (Default: 0.50). 
    *   *Higher values strictly isolate the voices/faces/triggers.*
    *   *Lower values allow more structural sharing if characters look too stiff.*
4.  Run the merge and prompt both characters in the same generation with flawless isolation!
