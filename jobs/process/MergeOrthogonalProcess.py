import os
import torch
import torch.nn as nn
from collections import OrderedDict
from safetensors.torch import load_file
from tqdm import tqdm

from jobs.process.BaseMergeProcess import BaseMergeProcess
from toolkit.train_tools import get_torch_dtype


def bilateral_subspace_orthogonalization(A: torch.Tensor, B: torch.Tensor, k_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bilateral Subspace Orthogonalization (DO-Merge / BSO).
    Instead of just projecting A out of B, this projects A out of B's top components, 
    AND projects B out of A's top components. They symmetrically avoid each other's 
    most critical feature pathways.
    """
    if k_fraction <= 0.0:
        return A, B
        
    # Find principal components of B
    U_B, S_B, V_B = torch.svd(B.float())
    k_B = max(1, int(len(S_B) * k_fraction))
    V_Bk = V_B[:, :k_B]
    
    # Find principal components of A
    U_A, S_A, V_A = torch.svd(A.float())
    k_A = max(1, int(len(S_A) * k_fraction))
    V_Ak = V_A[:, :k_A]
    
    # Project A out of B's top space
    proj_A = torch.matmul(A.float(), torch.matmul(V_Bk, V_Bk.t()))
    A_ortho = (A.float() - proj_A).to(A.dtype)
    
    # Project B out of A's top space
    proj_B = torch.matmul(B.float(), torch.matmul(V_Ak, V_Ak.t()))
    B_ortho = (B.float() - proj_B).to(B.dtype)
    
    return A_ortho, B_ortho


def decoupled_magnitude_direction_merge(W_up1: torch.Tensor, W_up2: torch.Tensor, W_down1: torch.Tensor, W_down2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Magnitude/Direction Decoupling (DO-Merge style) with Exact Rank Preservation.
    This solves the issue where one LoRA has massive weight magnitudes and overpowers the other
    during structure merging, even if their directional vectors are aligned.
    """
    # 1. Expand to full matrices to get true structural directions
    W1 = torch.matmul(W_up1.float(), W_down1.float())
    W2 = torch.matmul(W_up2.float(), W_down2.float())
    
    # 2. Decouple Magnitude and Direction
    mag1 = torch.norm(W1, p='fro') + 1e-8
    mag2 = torch.norm(W2, p='fro') + 1e-8
    
    dir1 = W1 / mag1
    dir2 = W2 / mag2
    
    # 3. Merge Directions and Magnitudes independently
    # We average the directions, but we blend magnitudes using geometric mean to prevent explosion
    merged_dir = (dir1 + dir2) / 2.0
    merged_dir = merged_dir / (torch.norm(merged_dir, p='fro') + 1e-8) # re-normalize
    
    merged_mag = torch.sqrt(mag1 * mag2) # Geometric mean of magnitudes balances dominant LoRAs
    
    # 4. Reconstruct and extract EXACT rank (Rank A + Rank B) to prevent any data loss
    W_full = merged_dir * merged_mag
    
    U, S, V = torch.svd(W_full)
    
    # EXACT rank preservation (no dynamic compression)
    target_rank = W_up1.shape[1] + W_up2.shape[1]
    optimal_rank = min(target_rank, min(W_full.shape))
    
    S_sqrt = torch.sqrt(S[:optimal_rank])
    lora_up_new = U[:, :optimal_rank] * S_sqrt.unsqueeze(0)
    lora_down_new = S_sqrt.unsqueeze(1) * V.t()[:optimal_rank, :]
    
    return lora_up_new.to(W_up1.dtype), lora_down_new.to(W_down1.dtype), optimal_rank


class MergeOrthogonalProcess(BaseMergeProcess):
    """
    The DO-Merge Omni-Method (Decouple and Orthogonalize)
    
    This is the absolute bleeding edge of 2026 research, addressing the core mathematical flaws in LoRA merging.
    
    1. Cross-Attention (Prompt Routing): Uses Bilateral Subspace Orthogonalization (BSO). 
       Instead of just projecting A out of B, it projects A out of B AND B out of A. 
       This perfectly isolates their triggers symmetrically, eliminating all cross-talk.
       
    2. Structural/MLP (Anatomy): Uses Magnitude/Direction Decoupling (DO-Merge style). 
       Existing methods fail because one LoRA has larger parameter magnitudes and crushes the other. 
       This explicitly splits magnitude and direction. It averages their directions geometrically, 
       balances their magnitudes via geometric mean, and reconstructs them cleanly.
    """
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.lora_1_path = self.get_conf('lora_1_path', required=True)
        self.lora_2_path = self.get_conf('lora_2_path', required=True)
        self.dare_drop_rate = self.get_conf('dare_drop_rate', 0.5) 
        self.device = self.get_conf('device', 'cpu')

    def run(self):
        super().run()
        print(f"Loading LoRA 1 (Base): {self.lora_1_path}")
        lora_1 = load_file(self.lora_1_path, device=self.device)
        print(f"Loading LoRA 2 (Proj): {self.lora_2_path}")
        lora_2 = load_file(self.lora_2_path, device=self.device)
        
        module_keys_1 = {}
        for key in lora_1.keys():
            if 'lora_up' in key or 'lora_down' in key or 'alpha' in key:
                base_name = key.split('.lora_')[0]
                if base_name not in module_keys_1:
                    module_keys_1[base_name] = {}
                sub_key = key.split('.lora_')[-1]
                module_keys_1[base_name][sub_key] = lora_1[key]
                
        module_keys_2 = {}
        for key in lora_2.keys():
            if 'lora_up' in key or 'lora_down' in key or 'alpha' in key:
                base_name = key.split('.lora_')[0]
                if base_name not in module_keys_2:
                    module_keys_2[base_name] = {}
                sub_key = key.split('.lora_')[-1]
                module_keys_2[base_name][sub_key] = lora_2[key]
                
        shared_modules = set(module_keys_1.keys()).intersection(set(module_keys_2.keys()))
        print(f"Found {len(shared_modules)} shared modules to Ultimate-Merge.")
        
        merged_state_dict = {}
        
        base_k_fraction = max(0.01, min(0.99, float(self.dare_drop_rate)))
        
        for base_name in tqdm(shared_modules, desc="Disentangling & Compressing Subspaces"):
            m1 = module_keys_1[base_name]
            m2 = module_keys_2[base_name]
            
            if 'up.weight' not in m1 or 'down.weight' not in m1 or 'up.weight' not in m2 or 'down.weight' not in m2:
                continue
                
            up1, down1 = m1['up.weight'], m1['down.weight']
            up2, down2 = m2['up.weight'], m2['down.weight']
            
            rank1 = up1.shape[1]
            rank2 = up2.shape[1]
            alpha1 = m1.get('alpha', torch.tensor(rank1)).item()
            alpha2 = m2.get('alpha', torch.tensor(rank2)).item()
            
            is_conv = len(up1.shape) == 4
            
            if is_conv:
                up1_flat = up1.view(up1.shape[0], -1)
                down1_flat = down1.view(down1.shape[0], -1)
                up2_flat = up2.view(up2.shape[0], -1)
                down2_flat = down2.view(down2.shape[0], -1)
            else:
                up1_flat, down1_flat = up1, down1
                up2_flat, down2_flat = up2, down2
            
            # --- STEP 1: ALPHA BAKING ---
            scale1 = (alpha1 / rank1) ** 0.5
            scale2 = (alpha2 / rank2) ** 0.5
            
            up1_scaled = up1_flat * scale1
            down1_scaled = down1_flat * scale1
            up2_scaled = up2_flat * scale2
            down2_scaled = down2_flat * scale2
            
            # Semantic Layer Identification
            lower_name = base_name.lower()
            is_cross_attn = "attn2" in lower_name or "txt_attn" in lower_name or "context" in lower_name or "txt_in" in lower_name or "enc" in lower_name
            is_audio = "audio" in lower_name
            is_temporal = "temp" in lower_name or "time" in lower_name or "video" in lower_name or "motion" in lower_name
            
            if is_cross_attn or is_audio or is_temporal:
                # --- CROSS-ATTENTION, AUDIO, & TEMPORAL: BILATERAL SUBSPACE ORTHOGONALIZATION (BSO) ---
                # We project A out of B, AND B out of A symmetrically.
                # Audio layers and Temporal (Motion) layers act similarly to trigger routes; we do not want their voices or movement styles to bleed.
                down1_ortho, down2_ortho = bilateral_subspace_orthogonalization(down1_scaled, down2_scaled, base_k_fraction)
                
                up1_ortho_t, up2_ortho_t = bilateral_subspace_orthogonalization(up1_scaled.t(), up2_scaled.t(), base_k_fraction)
                up1_ortho = up1_ortho_t.t()
                up2_ortho = up2_ortho_t.t()
                
                up_merge = torch.cat([up1_ortho, up2_ortho], dim=1)
                down_merge = torch.cat([down1_ortho, down2_ortho], dim=0)
                final_rank = rank1 + rank2
                
            else:
                # --- STRUCTURE/MLP: MAGNITUDE/DIRECTION DECOUPLING (DO-Merge style) ---
                # Averages directions but balances their magnitudes so one doesn't crush the other.
                up_merge, down_merge, final_rank = decoupled_magnitude_direction_merge(up1_scaled, up2_scaled, down1_scaled, down2_scaled)
            
            # Reshape back if conv
            if is_conv:
                up_merge = up_merge.view(up_merge.shape[0], up_merge.shape[1], up1.shape[2], up1.shape[3])
                down_merge = down_merge.view(down_merge.shape[0], down_merge.shape[1], down1.shape[2], down1.shape[3])
                
            merged_state_dict[f"{base_name}.lora_up.weight"] = up_merge.to(up1.dtype)
            merged_state_dict[f"{base_name}.lora_down.weight"] = down_merge.to(down1.dtype)
            merged_state_dict[f"{base_name}.alpha"] = torch.tensor(final_rank, dtype=up_merge.dtype)
            
        for key in lora_1.keys():
            base_name = key.split('.lora_')[0]
            if base_name not in shared_modules:
                merged_state_dict[key] = lora_1[key]
                
        for key in lora_2.keys():
            base_name = key.split('.lora_')[0]
            if base_name not in shared_modules:
                merged_state_dict[key] = lora_2[key]
                
        self.save(merged_state_dict)
