import torch
import torch.nn.functional as F

class FreeFuseState:
    _instance = None

    @classmethod
    def get_instance(cls):
        return cls._instance

    @classmethod
    def set_instance(cls, instance):
        cls._instance = instance

    def __init__(self, concepts, extract_step=4):
        self.concepts = concepts
        self.extract_step = extract_step
        self.phase = 1 # 1: extract, 2: generate
        self.video_sim_maps = None
        self.audio_sim_maps = None
        self.video_routing_mask = None
        self.audio_routing_mask = None
        self.video_attention_bias = None
        self.audio_attention_bias = None
        self.target_token_indices = []

    def find_token_indices(self, tokenizer, prompt, padding_side='left', max_length=1024):
        self.target_token_indices = []
        if isinstance(prompt, list):
            prompt = prompt[0]
            
        text_inputs = tokenizer(
            [prompt],
            padding="longest", # wait, we usually pad to max_length for ltx2
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids[0].tolist()
        
        # calculate pad length assuming padding="max_length" or manual pad
        pad_length = max_length - len(input_ids)
        if pad_length < 0:
            pad_length = 0
            
        for concept in self.concepts:
            trigger = concept.get('trigger', '')
            # Simple substring matching in token decoding
            trigger_idx = -1
            for i, token_id in enumerate(input_ids):
                token_str = tokenizer.decode([token_id])
                if trigger.lower() in token_str.lower() and len(token_str.strip()) > 0:
                    trigger_idx = i
                    break
            
            if trigger_idx == -1:
                print(f"WARNING: FreeFuse concept '{trigger}' not found in prompt!")
                trigger_idx = 0
            
            # account for left padding
            if padding_side == 'left':
                actual_idx = trigger_idx + pad_length
            else:
                actual_idx = trigger_idx
                
            self.target_token_indices.append(actual_idx)
            concept['token_idx'] = actual_idx
            
    def calculate_routing_masks(self):
        if self.video_sim_maps is not None:
            # Softmax to create routing masks
            self.video_routing_mask = torch.softmax(self.video_sim_maps / 0.1, dim=-1)
            b, num_vid_tokens, num_concepts = self.video_routing_mask.shape
            max_text_len = 1024
            
            bias = torch.zeros((b, 1, num_vid_tokens, max_text_len), device=self.video_routing_mask.device)
            for i, concept in enumerate(self.concepts):
                token_idx = concept['token_idx']
                concept_mask = self.video_routing_mask[:, :, i] # [B, NumVidTokens]
                
                for j, other_concept in enumerate(self.concepts):
                    if i != j:
                        other_token_idx = other_concept['token_idx']
                        bias[:, 0, :, other_token_idx] -= 10000.0 * concept_mask
            self.video_attention_bias = bias

        if self.audio_sim_maps is not None:
            self.audio_routing_mask = torch.softmax(self.audio_sim_maps / 0.1, dim=-1)
            b, num_aud_tokens, num_concepts = self.audio_routing_mask.shape
            max_text_len = 1024
            
            bias = torch.zeros((b, 1, num_aud_tokens, max_text_len), device=self.audio_routing_mask.device)
            for i, concept in enumerate(self.concepts):
                token_idx = concept['token_idx']
                concept_mask = self.audio_routing_mask[:, :, i]
                for j, other_concept in enumerate(self.concepts):
                    if i != j:
                        other_token_idx = other_concept['token_idx']
                        bias[:, 0, :, other_token_idx] -= 10000.0 * concept_mask
            self.audio_attention_bias = bias

class LTX2FreeFuseAttnProcessor:
    def __init__(self, state: FreeFuseState, is_video=True):
        self.state = state
        self.is_video = is_video

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs
    ):
        batch_size = hidden_states.shape[0]
        
        is_cross_attn = encoder_hidden_states is not None
        key_states = encoder_hidden_states if is_cross_attn else hidden_states
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(key_states)
        value = attn.to_v(key_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if is_cross_attn:
            if self.state.phase == 1:
                # Extract sim map
                sim_scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale
                # Shape: [B, Heads, Seq_len, Text_len]
                avg_sim = sim_scores.mean(dim=1) # [B, Seq_len, Text_len]
                
                # Extract only target concepts
                concept_sims = []
                for idx in self.state.target_token_indices:
                    concept_sims.append(avg_sim[:, :, idx:idx+1])
                    
                if len(concept_sims) > 0:
                    sim_map = torch.cat(concept_sims, dim=-1) # [B, Seq_len, NumConcepts]
                    
                    if self.is_video:
                        if self.state.video_sim_maps is None:
                            self.state.video_sim_maps = sim_map
                        else:
                            self.state.video_sim_maps += sim_map
                    else:
                        if self.state.audio_sim_maps is None:
                            self.state.audio_sim_maps = sim_map
                        else:
                            self.state.audio_sim_maps += sim_map
                            
            elif self.state.phase == 2:
                # Add bias
                bias = self.state.video_attention_bias if self.is_video else self.state.audio_attention_bias
                if bias is not None:
                    if attention_mask is None:
                        attention_mask = bias
                    else:
                        attention_mask = attention_mask + bias

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        
        return hidden_states

def inject_freefuse_processor(transformer, state: FreeFuseState):
    # LTX2 Transformer has video_a2v_cross_attn and audio_a2v_cross_attn or similar
    from diffusers.models.attention_processor import Attention
    for name, module in transformer.named_modules():
        if isinstance(module, Attention):
            is_video = True
            if "audio" in name:
                is_video = False
            module.set_processor(LTX2FreeFuseAttnProcessor(state, is_video=is_video))

def remove_freefuse_processor(transformer):
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    for name, module in transformer.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor2_0())
