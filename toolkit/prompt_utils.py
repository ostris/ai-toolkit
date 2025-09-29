import os
from typing import Optional, TYPE_CHECKING, List, Union, Tuple

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import random

from toolkit.train_tools import get_torch_dtype
import itertools

if TYPE_CHECKING:
    from toolkit.config_modules import SliderTargetConfig


class ACTION_TYPES_SLIDER:
    ERASE_NEGATIVE = 0
    ENHANCE_NEGATIVE = 1


class PromptEmbeds:
    # text_embeds: torch.Tensor
    # pooled_embeds: Union[torch.Tensor, None]
    # attention_mask: Union[torch.Tensor, List[torch.Tensor], None]

    def __init__(self, args: Union[Tuple[torch.Tensor], List[torch.Tensor], torch.Tensor], attention_mask=None) -> None:
        if isinstance(args, list) or isinstance(args, tuple):
            # xl
            self.text_embeds = args[0]
            self.pooled_embeds = args[1]
        else:
            # sdv1.x, sdv2.x
            self.text_embeds = args
            self.pooled_embeds = None

        self.attention_mask = attention_mask

    def to(self, *args, **kwargs):
        if isinstance(self.text_embeds, list) or isinstance(self.text_embeds, tuple):
            self.text_embeds = [t.to(*args, **kwargs) for t in self.text_embeds]
        else:
            self.text_embeds = self.text_embeds.to(*args, **kwargs)
        if self.pooled_embeds is not None:
            self.pooled_embeds = self.pooled_embeds.to(*args, **kwargs)
        if self.attention_mask is not None:
            if isinstance(self.attention_mask, list) or isinstance(self.attention_mask, tuple):
                self.attention_mask = [t.to(*args, **kwargs) for t in self.attention_mask]
            else:
                self.attention_mask = self.attention_mask.to(*args, **kwargs)
        return self

    def detach(self):
        new_embeds = self.clone()
        if isinstance(new_embeds.text_embeds, list) or isinstance(new_embeds.text_embeds, tuple):
            new_embeds.text_embeds = [t.detach() for t in new_embeds.text_embeds]
        else:
            new_embeds.text_embeds = new_embeds.text_embeds.detach()
        if new_embeds.pooled_embeds is not None:
            new_embeds.pooled_embeds = new_embeds.pooled_embeds.detach()
        if new_embeds.attention_mask is not None:
            if isinstance(new_embeds.attention_mask, list) or isinstance(new_embeds.attention_mask, tuple):
                new_embeds.attention_mask = [t.detach() for t in new_embeds.attention_mask]
            else:
                new_embeds.attention_mask = new_embeds.attention_mask.detach()
        return new_embeds

    def clone(self):
        if isinstance(self.text_embeds, list) or isinstance(self.text_embeds, tuple):
            cloned_text_embeds = [t.clone() for t in self.text_embeds]
        else:
            cloned_text_embeds = self.text_embeds.clone()
        if self.pooled_embeds is not None:
            prompt_embeds = PromptEmbeds([cloned_text_embeds, self.pooled_embeds.clone()])
        else:
            prompt_embeds = PromptEmbeds(cloned_text_embeds)

        if self.attention_mask is not None:
            if isinstance(self.attention_mask, list) or isinstance(self.attention_mask, tuple):
                prompt_embeds.attention_mask = [t.clone() for t in self.attention_mask]
            else:
                prompt_embeds.attention_mask = self.attention_mask.clone()
        return prompt_embeds

    def expand_to_batch(self, batch_size):
        pe = self.clone()
        if isinstance(pe.text_embeds, list) or isinstance(pe.text_embeds, tuple):
            current_batch_size = pe.text_embeds[0].shape[0]
        else:
            current_batch_size = pe.text_embeds.shape[0]
        if current_batch_size == batch_size:
            return pe
        if current_batch_size != 1:
            raise Exception("Can only expand batch size for batch size 1")
        if isinstance(pe.text_embeds, list) or isinstance(pe.text_embeds, tuple):
            pe.text_embeds = [t.expand(batch_size, -1) for t in pe.text_embeds]
        else:
            pe.text_embeds = pe.text_embeds.expand(batch_size, -1)
        if pe.pooled_embeds is not None:
            pe.pooled_embeds = pe.pooled_embeds.expand(batch_size, -1)
        if pe.attention_mask is not None:
            if isinstance(pe.attention_mask, list) or isinstance(pe.attention_mask, tuple):
                pe.attention_mask = [t.expand(batch_size, -1) for t in pe.attention_mask]
            else:
                pe.attention_mask = pe.attention_mask.expand(batch_size, -1)
        return pe

    def save(self, path: str):
        """
        Save the prompt embeds to a file.
        :param path: The path to save the prompt embeds.
        """
        pe = self.clone()
        state_dict = {}
        if isinstance(pe.text_embeds, list) or isinstance(pe.text_embeds, tuple):
            for i, text_embed in enumerate(pe.text_embeds):
                state_dict[f"text_embed_{i}"] = text_embed.cpu()
        else:
            state_dict["text_embed"] = pe.text_embeds.cpu()
            
        if pe.pooled_embeds is not None:
            state_dict["pooled_embed"] = pe.pooled_embeds.cpu()
        if pe.attention_mask is not None:
            if isinstance(pe.attention_mask, list) or isinstance(pe.attention_mask, tuple):
                for i, attn in enumerate(pe.attention_mask):
                    state_dict[f"attention_mask_{i}"] = attn.cpu()
            else:
                state_dict["attention_mask"] = pe.attention_mask.cpu()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(state_dict, path)
    
    @classmethod
    def load(cls, path: str) -> 'PromptEmbeds':
        """
        Load the prompt embeds from a file.
        :param path: The path to load the prompt embeds from.
        :return: An instance of PromptEmbeds.
        """
        state_dict = load_file(path, device='cpu')
        text_embeds = []
        pooled_embeds = None
        attention_mask = []
        for key in sorted(state_dict.keys()):
            if key.startswith("text_embed_"):
                text_embeds.append(state_dict[key])
            elif key == "text_embed":
                text_embeds.append(state_dict[key])
            elif key == "pooled_embed":
                pooled_embeds = state_dict[key]
            elif key.startswith("attention_mask_"):
                attention_mask.append(state_dict[key])
            elif key == "attention_mask":
                attention_mask.append(state_dict[key])
        pe = cls(None)
        pe.text_embeds = text_embeds
        if len(text_embeds) == 1:
            pe.text_embeds = text_embeds[0]
        if pooled_embeds is not None:
            pe.pooled_embeds = pooled_embeds
        if len(attention_mask) > 0:
            if len(attention_mask) == 1:
                pe.attention_mask = attention_mask[0]
            else:
                pe.attention_mask = attention_mask
        return pe



class EncodedPromptPair:
    def __init__(
            self,
            target_class,
            target_class_with_neutral,
            positive_target,
            positive_target_with_neutral,
            negative_target,
            negative_target_with_neutral,
            neutral,
            empty_prompt,
            both_targets,
            action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
            action_list=None,
            multiplier=1.0,
            multiplier_list=None,
            weight=1.0,
            target: 'SliderTargetConfig' = None,
    ):
        self.target_class: PromptEmbeds = target_class
        self.target_class_with_neutral: PromptEmbeds = target_class_with_neutral
        self.positive_target: PromptEmbeds = positive_target
        self.positive_target_with_neutral: PromptEmbeds = positive_target_with_neutral
        self.negative_target: PromptEmbeds = negative_target
        self.negative_target_with_neutral: PromptEmbeds = negative_target_with_neutral
        self.neutral: PromptEmbeds = neutral
        self.empty_prompt: PromptEmbeds = empty_prompt
        self.both_targets: PromptEmbeds = both_targets
        self.multiplier: float = multiplier
        self.target: 'SliderTargetConfig' = target
        if multiplier_list is not None:
            self.multiplier_list: list[float] = multiplier_list
        else:
            self.multiplier_list: list[float] = [multiplier]
        self.action: int = action
        if action_list is not None:
            self.action_list: list[int] = action_list
        else:
            self.action_list: list[int] = [action]
        self.weight: float = weight

    # simulate torch to for tensors
    def to(self, *args, **kwargs):
        self.target_class = self.target_class.to(*args, **kwargs)
        self.target_class_with_neutral = self.target_class_with_neutral.to(*args, **kwargs)
        self.positive_target = self.positive_target.to(*args, **kwargs)
        self.positive_target_with_neutral = self.positive_target_with_neutral.to(*args, **kwargs)
        self.negative_target = self.negative_target.to(*args, **kwargs)
        self.negative_target_with_neutral = self.negative_target_with_neutral.to(*args, **kwargs)
        self.neutral = self.neutral.to(*args, **kwargs)
        self.empty_prompt = self.empty_prompt.to(*args, **kwargs)
        self.both_targets = self.both_targets.to(*args, **kwargs)
        return self

    def detach(self):
        self.target_class = self.target_class.detach()
        self.target_class_with_neutral = self.target_class_with_neutral.detach()
        self.positive_target = self.positive_target.detach()
        self.positive_target_with_neutral = self.positive_target_with_neutral.detach()
        self.negative_target = self.negative_target.detach()
        self.negative_target_with_neutral = self.negative_target_with_neutral.detach()
        self.neutral = self.neutral.detach()
        self.empty_prompt = self.empty_prompt.detach()
        self.both_targets = self.both_targets.detach()
        return self


def concat_prompt_embeds(prompt_embeds: list["PromptEmbeds"]):
    # --- pad text_embeds ---
    if isinstance(prompt_embeds[0].text_embeds, (list, tuple)):
        embed_list = []
        for i in range(len(prompt_embeds[0].text_embeds)):
            max_len = max(p.text_embeds[i].shape[1] for p in prompt_embeds)
            padded = []
            for p in prompt_embeds:
                t = p.text_embeds[i]
                if t.shape[1] < max_len:
                    pad = torch.zeros(
                        (t.shape[0], max_len - t.shape[1], *t.shape[2:]),
                        dtype=t.dtype,
                        device=t.device,
                    )
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            embed_list.append(torch.cat(padded, dim=0))
        text_embeds = embed_list
    else:
        max_len = max(p.text_embeds.shape[1] for p in prompt_embeds)
        padded = []
        for p in prompt_embeds:
            t = p.text_embeds
            if t.shape[1] < max_len:
                pad = torch.zeros(
                    (t.shape[0], max_len - t.shape[1], *t.shape[2:]),
                    dtype=t.dtype,
                    device=t.device,
                )
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        text_embeds = torch.cat(padded, dim=0)

    # --- pooled embeds ---
    pooled_embeds = None
    if prompt_embeds[0].pooled_embeds is not None:
        pooled_embeds = torch.cat([p.pooled_embeds for p in prompt_embeds], dim=0)

    # --- attention mask ---
    attention_mask = None
    if prompt_embeds[0].attention_mask is not None:
        max_len = max(p.attention_mask.shape[1] for p in prompt_embeds)
        padded = []
        for p in prompt_embeds:
            m = p.attention_mask
            if m.shape[1] < max_len:
                pad = torch.zeros(
                    (m.shape[0], max_len - m.shape[1]),
                    dtype=m.dtype,
                    device=m.device,
                )
                m = torch.cat([m, pad], dim=1)
            padded.append(m)
        attention_mask = torch.cat(padded, dim=0)

    # wrap back into PromptEmbeds
    pe = PromptEmbeds([text_embeds, pooled_embeds])
    pe.attention_mask = attention_mask
    return pe


def concat_prompt_pairs(prompt_pairs: list[EncodedPromptPair]):
    weight = prompt_pairs[0].weight
    target_class = concat_prompt_embeds([p.target_class for p in prompt_pairs])
    target_class_with_neutral = concat_prompt_embeds([p.target_class_with_neutral for p in prompt_pairs])
    positive_target = concat_prompt_embeds([p.positive_target for p in prompt_pairs])
    positive_target_with_neutral = concat_prompt_embeds([p.positive_target_with_neutral for p in prompt_pairs])
    negative_target = concat_prompt_embeds([p.negative_target for p in prompt_pairs])
    negative_target_with_neutral = concat_prompt_embeds([p.negative_target_with_neutral for p in prompt_pairs])
    neutral = concat_prompt_embeds([p.neutral for p in prompt_pairs])
    empty_prompt = concat_prompt_embeds([p.empty_prompt for p in prompt_pairs])
    both_targets = concat_prompt_embeds([p.both_targets for p in prompt_pairs])
    # combine all the lists
    action_list = []
    multiplier_list = []
    weight_list = []
    for p in prompt_pairs:
        action_list += p.action_list
        multiplier_list += p.multiplier_list
    return EncodedPromptPair(
        target_class=target_class,
        target_class_with_neutral=target_class_with_neutral,
        positive_target=positive_target,
        positive_target_with_neutral=positive_target_with_neutral,
        negative_target=negative_target,
        negative_target_with_neutral=negative_target_with_neutral,
        neutral=neutral,
        empty_prompt=empty_prompt,
        both_targets=both_targets,
        action_list=action_list,
        multiplier_list=multiplier_list,
        weight=weight,
        target=prompt_pairs[0].target
    )


def split_prompt_embeds(concatenated: PromptEmbeds, num_parts=None) -> List[PromptEmbeds]:
    if num_parts is None:
        # use batch size
        num_parts = concatenated.text_embeds.shape[0]
        
    if isinstance(concatenated.text_embeds, list) or isinstance(concatenated.text_embeds, tuple):
        # split each part
        text_embeds_splits = [
            torch.chunk(text, num_parts, dim=0)
            for text in concatenated.text_embeds
        ]
        text_embeds_splits = list(zip(*text_embeds_splits))
    else:
        text_embeds_splits = torch.chunk(concatenated.text_embeds, num_parts, dim=0)

    if concatenated.pooled_embeds is not None:
        pooled_embeds_splits = torch.chunk(concatenated.pooled_embeds, num_parts, dim=0)
    else:
        pooled_embeds_splits = [None] * num_parts

    prompt_embeds_list = [
        PromptEmbeds([text, pooled])
        for text, pooled in zip(text_embeds_splits, pooled_embeds_splits)
    ]

    return prompt_embeds_list


def split_prompt_pairs(concatenated: EncodedPromptPair, num_embeds=None) -> List[EncodedPromptPair]:
    target_class_splits = split_prompt_embeds(concatenated.target_class, num_embeds)
    target_class_with_neutral_splits = split_prompt_embeds(concatenated.target_class_with_neutral, num_embeds)
    positive_target_splits = split_prompt_embeds(concatenated.positive_target, num_embeds)
    positive_target_with_neutral_splits = split_prompt_embeds(concatenated.positive_target_with_neutral, num_embeds)
    negative_target_splits = split_prompt_embeds(concatenated.negative_target, num_embeds)
    negative_target_with_neutral_splits = split_prompt_embeds(concatenated.negative_target_with_neutral, num_embeds)
    neutral_splits = split_prompt_embeds(concatenated.neutral, num_embeds)
    empty_prompt_splits = split_prompt_embeds(concatenated.empty_prompt, num_embeds)
    both_targets_splits = split_prompt_embeds(concatenated.both_targets, num_embeds)

    prompt_pairs = []
    for i in range(len(target_class_splits)):
        action_list_split = concatenated.action_list[i::len(target_class_splits)]
        multiplier_list_split = concatenated.multiplier_list[i::len(target_class_splits)]

        prompt_pair = EncodedPromptPair(
            target_class=target_class_splits[i],
            target_class_with_neutral=target_class_with_neutral_splits[i],
            positive_target=positive_target_splits[i],
            positive_target_with_neutral=positive_target_with_neutral_splits[i],
            negative_target=negative_target_splits[i],
            negative_target_with_neutral=negative_target_with_neutral_splits[i],
            neutral=neutral_splits[i],
            empty_prompt=empty_prompt_splits[i],
            both_targets=both_targets_splits[i],
            action_list=action_list_split,
            multiplier_list=multiplier_list_split,
            weight=concatenated.weight,
            target=concatenated.target
        )
        prompt_pairs.append(prompt_pair)

    return prompt_pairs


class PromptEmbedsCache:
    prompts: dict[str, PromptEmbeds] = {}

    def __setitem__(self, __name: str, __value: PromptEmbeds) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str) -> Optional[PromptEmbeds]:
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


class EncodedAnchor:
    def __init__(
            self,
            prompt,
            neg_prompt,
            multiplier=1.0,
            multiplier_list=None
    ):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.multiplier = multiplier

        if multiplier_list is not None:
            self.multiplier_list: list[float] = multiplier_list
        else:
            self.multiplier_list: list[float] = [multiplier]

    def to(self, *args, **kwargs):
        self.prompt = self.prompt.to(*args, **kwargs)
        self.neg_prompt = self.neg_prompt.to(*args, **kwargs)
        return self


def concat_anchors(anchors: list[EncodedAnchor]):
    prompt = concat_prompt_embeds([a.prompt for a in anchors])
    neg_prompt = concat_prompt_embeds([a.neg_prompt for a in anchors])
    return EncodedAnchor(
        prompt=prompt,
        neg_prompt=neg_prompt,
        multiplier_list=[a.multiplier for a in anchors]
    )


def split_anchors(concatenated: EncodedAnchor, num_anchors: int = 4) -> List[EncodedAnchor]:
    prompt_splits = split_prompt_embeds(concatenated.prompt, num_anchors)
    neg_prompt_splits = split_prompt_embeds(concatenated.neg_prompt, num_anchors)
    multiplier_list_splits = torch.chunk(torch.tensor(concatenated.multiplier_list), num_anchors)

    anchors = []
    for prompt, neg_prompt, multiplier in zip(prompt_splits, neg_prompt_splits, multiplier_list_splits):
        anchor = EncodedAnchor(
            prompt=prompt,
            neg_prompt=neg_prompt,
            multiplier=multiplier.tolist()
        )
        anchors.append(anchor)

    return anchors


def get_permutations(s, max_permutations=8):
    # Split the string by comma
    phrases = [phrase.strip() for phrase in s.split(',')]

    # remove empty strings
    phrases = [phrase for phrase in phrases if len(phrase) > 0]
    # shuffle the list
    random.shuffle(phrases)

    # Get all permutations
    permutations = list([p for p in itertools.islice(itertools.permutations(phrases), max_permutations)])

    # Convert the tuples back to comma separated strings
    return [', '.join(permutation) for permutation in permutations]


def get_slider_target_permutations(target: 'SliderTargetConfig', max_permutations=8) -> List['SliderTargetConfig']:
    from toolkit.config_modules import SliderTargetConfig
    pos_permutations = get_permutations(target.positive, max_permutations=max_permutations)
    neg_permutations = get_permutations(target.negative, max_permutations=max_permutations)

    permutations = []
    for pos, neg in itertools.product(pos_permutations, neg_permutations):
        permutations.append(
            SliderTargetConfig(
                target_class=target.target_class,
                positive=pos,
                negative=neg,
                multiplier=target.multiplier,
                weight=target.weight
            )
        )

    # shuffle the list
    random.shuffle(permutations)

    if len(permutations) > max_permutations:
        permutations = permutations[:max_permutations]

    return permutations


if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion


@torch.no_grad()
def encode_prompts_to_cache(
        prompt_list: list[str],
        sd: "StableDiffusion",
        cache: Optional[PromptEmbedsCache] = None,
        prompt_tensor_file: Optional[str] = None,
) -> PromptEmbedsCache:
    # TODO: add support for larger prompts
    if cache is None:
        cache = PromptEmbedsCache()

    if prompt_tensor_file is not None:
        # check to see if it exists
        if os.path.exists(prompt_tensor_file):
            # load it.
            print(f"Loading prompt tensors from {prompt_tensor_file}")
            prompt_tensors = load_file(prompt_tensor_file, device='cpu')
            # add them to the cache
            for prompt_txt, prompt_tensor in tqdm(prompt_tensors.items(), desc="Loading prompts", leave=False):
                if prompt_txt.startswith("te:"):
                    prompt = prompt_txt[3:]
                    # text_embeds
                    text_embeds = prompt_tensor
                    pooled_embeds = None
                    # find pool embeds
                    if f"pe:{prompt}" in prompt_tensors:
                        pooled_embeds = prompt_tensors[f"pe:{prompt}"]

                    # make it
                    prompt_embeds = PromptEmbeds([text_embeds, pooled_embeds])
                    cache[prompt] = prompt_embeds.to(device='cpu', dtype=torch.float32)

    if len(cache.prompts) == 0:
        print("Prompt tensors not found. Encoding prompts..")
        empty_prompt = ""
        # encode empty_prompt
        cache[empty_prompt] = sd.encode_prompt(empty_prompt)

        for p in tqdm(prompt_list, desc="Encoding prompts", leave=False):
            # build the cache
            if cache[p] is None:
                cache[p] = sd.encode_prompt(p).to(device="cpu", dtype=torch.float16)

        # should we shard? It can get large
        if prompt_tensor_file:
            print(f"Saving prompt tensors to {prompt_tensor_file}")
            state_dict = {}
            for prompt_txt, prompt_embeds in cache.prompts.items():
                state_dict[f"te:{prompt_txt}"] = prompt_embeds.text_embeds.to(
                    "cpu", dtype=get_torch_dtype('fp16')
                )
                if prompt_embeds.pooled_embeds is not None:
                    state_dict[f"pe:{prompt_txt}"] = prompt_embeds.pooled_embeds.to(
                        "cpu",
                        dtype=get_torch_dtype('fp16')
                    )
            save_file(state_dict, prompt_tensor_file)

    return cache


@torch.no_grad()
def build_prompt_pair_batch_from_cache(
        cache: PromptEmbedsCache,
        target: 'SliderTargetConfig',
        neutral: Optional[str] = '',
) -> list[EncodedPromptPair]:
    erase_negative = len(target.positive.strip()) == 0
    enhance_positive = len(target.negative.strip()) == 0

    both = not erase_negative and not enhance_positive

    prompt_pair_batch = []

    if both or erase_negative:
        # print("Encoding erase negative")
        prompt_pair_batch += [
            # erase standard
            EncodedPromptPair(
                target_class=cache[target.target_class],
                target_class_with_neutral=cache[f"{target.target_class} {neutral}"],
                positive_target=cache[f"{target.positive}"],
                positive_target_with_neutral=cache[f"{target.positive} {neutral}"],
                negative_target=cache[f"{target.negative}"],
                negative_target_with_neutral=cache[f"{target.negative} {neutral}"],
                neutral=cache[neutral],
                action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
                multiplier=target.multiplier,
                both_targets=cache[f"{target.positive} {target.negative}"],
                empty_prompt=cache[""],
                weight=target.weight,
                target=target
            ),
        ]
    if both or enhance_positive:
        # print("Encoding enhance positive")
        prompt_pair_batch += [
            # enhance standard, swap pos neg
            EncodedPromptPair(
                target_class=cache[target.target_class],
                target_class_with_neutral=cache[f"{target.target_class} {neutral}"],
                positive_target=cache[f"{target.negative}"],
                positive_target_with_neutral=cache[f"{target.negative} {neutral}"],
                negative_target=cache[f"{target.positive}"],
                negative_target_with_neutral=cache[f"{target.positive} {neutral}"],
                neutral=cache[neutral],
                action=ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE,
                multiplier=target.multiplier,
                both_targets=cache[f"{target.positive} {target.negative}"],
                empty_prompt=cache[""],
                weight=target.weight,
                target=target
            ),
        ]
    if both or enhance_positive:
        # print("Encoding erase positive (inverse)")
        prompt_pair_batch += [
            # erase inverted
            EncodedPromptPair(
                target_class=cache[target.target_class],
                target_class_with_neutral=cache[f"{target.target_class} {neutral}"],
                positive_target=cache[f"{target.negative}"],
                positive_target_with_neutral=cache[f"{target.negative} {neutral}"],
                negative_target=cache[f"{target.positive}"],
                negative_target_with_neutral=cache[f"{target.positive} {neutral}"],
                neutral=cache[neutral],
                action=ACTION_TYPES_SLIDER.ERASE_NEGATIVE,
                both_targets=cache[f"{target.positive} {target.negative}"],
                empty_prompt=cache[""],
                multiplier=target.multiplier * -1.0,
                weight=target.weight,
                target=target
            ),
        ]
    if both or erase_negative:
        # print("Encoding enhance negative (inverse)")
        prompt_pair_batch += [
            # enhance inverted
            EncodedPromptPair(
                target_class=cache[target.target_class],
                target_class_with_neutral=cache[f"{target.target_class} {neutral}"],
                positive_target=cache[f"{target.positive}"],
                positive_target_with_neutral=cache[f"{target.positive} {neutral}"],
                negative_target=cache[f"{target.negative}"],
                negative_target_with_neutral=cache[f"{target.negative} {neutral}"],
                both_targets=cache[f"{target.positive} {target.negative}"],
                neutral=cache[neutral],
                action=ACTION_TYPES_SLIDER.ENHANCE_NEGATIVE,
                empty_prompt=cache[""],
                multiplier=target.multiplier * -1.0,
                weight=target.weight,
                target=target
            ),
        ]

    return prompt_pair_batch


def build_latent_image_batch_for_prompt_pair(
        pos_latent,
        neg_latent,
        prompt_pair: EncodedPromptPair,
        prompt_chunk_size
):
    erase_negative = len(prompt_pair.target.positive.strip()) == 0
    enhance_positive = len(prompt_pair.target.negative.strip()) == 0
    both = not erase_negative and not enhance_positive

    prompt_pair_chunks = split_prompt_pairs(prompt_pair, prompt_chunk_size)
    if both and len(prompt_pair_chunks) != 4:
        raise Exception("Invalid prompt pair chunks")
    if (erase_negative or enhance_positive) and len(prompt_pair_chunks) != 2:
        raise Exception("Invalid prompt pair chunks")

    latent_list = []

    if both or erase_negative:
        latent_list.append(pos_latent)
    if both or enhance_positive:
        latent_list.append(pos_latent)
    if both or enhance_positive:
        latent_list.append(neg_latent)
    if both or erase_negative:
        latent_list.append(neg_latent)

    return torch.cat(latent_list, dim=0)


def inject_trigger_into_prompt(prompt, trigger=None, to_replace_list=None, add_if_not_present=True):
    if trigger is None:
        # process as empty string to remove any [trigger] tokens
        trigger = ''
    output_prompt = prompt
    default_replacements = ["[name]", "[trigger]"]

    replace_with = trigger
    if to_replace_list is None:
        to_replace_list = default_replacements
    else:
        to_replace_list += default_replacements

    # remove duplicates
    to_replace_list = list(set(to_replace_list))

    # replace them all
    for to_replace in to_replace_list:
        # replace it
        output_prompt = output_prompt.replace(to_replace, replace_with)

    if trigger.strip() != "":
        # see how many times replace_with is in the prompt
        num_instances = output_prompt.count(replace_with)

        if num_instances == 0 and add_if_not_present:
            # add it to the beginning of the prompt
            output_prompt = replace_with + " " + output_prompt

        # if num_instances > 1:
        #     print(
        #         f"Warning: {trigger} token appears {num_instances} times in prompt {output_prompt}. This may cause issues.")

    return output_prompt
