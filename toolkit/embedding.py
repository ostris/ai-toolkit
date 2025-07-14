import json
import os
from collections import OrderedDict

import safetensors
import torch
from typing import TYPE_CHECKING

from safetensors.torch import save_file

from toolkit.metadata import get_meta_for_safetensors

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.config_modules import EmbeddingConfig


# this is a frankenstein mix of automatic1111 and my own code

class Embedding:
    def __init__(
            self,
            sd: 'StableDiffusion',
            embed_config: 'EmbeddingConfig',
            state_dict: OrderedDict = None,
    ):
        self.name = embed_config.trigger
        self.sd = sd
        self.trigger = embed_config.trigger
        self.embed_config = embed_config
        self.step = 0
        # setup our embedding
        # Add the placeholder token in tokenizer
        placeholder_tokens = [self.embed_config.trigger]

        # add dummy tokens for multi-vector
        additional_tokens = []
        for i in range(1, self.embed_config.tokens):
            additional_tokens.append(f"{self.embed_config.trigger}_{i}")
        placeholder_tokens += additional_tokens

        # handle dual tokenizer
        self.tokenizer_list = self.sd.tokenizer if isinstance(self.sd.tokenizer, list) else [self.sd.tokenizer]
        self.text_encoder_list = self.sd.text_encoder if isinstance(self.sd.text_encoder, list) else [
            self.sd.text_encoder]

        self.placeholder_token_ids = []
        self.embedding_tokens = []

        print(f"Adding {placeholder_tokens} tokens to tokenizer")
        print(f"Adding {self.embed_config.tokens} tokens to tokenizer")

        for text_encoder, tokenizer in zip(self.text_encoder_list, self.tokenizer_list):
            num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
            if num_added_tokens != self.embed_config.tokens:
                raise ValueError(
                    f"The tokenizer already contains the token {self.embed_config.trigger}. Please pass a different"
                    f" `placeholder_token` that is not already in the tokenizer. Only added {num_added_tokens}"
                )

            # Convert the initializer_token, placeholder_token to ids
            init_token_ids = tokenizer.encode(self.embed_config.init_words, add_special_tokens=False)
            # if length of token ids is more than number of orm embedding tokens fill with *
            if len(init_token_ids) > self.embed_config.tokens:
                init_token_ids = init_token_ids[:self.embed_config.tokens]
            elif len(init_token_ids) < self.embed_config.tokens:
                pad_token_id = tokenizer.encode(["*"], add_special_tokens=False)
                init_token_ids += pad_token_id * (self.embed_config.tokens - len(init_token_ids))

            placeholder_token_ids = tokenizer.encode(placeholder_tokens, add_special_tokens=False)
            self.placeholder_token_ids.append(placeholder_token_ids)

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = text_encoder.get_input_embeddings().weight.data
            with torch.no_grad():
                for initializer_token_id, token_id in zip(init_token_ids, placeholder_token_ids):
                    token_embeds[token_id] = token_embeds[initializer_token_id].clone()

            # replace "[name] with this. on training. This is automatically generated in pipeline on inference
            self.embedding_tokens.append(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)))

        # backup text encoder embeddings
        self.orig_embeds_params = [x.get_input_embeddings().weight.data.clone() for x in self.text_encoder_list]

    def restore_embeddings(self):
        with torch.no_grad():
            # Let's make sure we don't update any embedding weights besides the newly added token
            for text_encoder, tokenizer, orig_embeds, placeholder_token_ids in zip(self.text_encoder_list,
                                                                                   self.tokenizer_list,
                                                                                   self.orig_embeds_params,
                                                                                   self.placeholder_token_ids):
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[ min(placeholder_token_ids): max(placeholder_token_ids) + 1] = False
                text_encoder.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds[index_no_updates]
                weight = text_encoder.get_input_embeddings().weight
                pass

    def get_trainable_params(self):
        params = []
        for text_encoder in self.text_encoder_list:
            params += text_encoder.get_input_embeddings().parameters()
        return params

    def _get_vec(self, text_encoder_idx=0):
        # should we get params instead
        # create vector from token embeds
        token_embeds = self.text_encoder_list[text_encoder_idx].get_input_embeddings().weight.data
        # stack the tokens along batch axis adding that axis
        new_vector = torch.stack(
            [token_embeds[token_id] for token_id in self.placeholder_token_ids[text_encoder_idx]],
            dim=0
        )
        return new_vector

    def _set_vec(self, new_vector, text_encoder_idx=0):
        # shape is (1, 768) for SD 1.5 for 1 token
        token_embeds = self.text_encoder_list[text_encoder_idx].get_input_embeddings().weight.data
        for i in range(new_vector.shape[0]):
            # apply the weights to the placeholder tokens while preserving gradient
            token_embeds[self.placeholder_token_ids[text_encoder_idx][i]] = new_vector[i].clone()

    # make setter and getter for vec
    @property
    def vec(self):
        return self._get_vec(0)

    @vec.setter
    def vec(self, new_vector):
        self._set_vec(new_vector, 0)

    @property
    def vec2(self):
        return self._get_vec(1)

    @vec2.setter
    def vec2(self, new_vector):
        self._set_vec(new_vector, 1)

    # diffusers automatically expands the token meaning test123 becomes test123 test123_1 test123_2 etc
    # however, on training we don't use that pipeline, so we have to do it ourselves
    def inject_embedding_to_prompt(self, prompt, expand_token=False, to_replace_list=None, add_if_not_present=True):
        output_prompt = prompt
        embedding_tokens = self.embedding_tokens[0]  # shoudl be the same
        default_replacements = ["[name]", "[trigger]"]

        replace_with = embedding_tokens if expand_token else self.trigger
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

        # see how many times replace_with is in the prompt
        num_instances = output_prompt.count(replace_with)

        if num_instances == 0 and add_if_not_present:
            # add it to the beginning of the prompt
            output_prompt = replace_with + " " + output_prompt

        if num_instances > 1:
            print(
                f"Warning: {replace_with} token appears {num_instances} times in prompt {output_prompt}. This may cause issues.")

        return output_prompt

    def state_dict(self):
        if self.sd.is_xl:
            state_dict = OrderedDict()
            state_dict['clip_l'] = self.vec
            state_dict['clip_g'] = self.vec2
        else:
            state_dict = OrderedDict()
            state_dict['emb_params'] = self.vec

        return state_dict

    def save(self, filename):
        # todo check to see how to get the vector out of the embedding

        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            # todo get these
            "sd_checkpoint": None,
            "sd_checkpoint_name": None,
            "notes": None,
        }
        # TODO we do not currently support this. Check how auto is doing it. Only safetensors supported sor sdxl
        if filename.endswith('.pt'):
            torch.save(embedding_data, filename)
        elif filename.endswith('.bin'):
            torch.save(embedding_data, filename)
        elif filename.endswith('.safetensors'):
            # save the embedding as a safetensors file
            state_dict = self.state_dict()
            # add all embedding data (except string_to_param), to metadata
            metadata = OrderedDict({k: json.dumps(v, ensure_ascii=False) for k, v in embedding_data.items() if k != "string_to_param"})
            metadata["string_to_param"] = {"*": "emb_params"}
            save_meta = get_meta_for_safetensors(metadata, name=self.name)
            save_file(state_dict, filename, metadata=save_meta)

    def load_embedding_from_file(self, file_path, device):
        # full path
        path = os.path.realpath(file_path)
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        tensors = {}
        ext = ext.upper()
        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            _, second_ext = os.path.splitext(name)
            if second_ext.upper() == '.PREVIEW':
                return

        if ext in ['.BIN', '.PT']:
            # todo check this
            if self.sd.is_xl:
               raise Exception("XL not supported yet for bin, pt")
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']:
            # rebuild the embedding from the safetensors file if it has it
            with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            # data = safetensors.torch.load_file(path, device="cpu")
            if metadata and 'string_to_param' in metadata and 'emb_params' in tensors:
                # our format
                def try_json(v):
                    try:
                        return json.loads(v)
                    except:
                        return v

                data = {k: try_json(v) for k, v in metadata.items()}
                data['string_to_param'] = {'*': tensors['emb_params']}
            else:
                # old format
                data = tensors
        else:
            return

        if self.sd.is_xl:
            self.vec = tensors['clip_l'].detach().to(device, dtype=torch.float32)
            self.vec2 = tensors['clip_g'].detach().to(device, dtype=torch.float32)
            if 'step' in data:
                self.step = int(data['step'])
        else:
            # textual inversion embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict,
                                         '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            # diffuser concepts
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)
            else:
                raise Exception(
                    f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

            if 'step' in data:
                self.step = int(data['step'])

            self.vec = emb.detach().to(device, dtype=torch.float32)
