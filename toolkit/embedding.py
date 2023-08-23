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
            embed_config: 'EmbeddingConfig'
    ):
        self.name = embed_config.trigger
        self.sd = sd
        self.embed_config = embed_config
        # setup our embedding
        # Add the placeholder token in tokenizer
        placeholder_tokens = [self.embed_config.trigger]

        # add dummy tokens for multi-vector
        additional_tokens = []
        for i in range(1, self.embed_config.tokens):
            additional_tokens.append(f"{self.embed_config.trigger}_{i}")
        placeholder_tokens += additional_tokens

        num_added_tokens = self.sd.tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != self.embed_config.tokens:
            raise ValueError(
                f"The tokenizer already contains the token {self.embed_config.trigger}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        # Convert the initializer_token, placeholder_token to ids
        init_token_ids = self.sd.tokenizer.encode(self.embed_config.init_words, add_special_tokens=False)
        # if length of token ids is more than number of orm embedding tokens fill with *
        if len(init_token_ids) > self.embed_config.tokens:
            init_token_ids = init_token_ids[:self.embed_config.tokens]
        elif len(init_token_ids) < self.embed_config.tokens:
            pad_token_id = self.sd.tokenizer.encode(["*"], add_special_tokens=False)
            init_token_ids += pad_token_id * (self.embed_config.tokens - len(init_token_ids))

        self.placeholder_token_ids = self.sd.tokenizer.convert_tokens_to_ids(placeholder_tokens)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        # todo SDXL has 2 text encoders, need to do both for all of this
        self.sd.text_encoder.resize_token_embeddings(len(self.sd.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.sd.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for initializer_token_id, token_id in zip(init_token_ids, self.placeholder_token_ids):
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()

        # this doesnt seem to be used again
        self.token_embeds = token_embeds

        # replace "[name] with this. This triggers it in the text encoder
        self.embedding_tokens = " ".join(self.sd.tokenizer.convert_ids_to_tokens(self.placeholder_token_ids))

    # returns the string to have in the prompt to trigger the embedding
    def get_embedding_string(self):
        return self.embedding_tokens

    def get_trainable_params(self):
        # todo only get this one as we could have more than one
        return self.sd.text_encoder.get_input_embeddings().parameters()

    # make setter and getter for vec
    @property
    def vec(self):
        # should we get params instead
        # create vector from token embeds
        token_embeds = self.sd.text_encoder.get_input_embeddings().weight.data
        # stack the tokens along batch axis adding that axis
        new_vector = torch.stack(
            [token_embeds[token_id].unsqueeze(0) for token_id in self.placeholder_token_ids],
            dim=0
        )
        return new_vector

    @vec.setter
    def vec(self, new_vector):
        # shape is (1, 768) for SD 1.5 for 1 token
        token_embeds = self.sd.text_encoder.get_input_embeddings().weight.data
        for i in range(new_vector.shape[0]):
            # apply the weights to the placeholder tokens while preserving gradient
            token_embeds[self.placeholder_token_ids[i]] = new_vector[i].clone()
            x = 1

    def save(self, filename):
        # todo check to see how to get the vector out of the embedding

        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": 0,
            # todo get these
            "sd_checkpoint": None,
            "sd_checkpoint_name": None,
            "notes": None,
        }
        if filename.endswith('.pt'):
            torch.save(embedding_data, filename)
        elif filename.endswith('.bin'):
            torch.save(embedding_data, filename)
        elif filename.endswith('.safetensors'):
            # save the embedding as a safetensors file
            state_dict = {"emb_params": self.vec}
            # add all embedding data (except string_to_param), to metadata
            metadata = OrderedDict({k: json.dumps(v) for k, v in embedding_data.items() if k != "string_to_param"})
            metadata["string_to_param"] = {"*": "emb_params"}
            save_meta = get_meta_for_safetensors(metadata, name=self.name)
            save_file(state_dict, filename, metadata=save_meta)

    def load_embedding_from_file(self, file_path, device):
        # full path
        path = os.path.realpath(file_path)
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        ext = ext.upper()
        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            _, second_ext = os.path.splitext(name)
            if second_ext.upper() == '.PREVIEW':
                return

        if ext in ['.BIN', '.PT']:
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']:
            # rebuild the embedding from the safetensors file if it has it
            tensors = {}
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

        self.vec = emb.detach().to(device, dtype=torch.float32)
