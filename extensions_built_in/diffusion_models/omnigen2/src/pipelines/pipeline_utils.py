import torch


def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    # count_prompt = len(prompt.split(" "))
    # count_negative_prompt = len(negative_prompt.split(" "))

    # create the tensor based on which prompt is longer
    # if count_prompt >= count_negative_prompt:
    input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding='longest').input_ids.to(device)
    # input_ids = pipeline.tokenizer(prompt, padding="max_length",
    #             max_length=pipeline.tokenizer.model_max_length,
    #             truncation=True,
    #             return_tensors="pt",).input_ids.to(device)
    shape_max_length = input_ids.shape[-1]

    if negative_prompt is not None:
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=True, padding="max_length",
                                        max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    # else:
    #     negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
    #     shape_max_length = negative_ids.shape[-1]
    #     input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
    #                                    max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        if hasattr(pipeline.text_encoder.config, "use_attention_mask") and pipeline.text_encoder.config.use_attention_mask:
            attention_mask = input_ids[:, i: i + max_length].attention_mask.to(device)
        else:
            attention_mask = None
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length],
                                                   attention_mask=attention_mask)[0])
        
        if negative_prompt is not None:
            if hasattr(pipeline.text_encoder.config, "use_attention_mask") and pipeline.text_encoder.config.use_attention_mask:
                attention_mask = negative_ids[:, i: i + max_length].attention_mask.to(device)
            else:
                attention_mask = None
            neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length],
                                                    attention_mask=attention_mask)[0])

    concat_embeds = torch.cat(concat_embeds, dim=1)

    if negative_prompt is not None:
        neg_embeds = torch.cat(neg_embeds, dim=1)
    else:
        neg_embeds = None

    return concat_embeds, neg_embeds
