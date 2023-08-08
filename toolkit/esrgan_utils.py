
to_basicsr_dict = {
    'model.0.weight': 'conv_first.weight',
    'model.0.bias': 'conv_first.bias',
    'model.1.sub.23.weight': 'conv_body.weight',
    'model.1.sub.23.bias': 'conv_body.bias',
    'model.3.weight': 'conv_up1.weight',
    'model.3.bias': 'conv_up1.bias',
    'model.6.weight': 'conv_up2.weight',
    'model.6.bias': 'conv_up2.bias',
    'model.8.weight': 'conv_hr.weight',
    'model.8.bias': 'conv_hr.bias',
    'model.10.bias': 'conv_last.bias',
    'model.10.weight': 'conv_last.weight',
    # 'model.1.sub.0.RDB1.conv1.0.weight': 'body.0.rdb1.conv1.weight'
}

def convert_state_dict_to_basicsr(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in to_basicsr_dict:
            new_state_dict[to_basicsr_dict[k]] = v
        elif k.startswith('model.1.sub.'):
            bsr_name = k.replace('model.1.sub.', 'body.').lower()
            bsr_name = bsr_name.replace('.0.weight', '.weight')
            bsr_name = bsr_name.replace('.0.bias', '.bias')
            new_state_dict[bsr_name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# just matching a commonly used format
def convert_basicsr_state_dict_to_save_format(state_dict):
    new_state_dict = {}
    to_basicsr_dict_values = list(to_basicsr_dict.values())
    for k, v in state_dict.items():
        if k in to_basicsr_dict_values:
            for key, value in to_basicsr_dict.items():
                if value == k:
                    new_state_dict[key] = v

        elif k.startswith('body.'):
            bsr_name = k.replace('body.', 'model.1.sub.').lower()
            bsr_name = bsr_name.replace('rdb', 'RDB')
            bsr_name = bsr_name.replace('.weight', '.0.weight')
            bsr_name = bsr_name.replace('.bias', '.0.bias')
            new_state_dict[bsr_name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
