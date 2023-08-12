

def value_map(inputs, min_in, max_in, min_out, max_out):
    return (inputs - min_in) * (max_out - min_out) / (max_in - min_in) + min_out
