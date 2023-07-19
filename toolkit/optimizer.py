import torch


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6
):
    if optimizer_type == 'dadaptation':
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        optimizer = dadaptation.DAdaptAdam(params, lr=1.0)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(learning_rate))
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    return optimizer
