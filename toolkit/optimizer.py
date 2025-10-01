import torch
import importlib

def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None
):
    if optimizer_params is None:
        optimizer_params = {}
    lower_type = optimizer_type.lower()
    if lower_type.startswith("dadaptation"):
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0
        if lower_type.endswith('lion'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type.endswith('adam'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type == 'dadaptation':
            # backwards compatibility
            optimizer = dadaptation.DAdaptAdam(params, eps=1e-6, lr=use_lr, **optimizer_params)
            # warn user that dadaptation is deprecated
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy8bit(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type == "prodigy":
        from prodigyopt import Prodigy

        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
    elif lower_type == "adam8bit":
        import bitsandbytes
        return bitsandbytes.optim.Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "ademamix8bit":
        import bitsandbytes
        return bitsandbytes.optim.AdEMAMix8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "adamw8bit":
        import bitsandbytes
        return bitsandbytes.optim.AdamW8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "lion8bit":
        import bitsandbytes
        return bitsandbytes.optim.Lion8bit(params, lr=learning_rate, **optimizer_params)
    elif lower_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=learning_rate, **optimizer_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        optimizer = torch.optim.Adagrad(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        if 'relative_step' not in optimizer_params:
            optimizer_params['relative_step'] = False
        if 'scale_parameter' not in optimizer_params:
            optimizer_params['scale_parameter'] = False
        if 'warmup_init' not in optimizer_params:
            optimizer_params['warmup_init'] = False
        optimizer = Adafactor(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        optimizer = Automagic(params, lr=float(learning_rate), **optimizer_params)
    else:
        # Try to dynamically import a user-defined optimizer
        try:
            # Split the string into module path and class name 
            parts = optimizer_type.split(".")
            if len(parts) < 2:
                raise ValueError(f"Unknown optimizer type {optimizer_type}")

            module_path = ".".join(parts[:-1])
            class_name = parts[-1]

            # Import module dynamically
            mod = importlib.import_module(module_path)

            # Get optimizer class from module
            opt_class = getattr(mod, class_name)

            # Instantiate optimizer
            try:
                optimizer = opt_class(params, lr=float(learning_rate), **optimizer_params)
            except TypeError:
                # In case the optimizer does not take lr or eps the same way
                optimizer = opt_class(params, **optimizer_params)

            print(f"Using user-defined optimizer: {optimizer_type}")
            return optimizer

        except Exception as e:
            raise ValueError(f'Unknown optimizer type.  Make sure your optimizer is installed in the virtual environment (venv).  {optimizer_type}. '
                             f'Failed to import dynamically. Error: {e}')
    return optimizer
