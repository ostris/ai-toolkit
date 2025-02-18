export interface Model {
    name_or_path: string;
    model_kwargs?: Record<string, boolean>;
    train_kwargs?: Record<string, boolean>;
}

export interface Option {
    model: Model[];
}


export const options = {
    model: [
        {
            name_or_path: "ostris/Flex.1-alpha",
            model_kwargs: {
                "is_flux": true
            },
            train_kwargs: {
                "bypass_guidance_embedding": true
            } 
        },
        {
            name_or_path: "black-forest-labs/FLUX.1-dev",
            model_kwargs: {
                "is_flux": true
            },
        },
        {
            name_or_path: "Alpha-VLLM/Lumina-Image-2.0",
            model_kwargs: {
                "is_lumina2": true
            },
        },
    ]

} as Option;