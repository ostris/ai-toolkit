from typing import Type, List, Union

BucketResolution = Type[{"width": int, "height": int}]

# resolutions SDXL was trained on with a 1024x1024 base resolution
resolutions_1024: List[BucketResolution] = [
    # SDXL Base resolution
    {"width": 1024, "height": 1024},
    # SDXL Resolutions, widescreen
    {"width": 2048, "height": 512},
    {"width": 1984, "height": 512},
    {"width": 1920, "height": 512},
    {"width": 1856, "height": 512},
    {"width": 1792, "height": 576},
    {"width": 1728, "height": 576},
    {"width": 1664, "height": 576},
    {"width": 1600, "height": 640},
    {"width": 1536, "height": 640},
    {"width": 1472, "height": 704},
    {"width": 1408, "height": 704},
    {"width": 1344, "height": 704},
    {"width": 1344, "height": 768},
    {"width": 1280, "height": 768},
    {"width": 1216, "height": 832},
    {"width": 1152, "height": 832},
    {"width": 1152, "height": 896},
    {"width": 1088, "height": 896},
    {"width": 1088, "height": 960},
    {"width": 1024, "height": 960},
    # SDXL Resolutions, portrait
    {"width": 960, "height": 1024},
    {"width": 960, "height": 1088},
    {"width": 896, "height": 1088},
    {"width": 896, "height": 1152},
    {"width": 832, "height": 1152},
    {"width": 832, "height": 1216},
    {"width": 768, "height": 1280},
    {"width": 768, "height": 1344},
    {"width": 704, "height": 1408},
    {"width": 704, "height": 1472},
    {"width": 640, "height": 1536},
    {"width": 640, "height": 1600},
    {"width": 576, "height": 1664},
    {"width": 576, "height": 1728},
    {"width": 576, "height": 1792},
    {"width": 512, "height": 1856},
    {"width": 512, "height": 1920},
    {"width": 512, "height": 1984},
    {"width": 512, "height": 2048},
]


def get_bucket_sizes(resolution: int = 512, divisibility: int = 8) -> List[BucketResolution]:
    # determine scaler form 1024 to resolution
    scaler = resolution / 1024

    bucket_size_list = []
    for bucket in resolutions_1024:
        # must be divisible by 8
        width = int(bucket["width"] * scaler)
        height = int(bucket["height"] * scaler)
        if width % divisibility != 0:
            width = width - (width % divisibility)
        if height % divisibility != 0:
            height = height - (height % divisibility)
        bucket_size_list.append({"width": width, "height": height})

    return bucket_size_list


def get_bucket_for_image_size(
        width: int,
        height: int,
        bucket_size_list: List[BucketResolution] = None,
        resolution: Union[int, None] = None
) -> BucketResolution:
    if bucket_size_list is None and resolution is None:
        raise ValueError("Must provide either bucket_size_list or resolution")
    if bucket_size_list is None:
        bucket_size_list = get_bucket_sizes(resolution=resolution)

    # Check for exact match first
    for bucket in bucket_size_list:
        if bucket["width"] == width and bucket["height"] == height:
            return bucket

    # If exact match not found, find the closest bucket
    closest_bucket = None
    min_removed_pixels = float("inf")

    for bucket in bucket_size_list:
        scale_w = bucket["width"] / width
        scale_h = bucket["height"] / height

        # To minimize pixels, we use the larger scale factor to minimize the amount that has to be cropped.
        scale = max(scale_w, scale_h)

        new_width = int(width * scale)
        new_height = int(height * scale)

        removed_pixels = (new_width - bucket["width"]) * new_height + (new_height - bucket["height"]) * new_width

        if removed_pixels < min_removed_pixels:
            min_removed_pixels = removed_pixels
            closest_bucket = bucket

    if closest_bucket is None:
        raise ValueError("No suitable bucket found")

    return closest_bucket
