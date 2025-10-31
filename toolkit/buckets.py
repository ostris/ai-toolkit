from typing import Type, List, Union, TypedDict


class BucketResolution(TypedDict):
    width: int
    height: int


# Video-friendly resolutions with common aspect ratios
# Base resolution: 1024×1024
# Keep only PRIMARY buckets to avoid videos being assigned to undersized buckets
resolutions_video_1024: List[BucketResolution] = [
    # Square
    {"width": 1024, "height": 1024},  # 1:1

    # 16:9 landscape (1.778 aspect - YouTube, TV standard)
    {"width": 1024, "height": 576},

    # 9:16 portrait (0.562 aspect - TikTok, Instagram Reels)
    {"width": 576, "height": 1024},

    # 4:3 landscape (1.333 aspect - older content)
    {"width": 1024, "height": 768},

    # 3:4 portrait (0.75 aspect)
    {"width": 768, "height": 1024},

    # Slightly wider/taller variants for flexibility
    {"width": 1024, "height": 640},  # 1.6 aspect
    {"width": 640, "height": 1024},  # 0.625 aspect
]

# SDXL resolutions (kept for backwards compatibility)
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
    {"width": 896, "height": 1152},  # 2:3
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
    # extra wides
    {"width": 8192, "height": 128},
    {"width": 128, "height": 8192},
]

def get_bucket_sizes(resolution: int = 512, divisibility: int = 8, use_video_buckets: bool = True, max_pixels_per_frame: int = None) -> List[BucketResolution]:
    # Use video-friendly buckets by default for better aspect ratio preservation
    base_resolutions = resolutions_video_1024 if use_video_buckets else resolutions_1024

    # If max_pixels_per_frame is specified, use pixel budget scaling
    # This maximizes resolution for each aspect ratio while keeping memory usage consistent
    if max_pixels_per_frame is not None:
        bucket_size_list = []
        for bucket in base_resolutions:
            # Calculate aspect ratio
            base_aspect = bucket["width"] / bucket["height"]

            # Calculate optimal dimensions for this aspect ratio within pixel budget
            # For aspect ratio a = w/h and pixel budget p = w*h:
            # w = sqrt(p * a), h = sqrt(p / a)
            optimal_width = (max_pixels_per_frame * base_aspect) ** 0.5
            optimal_height = (max_pixels_per_frame / base_aspect) ** 0.5

            # Round down to divisibility
            width = int(optimal_width)
            height = int(optimal_height)
            width = width - (width % divisibility)
            height = height - (height % divisibility)

            # Verify we're under budget (should always be true with round-down)
            actual_pixels = width * height
            if actual_pixels > max_pixels_per_frame:
                # Safety check - scale down if somehow over budget
                scale = (max_pixels_per_frame / actual_pixels) ** 0.5
                width = int(width * scale)
                height = int(height * scale)
                width = width - (width % divisibility)
                height = height - (height % divisibility)

            bucket_size_list.append({"width": width, "height": height})

        return bucket_size_list

    # Original scaling logic (for backwards compatibility)
    scaler = resolution / 1024
    bucket_size_list = []
    for bucket in base_resolutions:
        # must be divisible by 8
        width = int(bucket["width"] * scaler)
        height = int(bucket["height"] * scaler)
        if width % divisibility != 0:
            width = width - (width % divisibility)
        if height % divisibility != 0:
            height = height - (height % divisibility)

        # Filter buckets where any dimension exceeds the resolution parameter
        # This ensures memory usage stays within bounds for the target resolution
        if max(width, height) > resolution:
            continue

        bucket_size_list.append({"width": width, "height": height})

    return bucket_size_list


def get_resolution(width, height):
    num_pixels = width * height
    # determine same number of pixels for square image
    square_resolution = int(num_pixels ** 0.5)
    return square_resolution


def get_bucket_for_image_size(
        width: int,
        height: int,
        bucket_size_list: List[BucketResolution] = None,
        resolution: Union[int, None] = None,
        divisibility: int = 8,
        max_pixels_per_frame: int = None
) -> BucketResolution:

    if bucket_size_list is None and resolution is None:
        # get resolution from width and height
        resolution = get_resolution(width, height)
    if bucket_size_list is None:
        bucket_size_list = get_bucket_sizes(resolution=resolution, divisibility=divisibility, max_pixels_per_frame=max_pixels_per_frame)

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