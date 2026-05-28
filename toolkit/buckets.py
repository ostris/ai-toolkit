from typing import TypedDict


class BucketResolution(TypedDict):
    width: int
    height: int


def get_resolution(width, height):
    num_pixels = width * height
    # determine same number of pixels for square image
    square_resolution = int(num_pixels**0.5)
    return square_resolution


def get_bucket_for_image_size(
    width: int, height: int, resolution: int = 512, divisibility: int = 8
) -> BucketResolution:
    total_pixels = width * height
    max_pixels = resolution * resolution

    target_pixels = min(total_pixels, max_pixels)

    scaler = target_pixels / total_pixels
    new_width = int((width * scaler) // divisibility * divisibility)
    new_height = int((height * scaler) // divisibility * divisibility)

    return {"width": new_width, "height": new_height}
