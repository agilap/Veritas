import PIL.Image


def analyze_image(caption: str, image: PIL.Image.Image) -> dict:
    return {
        "layer": 2,
        "error": None,
        "caption": caption,
        "image_size": image.size,
    }


def blip2_verify(image: PIL.Image.Image) -> dict:
    return {
        "layer": 6,
        "error": None,
        "image_size": image.size,
    }