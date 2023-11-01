
caption_manipulation_steps = ['caption', 'caption_short']

default_long_prompt = 'caption this image. describe every single thing in the image in detail. Do not include any unnecessary words in your description for the sake of good grammar. I want many short statements that serve the single purpose of giving the most thorough description if items as possible in the smallest, comma separated way possible. be sure to describe people\'s moods, clothing, the environment, lighting, colors, and everything.'
default_short_prompt = 'caption this image in less than ten words'

default_replacements = [
    ("the image features", ""),
    ("the image shows", ""),
    ("the image depicts", ""),
    ("the image is", ""),
    ("in this image", ""),
    ("in the image", ""),
]


def clean_caption(cap, replacements=None):
    if replacements is None:
        replacements = default_replacements

    # remove any newlines
    cap = cap.replace("\n", ", ")
    cap = cap.replace("\r", ", ")
    cap = cap.replace(".", ",")
    cap = cap.replace("\"", "")

    # remove unicode characters
    cap = cap.encode('ascii', 'ignore').decode('ascii')

    # make lowercase
    cap = cap.lower()
    # remove any extra spaces
    cap = " ".join(cap.split())

    for replacement in replacements:
        if replacement[0].startswith('*'):
            # we are removing all text if it starts with this and the rest matches
            search_text = replacement[0][1:]
            if cap.startswith(search_text):
                cap = ""
        else:
            cap = cap.replace(replacement[0].lower(), replacement[1].lower())

    cap_list = cap.split(",")
    # trim whitespace
    cap_list = [c.strip() for c in cap_list]
    # remove empty strings
    cap_list = [c for c in cap_list if c != ""]
    # remove duplicates
    cap_list = list(dict.fromkeys(cap_list))
    # join back together
    cap = ", ".join(cap_list)
    return cap