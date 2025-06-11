import os
import requests
import tqdm
from typing import List, Optional, TYPE_CHECKING


def img_root_path(img_id: str):
    return os.path.dirname(os.path.dirname(img_id))


if TYPE_CHECKING:
    from .dataset_tools_config_modules import DatasetSyncCollectionConfig

img_exts = ['.jpg', '.jpeg', '.png', '.webp', '.avif']

class Photo:
    def __init__(
            self,
            id,
            host,
            width,
            height,
            url,
            filename
    ):
        self.id = str(id)
        self.host = host
        self.width = width
        self.height = height
        self.url = url
        self.filename = filename


def get_desired_size(img_width: int, img_height: int, min_width: int, min_height: int):
    if img_width > img_height:
        scale = min_height / img_height
    else:
        scale = min_width / img_width

    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    return new_width, new_height


def get_pexels_images(config: 'DatasetSyncCollectionConfig') -> List[Photo]:
    all_images = []
    next_page = f"https://api.pexels.com/v1/collections/{config.collection_id}?page=1&per_page=80&type=photos"

    while True:
        response = requests.get(next_page, headers={
            "Authorization": f"{config.api_key}"
        })
        response.raise_for_status()
        data = response.json()
        all_images.extend(data['media'])
        if 'next_page' in data and data['next_page']:
            next_page = data['next_page']
        else:
            break

    photos = []
    for image in all_images:
        new_width, new_height = get_desired_size(image['width'], image['height'], config.min_width, config.min_height)
        url = f"{image['src']['original']}?auto=compress&cs=tinysrgb&h={new_height}&w={new_width}"
        filename = os.path.basename(image['src']['original'])

        photos.append(Photo(
            id=image['id'],
            host="pexels",
            width=image['width'],
            height=image['height'],
            url=url,
            filename=filename
        ))

    return photos


def get_unsplash_images(config: 'DatasetSyncCollectionConfig') -> List[Photo]:
    headers = {
        # "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
        "Authorization": f"Client-ID {config.api_key}"
    }
    # headers['Authorization'] = f"Bearer {token}"

    url = f"https://api.unsplash.com/collections/{config.collection_id}/photos?page=1&per_page=30"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    res_headers = response.headers
    # parse the link header to get the next page
    # 'Link': '<https://api.unsplash.com/collections/mIPWwLdfct8/photos?page=82>; rel="last", <https://api.unsplash.com/collections/mIPWwLdfct8/photos?page=2>; rel="next"'
    has_next_page = False
    if 'Link' in res_headers:
        has_next_page = True
        link_header = res_headers['Link']
        link_header = link_header.split(',')
        link_header = [link.strip() for link in link_header]
        link_header = [link.split(';') for link in link_header]
        link_header = [[link[0].strip('<>'), link[1].strip().strip('"')] for link in link_header]
        link_header = {link[1]: link[0] for link in link_header}

        # get page number from last url
        last_page = link_header['rel="last']
        last_page = last_page.split('?')[1]
        last_page = last_page.split('&')
        last_page = [param.split('=') for param in last_page]
        last_page = {param[0]: param[1] for param in last_page}
        last_page = int(last_page['page'])

    all_images = response.json()

    if has_next_page:
        # assume we start on page 1, so we don't need to get it again
        for page in tqdm.tqdm(range(2, last_page + 1)):
            url = f"https://api.unsplash.com/collections/{config.collection_id}/photos?page={page}&per_page=30"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            all_images.extend(response.json())

    photos = []
    for image in all_images:
        new_width, new_height = get_desired_size(image['width'], image['height'], config.min_width, config.min_height)
        url = f"{image['urls']['raw']}&w={new_width}"
        filename = f"{image['id']}.jpg"

        photos.append(Photo(
            id=image['id'],
            host="unsplash",
            width=image['width'],
            height=image['height'],
            url=url,
            filename=filename
        ))

    return photos


def get_img_paths(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)
    local_files = os.listdir(dir_path)
    # remove non image files
    local_files = [file for file in local_files if os.path.splitext(file)[1].lower() in img_exts]
    # make full path
    local_files = [os.path.join(dir_path, file) for file in local_files]
    return local_files


def get_local_image_ids(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)
    local_files = get_img_paths(dir_path)
    # assuming local files are named after Unsplash IDs, e.g., 'abc123.jpg'
    return set([os.path.basename(file).split('.')[0] for file in local_files])


def get_local_image_file_names(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)
    local_files = get_img_paths(dir_path)
    # assuming local files are named after Unsplash IDs, e.g., 'abc123.jpg'
    return set([os.path.basename(file) for file in local_files])


def download_image(photo: Photo, dir_path: str, min_width: int = 1024, min_height: int = 1024):
    img_width = photo.width
    img_height = photo.height

    if img_width < min_width or img_height < min_height:
        raise ValueError(f"Skipping {photo.id} because it is too small: {img_width}x{img_height}")

    img_response = requests.get(photo.url)
    img_response.raise_for_status()
    os.makedirs(dir_path, exist_ok=True)

    filename = os.path.join(dir_path, photo.filename)
    with open(filename, 'wb') as file:
        file.write(img_response.content)


def update_caption(img_path: str):
    # if the caption is a txt file, convert it to a json file
    filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
    # see if it exists
    if os.path.exists(os.path.join(os.path.dirname(img_path), f"{filename_no_ext}.json")):
        # todo add poi and what not
        return  # we have a json file
    caption = ""
    # see if txt file exists
    if os.path.exists(os.path.join(os.path.dirname(img_path), f"{filename_no_ext}.txt")):
        # read it
        with open(os.path.join(os.path.dirname(img_path), f"{filename_no_ext}.txt"), 'r', encoding='utf-8') as file:
            caption = file.read()
    # write json file
    with open(os.path.join(os.path.dirname(img_path), f"{filename_no_ext}.json"), 'w', encoding='utf-8') as file:
        file.write(f'{{"caption": "{caption}"}}')

    # delete txt file
    os.remove(os.path.join(os.path.dirname(img_path), f"{filename_no_ext}.txt"))


# def equalize_img(img_path: str):
#     input_path = img_path
#     output_path = os.path.join(img_root_path(img_path), COLOR_CORRECTED_DIR, os.path.basename(img_path))
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     process_img(
#         img_path=input_path,
#         output_path=output_path,
#         equalize=True,
#         max_size=2056,
#         white_balance=False,
#         gamma_correction=False,
#         strength=0.6,
#     )


# def annotate_depth(img_path: str):
#     # make fake args
#     args = argparse.Namespace()
#     args.annotator = "midas"
#     args.res = 1024
#
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     output = annotate(img, args)
#
#     output = output.astype('uint8')
#     output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
#
#     os.makedirs(os.path.dirname(img_path), exist_ok=True)
#     output_path = os.path.join(img_root_path(img_path), DEPTH_DIR, os.path.basename(img_path))
#
#     cv2.imwrite(output_path, output)


# def invert_depth(img_path: str):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # invert the colors
#     img = cv2.bitwise_not(img)
#
#     os.makedirs(os.path.dirname(img_path), exist_ok=True)
#     output_path = os.path.join(img_root_path(img_path), INVERTED_DEPTH_DIR, os.path.basename(img_path))
#     cv2.imwrite(output_path, img)


    #
    # # update our list of raw images
    # raw_images = get_img_paths(raw_dir)
    #
    # # update raw captions
    # for image_id in tqdm.tqdm(raw_images, desc="Updating raw captions"):
    #     update_caption(image_id)
    #
    # # equalize images
    # for img_path in tqdm.tqdm(raw_images, desc="Equalizing images"):
    #     if img_path not in eq_images:
    #         equalize_img(img_path)
    #
    # # update our list of eq images
    # eq_images = get_img_paths(eq_dir)
    # # update eq captions
    # for image_id in tqdm.tqdm(eq_images, desc="Updating eq captions"):
    #     update_caption(image_id)
    #
    # # annotate depth
    # depth_dir = os.path.join(root_dir, DEPTH_DIR)
    # depth_images = get_img_paths(depth_dir)
    # for img_path in tqdm.tqdm(eq_images, desc="Annotating depth"):
    #     if img_path not in depth_images:
    #         annotate_depth(img_path)
    #
    # depth_images = get_img_paths(depth_dir)
    #
    # # invert depth
    # inv_depth_dir = os.path.join(root_dir, INVERTED_DEPTH_DIR)
    # inv_depth_images = get_img_paths(inv_depth_dir)
    # for img_path in tqdm.tqdm(depth_images, desc="Inverting depth"):
    #     if img_path not in inv_depth_images:
    #         invert_depth(img_path)
