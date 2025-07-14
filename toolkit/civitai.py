from toolkit.paths import MODELS_PATH
import requests
import os
import json
import tqdm


class ModelCache:
    def __init__(self):
        self.raw_cache = {}
        self.cache_path = os.path.join(MODELS_PATH, '.ai_toolkit_cache.json')
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                all_cache = json.load(f)
                if 'models' in all_cache:
                    self.raw_cache = all_cache['models']
                else:
                    self.raw_cache = all_cache

    def get_model_path(self, model_id: int, model_version_id: int = None):
        if str(model_id) not in self.raw_cache:
            return None
        if model_version_id is None:
            # get latest version
            model_version_id = max([int(x) for x in self.raw_cache[str(model_id)].keys()])
            if model_version_id is None:
                return None
            model_path = self.raw_cache[str(model_id)][str(model_version_id)]['model_path']
            # check if model path exists
            if not os.path.exists(model_path):
                # remove version from cache
                del self.raw_cache[str(model_id)][str(model_version_id)]
                self.save()
                return None
            return model_path
        else:
            if str(model_version_id) not in self.raw_cache[str(model_id)]:
                return None
            model_path = self.raw_cache[str(model_id)][str(model_version_id)]['model_path']
            # check if model path exists
            if not os.path.exists(model_path):
                # remove version from cache
                del self.raw_cache[str(model_id)][str(model_version_id)]
                self.save()
                return None
            return model_path

    def update_cache(self, model_id: int, model_version_id: int, model_path: str):
        if str(model_id) not in self.raw_cache:
            self.raw_cache[str(model_id)] = {}
        if str(model_version_id) not in self.raw_cache[str(model_id)]:
            self.raw_cache[str(model_id)][str(model_version_id)] = {}
        self.raw_cache[str(model_id)][str(model_version_id)] = {
            'model_path': model_path
        }
        self.save()

    def save(self):
        if not os.path.exists(os.path.dirname(self.cache_path)):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        all_cache = {'models': {}}
        if os.path.exists(self.cache_path):
            # load it first
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                all_cache = json.load(f)

        all_cache['models'] = self.raw_cache

        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(all_cache, f, indent=2, ensure_ascii=False)


def get_model_download_info(model_id: int, model_version_id: int = None):
    # curl https://civitai.com/api/v1/models?limit=3&types=TextualInversion \
    # -H "Content-Type: application/json" \
    # -X GET
    print(
        f"Getting model info for model id: {model_id}{f' and version id: {model_version_id}' if model_version_id is not None else ''}")
    endpoint = f"https://civitai.com/api/v1/models/{model_id}"

    # get the json
    response = requests.get(endpoint)
    response.raise_for_status()
    model_data = response.json()

    model_version = None

    # go through versions and get the top one if one is not set
    for version in model_data['modelVersions']:
        if model_version_id is not None:
            if str(version['id']) == str(model_version_id):
                model_version = version
                break
        else:
            # get first version
            model_version = version
            break

    if model_version is None:
        raise ValueError(
            f"Could not find a model version for model id: {model_id}{f' and version id: {model_version_id}' if model_version_id is not None else ''}")

    model_file = None
    # go through files and prefer fp16 safetensors
    # "metadata": {
    #   "fp": "fp16",
    #   "size": "pruned",
    #   "format": "SafeTensor"
    # },
    # todo check pickle scans and skip if not good
    # try to get fp16 safetensor
    for file in model_version['files']:
        if file['metadata']['fp'] == 'fp16' and file['metadata']['format'] == 'SafeTensor':
            model_file = file
            break

    if model_file is None:
        # try to get primary
        for file in model_version['files']:
            if file['primary']:
                model_file = file
                break

    if model_file is None:
        # try to get any safetensor
        for file in model_version['files']:
            if file['metadata']['format'] == 'SafeTensor':
                model_file = file
                break

    if model_file is None:
        # try to get any fp16
        for file in model_version['files']:
            if file['metadata']['fp'] == 'fp16':
                model_file = file
                break

    if model_file is None:
        # try to get any
        for file in model_version['files']:
            model_file = file
            break

    if model_file is None:
        raise ValueError(f"Could not find a model file to download for model id: {model_id}")

    return model_file, model_version['id']


def get_model_path_from_url(url: str):
    # get query params form url if they are set
    # https: // civitai.com / models / 25694?modelVersionId = 127742
    query_params = {}
    if '?' in url:
        query_string = url.split('?')[1]
        query_params = dict(qc.split("=") for qc in query_string.split("&"))

    # get model id from url
    model_id = url.split('/')[-1]
    # remove query params from model id
    if '?' in model_id:
        model_id = model_id.split('?')[0]
    if model_id.isdigit():
        model_id = int(model_id)
    else:
        raise ValueError(f"Invalid model id: {model_id}")

    model_cache = ModelCache()
    model_path = model_cache.get_model_path(model_id, query_params.get('modelVersionId', None))
    if model_path is not None:
        return model_path
    else:
        # download model
        file_info, model_version_id = get_model_download_info(model_id, query_params.get('modelVersionId', None))

        download_url = file_info['downloadUrl']  # url does not work directly
        size_kb = file_info['sizeKB']
        filename = file_info['name']
        model_path = os.path.join(MODELS_PATH, filename)

        # download model
        print(f"Did not find model locally, downloading from model from: {download_url}")

        # use tqdm to show status of downlod
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        tmp_path = os.path.join(MODELS_PATH, f".download_tmp_{filename}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # remove tmp file if it exists
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        try:

            with open(tmp_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
                progress_bar.close()
                # move to final path
                os.rename(tmp_path, model_path)
                model_cache.update_cache(model_id, model_version_id, model_path)

                return model_path
        except Exception as e:
            # remove tmp file
            os.remove(tmp_path)
            raise e


# if is main
if __name__ == '__main__':
    model_path = get_model_path_from_url("https://civitai.com/models/25694?modelVersionId=127742")
    print(model_path)
