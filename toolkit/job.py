from jobs import BaseJob
from toolkit.config import get_config


def get_job(config_path) -> BaseJob:
    config = get_config(config_path)
    if not config['job']:
        raise ValueError('config file is invalid. Missing "job" key')

    job = config['job']
    if job == 'extract':
        from jobs import ExtractJob
        return ExtractJob(config)
    elif job == 'train':
        from jobs import TrainJob
        return TrainJob(config)
    else:
        raise ValueError(f'Unknown job type {job}')
