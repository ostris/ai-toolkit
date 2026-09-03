from toolkit.extension import Extension
from toolkit.dataset_sources.registry import register_source


class PixlStashFetchExtension(Extension):
    uid = "pixlstash_fetch"
    name = "PixlStash Dataset Fetch"

    @classmethod
    def get_process(cls):
        from .PixlStashFetchProcess import PixlStashFetchProcess
        return PixlStashFetchProcess


# Register the dataset source so the dataloader and UI can discover it
def _register():
    from .pixlstash_source import PixlStashDatasetSource
    register_source(PixlStashDatasetSource)


_register()

AI_TOOLKIT_EXTENSIONS = [
    PixlStashFetchExtension,
]
