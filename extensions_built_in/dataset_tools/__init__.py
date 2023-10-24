from toolkit.extension import Extension


class DatasetToolsExtension(Extension):
    uid = "dataset_tools"

    # name is the name of the extension for printing
    name = "Dataset Tools"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .DatasetTools import DatasetTools
        return DatasetTools


class SyncFromCollectionExtension(Extension):
    uid = "sync_from_collection"
    name = "Sync from Collection"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .SyncFromCollection import SyncFromCollection
        return SyncFromCollection
    
    
class SuperTaggerExtension(Extension):
    uid = "super_tagger"
    name = "Super Tagger"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .SuperTagger import SuperTagger
        return SuperTagger


AI_TOOLKIT_EXTENSIONS = [
    SyncFromCollectionExtension, DatasetToolsExtension, SuperTaggerExtension
]
