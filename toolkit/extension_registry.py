from typing import Dict, Type, Optional, List
import os
import importlib
import pkgutil
from toolkit.paths import TOOLKIT_ROOT
from toolkit.extension import Extension


class ExtensionRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._extensions: Dict[str, Extension] = {}
            cls._instance._loaded_modules: set = set()
            # Load extensions from directories on first instantiation
            cls._instance._load_directory_extensions()
        return cls._instance

    def _load_directory_extensions(self) -> None:
        """Load extensions from the standard extension directories"""
        extension_folders = ["extensions", "extensions_built_in"]

        for sub_dir in extension_folders:
            extensions_dir = os.path.join(TOOLKIT_ROOT, sub_dir)
            if not os.path.exists(extensions_dir):
                continue

            for _, name, _ in pkgutil.iter_modules([extensions_dir]):
                try:
                    # Skip if we've already loaded this module
                    module_path = f"{sub_dir}.{name}"
                    if module_path in self._loaded_modules:
                        continue

                    # Import the module
                    module = importlib.import_module(module_path)
                    self._loaded_modules.add(module_path)

                    # Get the value of the AI_TOOLKIT_EXTENSIONS variable
                    extensions = getattr(module, "AI_TOOLKIT_EXTENSIONS", None)
                    if isinstance(extensions, list):
                        # Register each extension
                        for ext in extensions:
                            self.register(ext, allow_override=False)
                except ImportError as e:
                    print(f"Failed to import the {name} module. Error: {str(e)}")

    def register(self, extension: Type[Extension], allow_override: bool = True) -> None:
        """Register an extension at runtime"""
        if not issubclass(extension, Extension):
            raise ValueError(
                f"Extension must be a subclass of Extension, got {extension}"
            )

        if extension.uid in self._extensions and not allow_override:
            raise ValueError(f"Extension with uid {extension.uid} already registered")

        self._extensions[extension.uid] = extension

    def register_process(
        self,
        uid: str,
        process_class: Type,
        name: str = None,
        allow_override: bool = True,
    ) -> None:
        """Register a process directly without creating an Extension class"""

        # Create an anonymous Extension class
        class DynamicExtension(Extension):
            uid = uid
            name = name or uid

            @classmethod
            def get_process(cls):
                return process_class

        self.register(DynamicExtension, allow_override=allow_override)

    def reload_directory_extensions(self) -> None:
        """Force a reload of all directory-based extensions"""
        self._loaded_modules.clear()
        self._load_directory_extensions()

    def unregister(self, uid: str) -> None:
        """Unregister an extension"""
        if uid in self._extensions:
            del self._extensions[uid]

    def get_extension(self, uid: str) -> Optional[Extension]:
        """Get an extension by uid"""
        return self._extensions.get(uid)

    def get_all_extensions(self) -> Dict[str, Extension]:
        """Get all registered extensions"""
        return self._extensions.copy()

    def get_process_dict(self) -> Dict[str, Type]:
        """Get a dictionary of extension uid -> process class"""
        return {uid: ext.get_process() for uid, ext in self._extensions.items()}
