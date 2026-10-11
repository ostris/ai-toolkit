"""
Base class and supporting types for remote dataset sources (data-source plugins).

A RemoteDatasetSource plugin knows how to:
  1. Declare what settings it needs from the user (URL, token, etc.)
  2. Return grouped, browseable items available on the remote
  3. Declare source-specific import form fields (e.g. caption mode, score filter)

Download logic lives in each extension's FetchProcess (e.g. PixlStashFetchProcess),
triggered explicitly by the user via the UI — never automatically during training.

Implementations live in extensions, e.g. extensions/pixlstash/.
They register themselves via toolkit.dataset_sources.registry.register_source().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SettingField:
    """Describes one user-configurable setting required by a source."""

    # Key stored in the AI-Toolkit settings DB (e.g. "PIXLSTASH_URL")
    key: str
    # Human-readable label shown in the UI
    label: str
    # Input type hint for the UI: "text" | "password"
    input_type: str = "text"
    # Shown below the label in the UI
    description: str = ""
    # Placeholder text inside the input
    placeholder: str = ""
    # Whether this setting must be non-empty for the plugin to be considered configured
    required: bool = True


@dataclass
class SourceItem:
    """One item inside a SourceGroup (e.g. a single character or album)."""

    id: str                   # string to handle both ints and UUIDs
    name: str
    picture_count: int = -1   # -1 = unknown
    # ID and type passed to the thumbnail proxy route
    thumbnail_id: str = ""
    thumbnail_type: str = ""


@dataclass
class SourceGroup:
    """A labelled collection of SourceItems shown as one tab in the browse modal."""

    id: str           # e.g. "character", "picture_set", "person", "album"
    label: str        # e.g. "Characters", "Albums"
    items: List[SourceItem] = field(default_factory=list)


@dataclass
class ImportField:
    """Describes one source-specific field shown in the import form."""

    id: str
    label: str
    # "select" | "text" | "checkbox"
    field_type: str = "text"
    # For "select": list of {"value": ..., "label": ...} dicts
    options: List[dict] = field(default_factory=list)
    default: Any = None
    required: bool = False


class RemoteDatasetSource(ABC):
    """
    Abstract base class for remote dataset source plugins.

    Subclasses must set ``type_id`` to a unique string identifier.
    """

    # Unique plugin identifier, e.g. "pixlstash"
    type_id: str = None

    # Human-readable name shown in the UI, e.g. "PixlStash"
    display_name: str = ""

    # Optional absolute path to an icon image (PNG/SVG) shown in the UI
    icon_path: Optional[str] = None

    def __init__(self, settings: dict) -> None:
        """
        Parameters
        ----------
        settings:
            Key/value pairs loaded from the AI-Toolkit settings DB.
        """
        self.settings = settings

    # ------------------------------------------------------------------
    # Schema — settings this plugin needs the user to configure
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def get_settings_schema(cls) -> List[SettingField]:
        """Return the list of settings fields this source requires."""

    # ------------------------------------------------------------------
    # Thumbnail — serve a thumbnail image to the UI
    # ------------------------------------------------------------------

    @abstractmethod
    def get_thumbnail(self, thumbnail_id: str, thumbnail_type: str) -> tuple:
        """
        Fetch a thumbnail image for a SourceItem.

        Returns
        -------
        (image_bytes: bytes, content_type: str)
        """

    # ------------------------------------------------------------------
    # Job config — describe how to run the import job
    # ------------------------------------------------------------------

    @abstractmethod
    def build_job_config(self, params: dict) -> dict:
        """
        Build and return the process-config dict for run.py.

        ``params`` mirrors the POST body from the import route:
          source_type, source_id, trigger_word, dataset_name, overwrite,
          plus any source-specific fields from get_import_fields().

        The returned dict is placed inside:
          { job: 'extension', config: { name: '...', process: [<returned dict>] } }
        """

    # ------------------------------------------------------------------
    # Browse — return grouped items the user can pick from
    # ------------------------------------------------------------------

    @abstractmethod
    def browse(self) -> List[SourceGroup]:
        """
        Return a list of SourceGroups, each representing one browseable
        category (e.g. Characters, Albums).  Items within each group are
        displayed as thumbnails in the UI.
        """

    # ------------------------------------------------------------------
    # Import form — extra fields beyond trigger_word / dataset_name
    # ------------------------------------------------------------------

    @classmethod
    def get_import_fields(cls) -> List[ImportField]:
        """
        Return source-specific fields to show in the import form.
        The base implementation returns an empty list (no extra fields).
        Override in subclasses to add e.g. caption_mode or min_score.
        """
        return []

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        return self.settings.get(key, default)

    @classmethod
    def is_configured(cls, settings: dict) -> bool:
        """Return True if all required settings have non-empty values."""
        return all(
            settings.get(f.key, "").strip()
            for f in cls.get_settings_schema()
            if f.required
        )
