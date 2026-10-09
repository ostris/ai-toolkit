"""
PixlStash API client — corrected against live API (v1.0.0b3).

Key endpoint notes (verified from /redoc):
  - Base prefix:     /api/v1
  - Picture sets:    /api/v1/picture_sets/{id}  (underscore, not hyphen)
  - Picture file:    /api/v1/pictures/{id}.{ext}
  - Thumbnail:       /api/v1/pictures/thumbnails/{id}.webp
  - Set members:     /api/v1/picture_sets/{id}/members  -> returns integer IDs only
  - Char pictures:   /api/v1/pictures/list?character_id={id}
  - Auth:            Authorization: Bearer <token> header on all requests
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import requests
import urllib3

_EMPTY_TAG_SENTINEL = ""


class PixlStashError(RuntimeError):
    """Raised when the PixlStash server returns an unexpected response."""


class PixlStashClient:
    """Thin wrapper around the PixlStash REST API."""

    def __init__(self, base_url: str, token: str, verify_ssl: bool = True) -> None:
        self.base_url = base_url.rstrip("/") + "/api/v1"
        self.token = token
        self._session = requests.Session()
        self._session.verify = verify_ssl
        self._session.headers.update({"Authorization": f"Bearer {token}"})
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def login(self) -> None:
        """No-op: authentication is handled via Bearer token header."""
        pass

    def _get(self, path: str, **params) -> requests.Response:
        r = self._session.get(
            f"{self.base_url}{path}", params=params or None, timeout=60
        )
        if not r.ok:
            raise PixlStashError(f"GET {path} failed ({r.status_code}): {r.text[:200]}")
        return r

    # ------------------------------------------------------------------
    # Characters
    # ------------------------------------------------------------------

    def get_character(self, character_id: int) -> dict:
        return self._get(f"/characters/{character_id}").json()

    def list_characters(self, name: Optional[str] = None) -> List[dict]:
        params = {}
        if name:
            params["name"] = name
        return self._get("/characters", **params).json()

    # ------------------------------------------------------------------
    # Picture sets
    # ------------------------------------------------------------------

    def get_picture_set(self, set_id: int) -> dict:
        """Return picture set metadata."""
        return self._get(f"/picture_sets/{set_id}").json()["set"]

    def list_picture_sets(self) -> List[dict]:
        """Return all non-reference picture sets."""
        all_sets = self._get("/picture_sets").json()
        # Reference sets are auto-created per character for face recognition;
        # they have a non-null `reference_character` field and are not useful
        # as training datasets.
        return [s for s in all_sets if s.get("reference_character") is None]

    # ------------------------------------------------------------------
    # Picture listing
    # ------------------------------------------------------------------

    def list_pictures_for_character(self, character_id: int) -> List[dict]:
        """Return the listing rows for every picture this character appears in.

        Rows carry only scalar grid fields (id, score, format) — enough to
        filter and download.  The natural-language ``description`` and WD14
        ``tags`` are projected out of the listing response, so they are read
        per-picture via :meth:`get_picture_metadata`.
        """
        return self._get("/pictures", character_id=character_id).json()

    def list_pictures_for_set(self, set_id: int) -> List[dict]:
        """Return the picture rows for every member of a set.

        Uses a single GET /picture_sets/{id} call which embeds all members.
        As with the character listing, ``description`` and ``tags`` must be
        fetched per-picture via :meth:`get_picture_metadata`.
        """
        return self._get(f"/picture_sets/{set_id}").json()["pictures"]

    def get_picture_metadata(self, pic_id: int) -> dict:
        """Return full metadata for one picture, including description and tags.

        The listing endpoints only return scalar grid fields — the
        natural-language caption and the WD14 tags are not included — so both
        are read from the per-picture metadata endpoint::

            GET /pictures/{id}/metadata

        ``description`` comes back as a plain string and ``tags`` as a list of
        ``{"id": int, "tag": str}`` objects, exactly what :meth:`build_caption`
        consumes.
        """
        return self._get(f"/pictures/{pic_id}/metadata").json()

    def download_image_bytes(self, pic_id: int, fmt: str = "jpg") -> bytes:
        """Return raw image bytes. Endpoint: GET /pictures/{id}.{ext}"""
        r = self._session.get(
            f"{self.base_url}/pictures/{pic_id}.{fmt}",
            timeout=120,
        )
        if not r.ok:
            raise PixlStashError(
                f"Image download for id={pic_id} failed ({r.status_code})"
            )
        return r.content

    # ------------------------------------------------------------------
    # Thumbnail URLs (used by the UI browse modal)
    # ------------------------------------------------------------------

    def thumbnail_url(self, pic_id: int) -> str:
        """Full URL for a picture's WebP thumbnail."""
        return f"{self.base_url}/pictures/thumbnails/{pic_id}.webp"

    def picture_set_thumbnail_url(self, set_id: int) -> str:
        return f"{self.base_url}/picture_sets/{set_id}/thumbnail"

    def character_thumbnail_url(self, character_id: int) -> str:
        return f"{self.base_url}/characters/{character_id}/thumbnail"

    # ------------------------------------------------------------------
    # Caption building
    # ------------------------------------------------------------------

    @staticmethod
    def tags_to_string(meta: dict) -> str:
        return ", ".join(
            t["tag"]
            for t in (meta.get("tags") or [])
            if t.get("tag") and t["tag"] != _EMPTY_TAG_SENTINEL
        )

    @classmethod
    def build_caption(
        cls,
        meta: dict,
        mode: str = "description",
        trigger: str = "",
    ) -> str:
        """
        Build a caption string for one picture.

        mode: "description" | "tags" | "both"
        trigger: optional token prepended to every caption.
        """
        parts: List[str] = []

        if trigger:
            parts.append(trigger)

        if mode in ("description", "both"):
            desc = (meta.get("description") or "").strip()
            if desc:
                parts.append(desc)

        if mode in ("tags", "both"):
            tag_str = cls.tags_to_string(meta)
            if tag_str:
                parts.append(tag_str)

        content_count = len(parts) - (1 if trigger else 0)
        if content_count == 0:
            fallback = (meta.get("description") or "").strip() or cls.tags_to_string(
                meta
            )
            if fallback:
                parts.append(fallback)

        return ", ".join(parts)
