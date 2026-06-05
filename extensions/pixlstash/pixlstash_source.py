"""
PixlStash implementation of RemoteDatasetSource.

Registered under type_id = "pixlstash".
"""

from __future__ import annotations

import io
import os
from typing import List

from tqdm import tqdm

from toolkit.dataset_sources.base import (
    ImportField,
    RemoteDatasetSource,
    SettingField,
    SourceGroup,
    SourceItem,
)


class PixlStashDatasetSource(RemoteDatasetSource):
    type_id = "pixlstash"
    display_name = "PixlStash"
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

    # Settings keys stored in the AI-Toolkit DB
    SETTING_URL = "PIXLSTASH_URL"
    SETTING_TOKEN = "PIXLSTASH_TOKEN"

    SETTING_VERIFY_SSL = "PIXLSTASH_VERIFY_SSL"

    @classmethod
    def get_settings_schema(cls) -> List[SettingField]:
        return [
            SettingField(
                key=cls.SETTING_URL,
                label="PixlStash URL",
                input_type="text",
                description="Base URL of your PixlStash server.",
                placeholder="http://localhost:9537",
            ),
            SettingField(
                key=cls.SETTING_TOKEN,
                label="PixlStash API Token",
                input_type="password",
                description="Create a token in PixlStash → Settings → API Tokens.",
                placeholder="Paste your API token here",
            ),
            SettingField(
                key=cls.SETTING_VERIFY_SSL,
                label="Verify SSL Certificate",
                input_type="checkbox",
                description="Uncheck to allow self-signed certificates (e.g. local HTTPS).",
                placeholder="",
                required=False,
            ),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_client(self):
        from extensions.pixlstash.pixlstash_client import (
            PixlStashClient,
            PixlStashError,
        )

        url = self.get_setting(self.SETTING_URL)
        token = self.get_setting(self.SETTING_TOKEN)
        if not url or not token:
            raise ValueError(
                "PixlStash URL and API token must be set in AI-Toolkit Settings "
                "before using a PixlStash dataset source."
            )
        verify_ssl = self.get_setting(
            self.SETTING_VERIFY_SSL, default="true"
        ).lower() not in ("false", "0", "no", "off")
        client = PixlStashClient(url, token, verify_ssl=verify_ssl)
        client.login()
        return client

    # ------------------------------------------------------------------
    # Thumbnail
    # ------------------------------------------------------------------

    def get_thumbnail(self, thumbnail_id: str, thumbnail_type: str) -> tuple:
        client = self._make_client()
        if thumbnail_type == "character":
            url = client.character_thumbnail_url(int(thumbnail_id))
        elif thumbnail_type == "set":
            url = client.picture_set_thumbnail_url(int(thumbnail_id))
        else:
            url = client.thumbnail_url(int(thumbnail_id))
        r = client._session.get(url, timeout=30)
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "image/webp")

    # ------------------------------------------------------------------
    # Job config
    # ------------------------------------------------------------------

    def build_job_config(self, params: dict) -> dict:
        verify_ssl_raw = self.get_setting(self.SETTING_VERIFY_SSL, default="true")
        verify_ssl = verify_ssl_raw.lower() not in ("false", "0", "no", "off")
        cfg = {
            "type": "pixlstash_fetch",
            "pixlstash_url": self.get_setting(self.SETTING_URL),
            "pixlstash_token": self.get_setting(self.SETTING_TOKEN),
            "verify_ssl": verify_ssl,
            "source_type": params["source_type"],
            "source_id": int(params["source_id"]),
            "caption_mode": params.get("caption_mode", "description"),
            "overwrite": bool(params.get("overwrite", False)),
        }
        if params.get("trigger_word"):
            cfg["trigger_word"] = params["trigger_word"]
        if params.get("dataset_name"):
            cfg["dataset_name"] = params["dataset_name"]
        score = int(params.get("min_score") or 0)
        if score > 0:
            cfg["min_score"] = score
        return cfg

    # ------------------------------------------------------------------
    # Browse — grouped items for the UI
    # ------------------------------------------------------------------

    def browse(self) -> List[SourceGroup]:
        client = self._make_client()
        characters = client.list_characters()
        picture_sets = client.list_picture_sets()

        char_items = [
            SourceItem(
                id=str(c["id"]),
                name=c["name"],
                picture_count=c.get("picture_count", -1) or -1,
                thumbnail_id=str(c["id"]),
                thumbnail_type="character",
            )
            for c in characters
        ]

        set_items = [
            SourceItem(
                id=str(s["id"]),
                name=s["name"],
                picture_count=s.get("member_count", -1) or -1,
                thumbnail_id=str(s["id"]),
                thumbnail_type="set",
            )
            for s in picture_sets
        ]

        return [
            SourceGroup(id="character", label="Characters", items=char_items),
            SourceGroup(id="picture_set", label="Picture Sets", items=set_items),
        ]

    # ------------------------------------------------------------------
    # Source-specific import fields
    # ------------------------------------------------------------------

    @classmethod
    def get_import_fields(cls) -> List[ImportField]:
        return [
            ImportField(
                id="caption_mode",
                label="Caption Mode",
                field_type="select",
                options=[
                    {"value": "description", "label": "Description"},
                    {"value": "tags", "label": "Tags"},
                    {"value": "both", "label": "Description + Tags"},
                ],
                default="description",
            ),
            ImportField(
                id="min_score",
                label="Minimum Star Rating",
                field_type="select",
                options=[
                    {"value": 0, "label": "All"},
                    {"value": 3, "label": "3★ and above"},
                    {"value": 4, "label": "4★ and above"},
                    {"value": 5, "label": "5★ only"},
                ],
                default=0,
            ),
        ]
