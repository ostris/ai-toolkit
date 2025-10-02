from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


class MemoryManager:
    def __init__(
        self,
        model: "BaseModel",
    ):
        self.model: "BaseModel" = model
