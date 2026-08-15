from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


SUPPORTED_ACTIVATOR_MODES = frozenset(
    {
        "full",
        "embedding_only",
        "tap_only",
        "internal_only",
        "activator_bypass",
        "stock_literal",
    }
)


class TriggerBindingError(ValueError):
    pass


class TriggerPlaceholderError(TriggerBindingError):
    pass


class TriggerConflictError(TriggerBindingError):
    pass


class TriggerTokenizerError(TriggerBindingError):
    pass


class TriggerTruncationError(TriggerBindingError):
    pass


class TriggerAtomicityError(TriggerBindingError):
    pass


class ActivatorModeError(TriggerBindingError):
    pass


@dataclass(frozen=True)
class ResolvedTriggerText:
    raw_text: str
    text: str
    placeholder: str
    literal: str
    spans: Tuple[Tuple[int, int], ...]

    @property
    def occurrence_count(self) -> int:
        return len(self.spans)


@dataclass(frozen=True)
class TriggerBindingMetadata:
    raw_text: str
    resolved_text: str
    rendered_text: str
    literal: str
    character_spans: Tuple[Tuple[int, int], ...]
    token_spans: Tuple[Tuple[int, int], ...]
    token_indices: Tuple[int, ...]
    virtual_token_indices: Tuple[int, ...]
    occurrence_indices: Tuple[int, ...]
    input_ids: Tuple[int, ...]
    attention_mask: Tuple[int, ...]
    trigger_mask: Tuple[int, ...]
    atomic_token_id: Optional[int] = None
    virtual_tokens: int = 1

    @property
    def occurrence_count(self) -> int:
        return len(self.character_spans)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "resolved_text": self.resolved_text,
            "rendered_text": self.rendered_text,
            "literal": self.literal,
            "character_spans": [list(span) for span in self.character_spans],
            "token_spans": [list(span) for span in self.token_spans],
            "token_indices": list(self.token_indices),
            "virtual_token_indices": list(self.virtual_token_indices),
            "occurrence_indices": list(self.occurrence_indices),
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "trigger_mask": list(self.trigger_mask),
            "atomic_token_id": self.atomic_token_id,
            "virtual_tokens": self.virtual_tokens,
        }


@dataclass
class TriggerBindingBatch:
    items: Tuple[TriggerBindingMetadata, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trigger_mask: torch.Tensor
    token_indices: torch.Tensor
    occurrence_indices: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, *args, **kwargs) -> "TriggerBindingBatch":
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.trigger_mask = self.trigger_mask.to(*args, **kwargs)
        self.token_indices = self.token_indices.to(*args, **kwargs)
        self.occurrence_indices = self.occurrence_indices.to(*args, **kwargs)
        return self

    def runtime_metadata(self) -> Dict[str, Any]:
        return {
            **self.metadata,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class ActivatorRuntimeState:
    mode: str
    embedding_enabled: bool
    internal_enabled: bool
    tap_enabled: bool
    activator_bypassed: bool
    stock_literal: bool


_MODE_STATES = {
    "full": ActivatorRuntimeState("full", True, True, True, False, False),
    "embedding_only": ActivatorRuntimeState("embedding_only", True, False, False, False, False),
    "tap_only": ActivatorRuntimeState("tap_only", False, False, True, False, False),
    "internal_only": ActivatorRuntimeState("internal_only", False, True, False, False, False),
    "activator_bypass": ActivatorRuntimeState("activator_bypass", False, False, False, True, False),
    "stock_literal": ActivatorRuntimeState("stock_literal", False, False, False, False, True),
}


def resolve_trigger_literal(
    raw_text: str,
    literal: str,
    *,
    placeholder: str = "[trigger]",
    require_placeholder: bool = True,
    reject_literal_conflicts: bool = True,
) -> ResolvedTriggerText:
    if not isinstance(raw_text, str):
        raise TriggerPlaceholderError("raw trigger text must be a string")
    if not isinstance(placeholder, str) or not placeholder:
        raise TriggerPlaceholderError("trigger placeholder must be a non-empty string")
    if not isinstance(literal, str) or not literal:
        raise TriggerPlaceholderError("trigger literal must be a non-empty string")

    placeholder_count = raw_text.count(placeholder)
    if require_placeholder and placeholder_count == 0:
        raise TriggerPlaceholderError(f"caption does not contain required placeholder {placeholder!r}")
    if reject_literal_conflicts and literal in raw_text:
        raise TriggerConflictError("raw caption already contains the literal trigger; binding would be ambiguous")

    parts = raw_text.split(placeholder)
    resolved_parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for index, part in enumerate(parts):
        resolved_parts.append(part)
        cursor += len(part)
        if index < len(parts) - 1:
            spans.append((cursor, cursor + len(literal)))
            resolved_parts.append(literal)
            cursor += len(literal)

    return ResolvedTriggerText(
        raw_text=raw_text,
        text="".join(resolved_parts),
        placeholder=placeholder,
        literal=literal,
        spans=tuple(spans),
    )


def find_literal_spans(text: str, literal: str) -> Tuple[Tuple[int, int], ...]:
    if not literal:
        raise TriggerPlaceholderError("trigger literal must be non-empty")
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(literal, start)
        if index < 0:
            return tuple(spans)
        spans.append((index, index + len(literal)))
        start = index + len(literal)


def render_chat_prompt(tokenizer: Any, text: str, *, add_generation_prompt: bool = True) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except (AttributeError, TypeError) as exc:
        raise TriggerTokenizerError("tokenizer must support apply_chat_template(..., tokenize=False)") from exc
    if not isinstance(rendered, str):
        raise TriggerTokenizerError("chat template must return rendered text when tokenize=False")
    return rendered


def _as_flat_list(value: Any, name: str) -> List[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TriggerTokenizerError(f"tokenizer output {name!r} must be a sequence")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise TriggerTokenizerError("single-prompt tokenization unexpectedly returned a batch")
        value = list(value[0])
    return value


def _tokenize_with_offsets(tokenizer: Any, rendered_text: str, max_length: Optional[int]) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    if getattr(tokenizer, "is_fast", None) is False:
        raise TriggerTokenizerError("trigger binding requires a fast tokenizer with offset mapping")
    kwargs: Dict[str, Any] = {
        "add_special_tokens": False,
        "return_offsets_mapping": True,
        "truncation": max_length is not None,
    }
    if max_length is not None:
        if max_length <= 0:
            raise TriggerTokenizerError("max_length must be positive")
        kwargs["max_length"] = int(max_length)
    try:
        encoded = tokenizer(rendered_text, **kwargs)
    except (TypeError, NotImplementedError) as exc:
        raise TriggerTokenizerError("tokenizer does not provide fast offset mapping") from exc
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise TriggerTokenizerError("tokenizer output must contain input_ids and offset_mapping")

    input_ids = [int(value) for value in _as_flat_list(encoded["input_ids"], "input_ids")]
    raw_offsets = encoded["offset_mapping"]
    if isinstance(raw_offsets, torch.Tensor):
        raw_offsets = raw_offsets.detach().cpu().tolist()
    if isinstance(raw_offsets, tuple):
        raw_offsets = list(raw_offsets)
    if (
        isinstance(raw_offsets, list)
        and len(raw_offsets) == 1
        and raw_offsets
        and isinstance(raw_offsets[0], list)
        and (not raw_offsets[0] or isinstance(raw_offsets[0][0], (list, tuple)))
    ):
        raw_offsets = raw_offsets[0]
    if not isinstance(raw_offsets, list):
        raise TriggerTokenizerError("tokenizer output 'offset_mapping' must be a sequence")
    offsets = [(int(pair[0]), int(pair[1])) for pair in raw_offsets]
    if len(input_ids) != len(offsets):
        raise TriggerTokenizerError("input_ids and offset_mapping lengths differ")
    if "attention_mask" in encoded:
        attention_mask = [int(value) for value in _as_flat_list(encoded["attention_mask"], "attention_mask")]
        if len(attention_mask) != len(input_ids):
            raise TriggerTokenizerError("input_ids and attention_mask lengths differ")
    else:
        attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, offsets


def validate_atomic_token_id(
    tokenizer: Any,
    literal: str,
    *,
    expected_token_id: Optional[int] = None,
) -> int:
    try:
        encoded = tokenizer(literal, add_special_tokens=False, truncation=False)
    except TypeError as exc:
        raise TriggerAtomicityError("tokenizer cannot encode literal trigger") from exc
    if "input_ids" not in encoded:
        raise TriggerAtomicityError("tokenizer output does not contain input_ids")
    token_ids = [int(value) for value in _as_flat_list(encoded["input_ids"], "input_ids")]
    if len(token_ids) != 1:
        raise TriggerAtomicityError(
            f"literal trigger must map to exactly one token ID, got {len(token_ids)}: {token_ids}"
        )
    token_id = token_ids[0]
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and token_id == int(unk_token_id):
        raise TriggerAtomicityError("literal trigger maps to tokenizer unknown token ID")
    if expected_token_id is not None and token_id != int(expected_token_id):
        raise TriggerAtomicityError(
            f"literal trigger token ID {token_id} does not match expected ID {int(expected_token_id)}"
        )
    return token_id


def map_trigger_offsets(
    offsets: Sequence[Tuple[int, int]],
    character_spans: Sequence[Tuple[int, int]],
    *,
    mask_all_occurrences: bool = True,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    spans = tuple(character_spans if mask_all_occurrences else character_spans[:1])
    token_indices: List[int] = []
    token_spans: List[Tuple[int, int]] = []
    for char_start, char_end in spans:
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and token_start < char_end and token_end > char_start
        ]
        if not overlapping:
            raise TriggerTruncationError(
                f"trigger occurrence at character span ({char_start}, {char_end}) was truncated or not tokenized"
            )
        covered_start = min(offsets[index][0] for index in overlapping)
        covered_end = max(offsets[index][1] for index in overlapping)
        if covered_start > char_start or covered_end < char_end:
            raise TriggerTruncationError(
                f"trigger occurrence at character span ({char_start}, {char_end}) is only partially tokenized"
            )
        token_indices.extend(overlapping)
        token_spans.append((overlapping[0], overlapping[-1] + 1))
    return tuple(sorted(set(token_indices))), tuple(token_spans)


def bind_trigger_prompt(
    tokenizer: Any,
    raw_text: str,
    literal: str,
    *,
    placeholder: str = "[trigger]",
    max_length: Optional[int] = None,
    require_placeholder: bool = True,
    reject_literal_conflicts: bool = True,
    mask_all_occurrences: bool = True,
    require_atomic: bool = False,
    expected_token_id: Optional[int] = None,
    add_generation_prompt: bool = True,
    virtual_tokens: int = 1,
) -> TriggerBindingMetadata:
    if virtual_tokens not in (1, 2, 4):
        raise TriggerBindingError("virtual_tokens must be one of 1, 2 or 4")
    resolved = resolve_trigger_literal(
        raw_text,
        literal,
        placeholder=placeholder,
        require_placeholder=require_placeholder,
        reject_literal_conflicts=reject_literal_conflicts,
    )
    rendered = render_chat_prompt(tokenizer, resolved.text, add_generation_prompt=add_generation_prompt)
    rendered_spans = find_literal_spans(rendered, literal)
    if len(rendered_spans) != resolved.occurrence_count:
        raise TriggerConflictError(
            "chat template changed or duplicated literal-trigger occurrences; offset mapping is ambiguous"
        )

    tokenizer_max_length = max_length if virtual_tokens == 1 else None
    input_ids, attention_mask, offsets = _tokenize_with_offsets(
        tokenizer, rendered, tokenizer_max_length
    )
    token_indices, token_spans = map_trigger_offsets(
        offsets,
        rendered_spans,
        mask_all_occurrences=mask_all_occurrences,
    )
    atomic_token_id = None
    if require_atomic or expected_token_id is not None or virtual_tokens > 1:
        atomic_token_id = validate_atomic_token_id(
            tokenizer,
            literal,
            expected_token_id=expected_token_id,
        )
        for index in token_indices:
            if input_ids[index] != atomic_token_id:
                raise TriggerAtomicityError(
                    "literal is atomic in isolation but has a different contextual token ID"
                )
        expected_occurrences = resolved.occurrence_count if mask_all_occurrences else min(1, resolved.occurrence_count)
        if len(token_indices) != expected_occurrences:
            raise TriggerAtomicityError("each masked trigger occurrence must map to exactly one token")

    expanded_ids: List[int] = []
    expanded_attention: List[int] = []
    expanded_mask: List[int] = []
    expanded_virtual_indices: List[int] = []
    expanded_occurrence_indices: List[int] = []
    expanded_token_indices: List[int] = []
    selected_positions = {position: occurrence for occurrence, position in enumerate(token_indices)}
    for original_index, token_id in enumerate(input_ids):
        occurrence = selected_positions.get(original_index)
        if occurrence is None:
            expanded_ids.append(token_id)
            expanded_attention.append(attention_mask[original_index])
            expanded_mask.append(0)
            expanded_virtual_indices.append(0)
            expanded_occurrence_indices.append(-1)
            continue
        for virtual_index in range(virtual_tokens):
            expanded_token_indices.append(len(expanded_ids))
            expanded_ids.append(token_id)
            expanded_attention.append(attention_mask[original_index])
            expanded_mask.append(1)
            expanded_virtual_indices.append(virtual_index)
            expanded_occurrence_indices.append(occurrence)

    if max_length is not None and len(expanded_ids) > max_length:
        raise TriggerTruncationError(
            "virtual-token expansion exceeds max_length; increase the text limit or reduce virtual tokens"
        )

    return TriggerBindingMetadata(
        raw_text=raw_text,
        resolved_text=resolved.text,
        rendered_text=rendered,
        literal=literal,
        character_spans=rendered_spans,
        token_spans=token_spans,
        token_indices=tuple(expanded_token_indices),
        virtual_token_indices=tuple(expanded_virtual_indices),
        occurrence_indices=tuple(expanded_occurrence_indices),
        input_ids=tuple(expanded_ids),
        attention_mask=tuple(expanded_attention),
        trigger_mask=tuple(expanded_mask),
        atomic_token_id=atomic_token_id,
        virtual_tokens=virtual_tokens,
    )


def bind_trigger_batch(
    tokenizer: Any,
    raw_texts: Sequence[str],
    literal: str,
    *,
    pad_token_id: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **binding_kwargs: Any,
) -> TriggerBindingBatch:
    if not raw_texts:
        raise TriggerBindingError("trigger binding batch must not be empty")
    items = tuple(bind_trigger_prompt(tokenizer, text, literal, **binding_kwargs) for text in raw_texts)
    max_tokens = max(len(item.input_ids) for item in items)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    ids: List[List[int]] = []
    attention: List[List[int]] = []
    masks: List[List[int]] = []
    virtual_indices: List[List[int]] = []
    occurrence_indices: List[List[int]] = []
    for item in items:
        padding = max_tokens - len(item.input_ids)
        ids.append(list(item.input_ids) + [int(pad_token_id)] * padding)
        attention.append(list(item.attention_mask) + [0] * padding)
        masks.append(list(item.trigger_mask) + [0] * padding)
        virtual_indices.append(list(item.virtual_token_indices) + [0] * padding)
        occurrence_indices.append(list(item.occurrence_indices) + [-1] * padding)

    batch_metadata = dict(metadata or {})
    batch_metadata.update(
        {
            "architecture_version": 8,
            "batch_size": len(items),
            "sequence_length": max_tokens,
            "occurrence_counts": [item.occurrence_count for item in items],
            "token_indices": [list(item.token_indices) for item in items],
            "character_spans": [[list(span) for span in item.character_spans] for item in items],
            "virtual_tokens": [item.virtual_tokens for item in items],
            "literal": literal,
        }
    )
    return TriggerBindingBatch(
        items=items,
        input_ids=torch.tensor(ids, dtype=torch.long),
        attention_mask=torch.tensor(attention, dtype=torch.long),
        trigger_mask=torch.tensor(masks, dtype=torch.bool),
        token_indices=torch.tensor(virtual_indices, dtype=torch.long),
        occurrence_indices=torch.tensor(occurrence_indices, dtype=torch.long),
        metadata=batch_metadata,
    )


def get_activator_runtime_state(mode: str) -> ActivatorRuntimeState:
    try:
        return _MODE_STATES[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_ACTIVATOR_MODES))
        raise ActivatorModeError(f"unsupported activator runtime mode {mode!r}; expected one of: {supported}") from exc


def _get_runtime_mode(target: Any) -> Any:
    if isinstance(target, Mapping):
        return target.get("runtime_mode")
    return getattr(target, "runtime_mode", None)


def _set_runtime_mode(target: Any, mode: Any) -> None:
    if isinstance(target, MutableMapping):
        target["runtime_mode"] = mode
        return
    setattr(target, "runtime_mode", mode)


def _delete_runtime_mode(target: Any) -> None:
    if isinstance(target, MutableMapping):
        target.pop("runtime_mode", None)
        return
    try:
        delattr(target, "runtime_mode")
    except AttributeError:
        pass


@contextmanager
def activator_runtime_mode(target: Any, mode: str) -> Iterator[ActivatorRuntimeState]:
    state = get_activator_runtime_state(mode)
    if target is None:
        raise ActivatorModeError("activator runtime target must not be None")
    existed = "runtime_mode" in target if isinstance(target, Mapping) else hasattr(target, "runtime_mode")
    previous = _get_runtime_mode(target)
    _set_runtime_mode(target, mode)
    try:
        yield state
    finally:
        if existed:
            _set_runtime_mode(target, previous)
        else:
            _delete_runtime_mode(target)
