import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


MIXED_CAPTION_VARIANTS = ('tags', 'nl', 'tags_nl', 'nl_tags')


def _join_with_keep_separator(protected: str, processable: str, separator: str) -> str:
    """Keep a protected prefix marked until later caption processing."""
    return f'{protected.strip()} {separator} {processable.strip()}'.strip()


def _join_caption_parts(*parts: str) -> str:
    """Join ordered caption sections while normalizing their comma boundaries."""
    caption = ''
    for part in parts:
        caption = join_caption_sections(caption, part)
    return caption


@dataclass(frozen=True)
class MixedCaptionSelection:
    """A mixed-caption choice whose tag and natural-language sections stay distinct."""

    variant: str
    protected_tags: str
    processable_tags: str
    nl_caption: str
    keep_tokens_separator: Optional[str] = None

    @property
    def includes_tags(self) -> bool:
        return self.variant in ('tags', 'tags_nl', 'nl_tags')

    @property
    def includes_nl(self) -> bool:
        return self.variant in ('nl', 'tags_nl', 'nl_tags')

    def render(self) -> str:
        """Render the selected raw caption, retaining its keep-token boundary."""
        tags = self.processable_tags if self.includes_tags else ''
        nl_caption = self.nl_caption if self.includes_nl else ''
        if self.variant == 'tags':
            processable = tags
        elif self.variant == 'nl':
            processable = nl_caption
        elif self.variant == 'tags_nl':
            processable = _join_caption_parts(tags, nl_caption)
        else:
            processable = _join_caption_parts(nl_caption, tags)

        if self.keep_tokens_separator is not None:
            return _join_with_keep_separator(
                self.protected_tags,
                processable,
                self.keep_tokens_separator,
            )
        return processable

    def render_processed(
        self,
        protected_tags: str,
        processable_tags: str,
        nl_caption: str,
        extra_tags: str = '',
    ) -> str:
        """Render processed tags around an untouched natural-language section."""
        if self.variant == 'tags':
            parts = (protected_tags, processable_tags, extra_tags)
        elif self.variant == 'nl':
            parts = (protected_tags, nl_caption, extra_tags)
        elif self.variant == 'tags_nl':
            parts = (protected_tags, processable_tags, extra_tags, nl_caption)
        else:
            parts = (protected_tags, nl_caption, processable_tags, extra_tags)
        return _join_caption_parts(*parts)


def select_mixed_caption_selection(
    tags_caption: str,
    nl_caption: str,
    weights: Dict[str, float],
    keep_tokens_separator: Optional[str] = None,
) -> MixedCaptionSelection:
    """Select a weighted variant without flattening its tags and natural language."""
    tags_caption = tags_caption.strip()
    nl_caption = nl_caption.strip()
    protected, processable_tags, has_keep_separator = split_caption_at_separator(
        tags_caption, keep_tokens_separator
    )
    has_tags = bool(protected.strip() or processable_tags.strip())

    if not has_tags:
        selected = 'nl'
    elif not nl_caption:
        selected = 'tags'
    else:
        variant_names = list(MIXED_CAPTION_VARIANTS)
        selected = random.choices(
            variant_names,
            weights=[weights[name] for name in variant_names],
            k=1,
        )[0]

    return MixedCaptionSelection(
        variant=selected,
        protected_tags=protected.strip(),
        processable_tags=processable_tags.strip(),
        nl_caption=nl_caption,
        keep_tokens_separator=(
            keep_tokens_separator if has_keep_separator and has_tags else None
        ),
    )


def select_mixed_caption(
    tags_caption: str,
    nl_caption: str,
    weights: Dict[str, float],
    keep_tokens_separator: Optional[str] = None,
) -> str:
    """Select one weighted caption variant, falling back when one side is absent."""
    return select_mixed_caption_selection(
        tags_caption,
        nl_caption,
        weights,
        keep_tokens_separator,
    ).render()


def _get_caption_tag_groups(caption: str, secondary_separator: Optional[str] = None):
    """Return comma-separated units while retaining secondary tag groups."""
    groups = [group.strip() for group in caption.split(',') if group.strip()]
    if secondary_separator:
        normalized_groups = []
        for group in groups:
            tags = [tag.strip() for tag in group.split(secondary_separator) if tag.strip()]
            if tags:
                normalized_groups.append(secondary_separator.join(tags))
        groups = normalized_groups
    return groups


def shuffle_caption_tags(caption: str, secondary_separator: Optional[str] = None) -> str:
    """Shuffle comma-separated tags or secondary-separated tag groups."""
    groups = _get_caption_tag_groups(caption, secondary_separator)
    random.shuffle(groups)
    return ', '.join(groups)


def drop_caption_tags(
    caption: str,
    dropout_rate: float,
    keep_tokens: int = 0,
    secondary_separator: Optional[str] = None,
) -> str:
    """Randomly drop tags or secondary-separated tag groups."""
    groups = _get_caption_tag_groups(caption, secondary_separator)
    kept_groups = []
    for index, group in enumerate(groups):
        if index < keep_tokens:
            kept_groups.append(group)
        elif dropout_rate < 1.0 and random.random() > dropout_rate:
            kept_groups.append(group)
    return ', '.join(kept_groups)


def expand_secondary_separator(caption: str, secondary_separator: Optional[str]) -> str:
    """Expand grouped tags into a normal comma-separated training caption."""
    if not secondary_separator:
        return caption
    groups = _get_caption_tag_groups(caption, secondary_separator)
    tags = []
    for group in groups:
        tags.extend(tag.strip() for tag in group.split(secondary_separator) if tag.strip())
    return ', '.join(tags)


def split_caption_at_separator(caption: str, separator: Optional[str]) -> Tuple[str, str, bool]:
    """Split a caption into protected and processable sections."""
    if separator and separator in caption:
        protected, processable = caption.split(separator, 1)
        return protected, processable, True
    return '', caption, False


def join_caption_sections(protected: str, processable: str) -> str:
    """Join caption sections after removing their separator boundary."""
    protected = protected.strip().rstrip(',').rstrip()
    processable = processable.strip().lstrip(',').lstrip()
    return ', '.join(part for part in (protected, processable) if part)
