"""
Tests for DOP (Differential Output Preservation) dataloader integration.

Tests the new clean separation: dataloader handles all embedding I/O,
trainer consumes batch.dop_prompt_embeds directly.
"""

import pytest
from toolkit.prompt_utils import build_dop_replacement_pairs, apply_dop_replacements


class TestDOPTextTransformations:
    """Test DOP text transformation utilities."""

    def test_build_dop_replacement_pairs_simple(self):
        """Test building replacement pairs with equal triggers and classes."""
        pairs, digest = build_dop_replacement_pairs("Jinx, Zapper", "Woman, Gun")

        assert len(pairs) == 2
        # Should be sorted by length DESC
        assert pairs[0] == ("Zapper", "Gun")
        assert pairs[1] == ("Jinx", "Woman")
        assert isinstance(digest, str)
        assert len(digest) > 0

    def test_build_dop_replacement_pairs_more_triggers(self):
        """Test with more triggers than classes - missing classes become empty string."""
        pairs, digest = build_dop_replacement_pairs("Jinx, Zapper, Vest", "Woman, Gun")

        assert len(pairs) == 3
        assert pairs[0] == ("Zapper", "Gun")
        assert pairs[1] == ("Jinx", "Woman")
        assert pairs[2] == ("Vest", "")

    def test_build_dop_replacement_pairs_sorting(self):
        """Test that pairs are sorted by trigger length DESC to avoid substring issues."""
        pairs, _ = build_dop_replacement_pairs("Jinx, Jinx Master", "Woman, Veteran")

        # "Jinx Master" should come first (longer)
        assert len(pairs) == 2
        assert pairs[0] == ("Jinx Master", "Veteran")
        assert pairs[1] == ("Jinx", "Woman")

    def test_apply_dop_replacements_simple(self):
        """Test simple trigger→class replacement."""
        pairs = [("Jinx", "Woman"), ("Zapper", "Gun")]
        result = apply_dop_replacements("Jinx with a Zapper", pairs)

        assert result == "Woman with a Gun"

    def test_apply_dop_replacements_multiple_occurrences(self):
        """Test replacement of multiple occurrences of same trigger."""
        pairs = [("Jinx", "Woman")]
        result = apply_dop_replacements("Jinx likes Jinx", pairs)

        assert result == "Woman likes Woman"

    def test_apply_dop_replacements_longest_first(self):
        """Test that longer triggers are replaced first (substring protection)."""
        pairs = [("Jinx Master", "Veteran"), ("Jinx", "Woman")]  # Already sorted
        result = apply_dop_replacements("Jinx Master and Jinx", pairs)

        # Should replace "Jinx Master" first, then "Jinx"
        assert result == "Veteran and Woman"

    def test_apply_dop_replacements_empty_class(self):
        """Test replacement with empty class (removes trigger)."""
        pairs = [("Vest", ""), ("Jinx", "Woman")]
        result = apply_dop_replacements("Jinx wearing Vest", pairs)

        # "Vest" replaced with empty string
        assert result == "Woman wearing"

    def test_apply_dop_replacements_normalizes_whitespace(self):
        """Test that result has normalized whitespace."""
        pairs = [("item1", ""), ("item2", "foo")]
        result = apply_dop_replacements("item1    item2", pairs)

        # Should collapse repeated whitespace
        assert result == "foo"

    def test_digest_changes_with_inputs(self):
        """Test that digest changes when inputs change."""
        _, digest1 = build_dop_replacement_pairs("Jinx", "Woman")
        _, digest2 = build_dop_replacement_pairs("Jinx", "Girl")
        _, digest3 = build_dop_replacement_pairs("Zapper", "Woman")

        # All should be different
        assert digest1 != digest2
        assert digest1 != digest3
        assert digest2 != digest3

    def test_digest_stable_for_same_inputs(self):
        """Test that digest is stable for identical inputs."""
        _, digest1 = build_dop_replacement_pairs("Jinx, Zapper", "Woman, Gun")
        _, digest2 = build_dop_replacement_pairs("Jinx, Zapper", "Woman, Gun")

        assert digest1 == digest2


class TestDOPIntegration:
    """Integration tests for DOP dataloader patterns."""

    def test_dop_embeds_collation_pattern(self):
        """Test the all-or-nothing collation pattern for DOP embeddings.

        This tests the pattern used in DataLoaderBatchDTO where DOP embeddings
        are only collated if ALL items have them.
        """
        from toolkit.prompt_utils import PromptEmbeds
        import torch

        # Simulate file items with DOP embeddings
        class MockFileItem:
            def __init__(self, has_dop):
                if has_dop:
                    self.dop_prompt_embeds = PromptEmbeds(torch.randn(77, 768))
                else:
                    self.dop_prompt_embeds = None

        # Case 1: All items have DOP embeds
        file_items_all = [MockFileItem(True), MockFileItem(True), MockFileItem(True)]

        dop_list = []
        for x in file_items_all:
            if getattr(x, 'dop_prompt_embeds', None) is None:
                dop_list = None
                break
            dop_list.append(x.dop_prompt_embeds)

        assert dop_list is not None
        assert len(dop_list) == 3

        # Case 2: Some items missing DOP embeds (should fail all-or-nothing check)
        file_items_partial = [MockFileItem(True), MockFileItem(False), MockFileItem(True)]

        dop_list = []
        for x in file_items_partial:
            if getattr(x, 'dop_prompt_embeds', None) is None:
                dop_list = None
                break
            dop_list.append(x.dop_prompt_embeds)

        assert dop_list is None  # All-or-nothing: should be None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
