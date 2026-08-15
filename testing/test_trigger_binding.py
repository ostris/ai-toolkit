import json
import unittest

import torch

from toolkit.trigger_binding import (
    ActivatorModeError,
    TriggerAtomicityError,
    TriggerConflictError,
    TriggerPlaceholderError,
    TriggerTokenizerError,
    TriggerTruncationError,
    activator_runtime_mode,
    bind_trigger_batch,
    bind_trigger_prompt,
    get_activator_runtime_state,
    resolve_trigger_literal,
    validate_atomic_token_id,
)


class _FastTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = 1

    def __init__(self, literal="<trigger>", atomic=True):
        self.literal = literal
        self.atomic = atomic
        self.literal_id = 700

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = messages[0]["content"][0]["text"]
        suffix = "<assistant>" if add_generation_prompt else ""
        return f"<user>{text}</user>{suffix}"

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        truncation=False,
        max_length=None,
    ):
        input_ids = []
        offsets = []
        index = 0
        while index < len(text):
            if self.atomic and text.startswith(self.literal, index):
                input_ids.append(self.literal_id)
                offsets.append((index, index + len(self.literal)))
                index += len(self.literal)
            else:
                input_ids.append(100 + (ord(text[index]) % 500))
                offsets.append((index, index + 1))
                index += 1
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
            offsets = offsets[:max_length]
        result = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class _SlowTokenizer(_FastTokenizer):
    is_fast = False


class _DuplicatingTemplateTokenizer(_FastTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = messages[0]["content"][0]["text"]
        return f"{text} {self.literal}"


class _Runtime:
    def __init__(self, runtime_mode="full"):
        self.runtime_mode = runtime_mode


class TriggerBindingTest(unittest.TestCase):
    def setUp(self):
        self.literal = "<trigger>"
        self.tokenizer = _FastTokenizer(self.literal)

    def test_resolver_replaces_all_occurrences_and_records_spans(self):
        resolved = resolve_trigger_literal(
            "alpha [trigger] beta [trigger] omega",
            self.literal,
        )
        self.assertEqual(resolved.text, "alpha <trigger> beta <trigger> omega")
        self.assertEqual(resolved.occurrence_count, 2)
        self.assertEqual(
            tuple(resolved.text[start:end] for start, end in resolved.spans),
            (self.literal, self.literal),
        )

    def test_resolver_rejects_missing_placeholder_and_literal_conflict(self):
        with self.assertRaises(TriggerPlaceholderError):
            resolve_trigger_literal("plain caption", self.literal)
        with self.assertRaises(TriggerConflictError):
            resolve_trigger_literal("[trigger] plus <trigger>", self.literal)

    def test_chat_template_offsets_create_all_occurrence_mask(self):
        binding = bind_trigger_prompt(
            self.tokenizer,
            "alpha [trigger] beta [trigger]",
            self.literal,
            require_atomic=True,
            expected_token_id=self.tokenizer.literal_id,
        )
        self.assertTrue(binding.rendered_text.startswith("<user>alpha"))
        self.assertEqual(binding.occurrence_count, 2)
        self.assertEqual(len(binding.token_indices), 2)
        self.assertEqual(sum(binding.trigger_mask), 2)
        self.assertTrue(all(binding.input_ids[index] == self.tokenizer.literal_id for index in binding.token_indices))
        self.assertEqual(
            tuple(binding.rendered_text[start:end] for start, end in binding.character_spans),
            (self.literal, self.literal),
        )

    def test_first_occurrence_mode_leaves_other_occurrences_unmasked(self):
        binding = bind_trigger_prompt(
            self.tokenizer,
            "[trigger] then [trigger]",
            self.literal,
            mask_all_occurrences=False,
        )
        self.assertEqual(binding.occurrence_count, 2)
        self.assertEqual(len(binding.token_indices), 1)
        self.assertEqual(sum(binding.trigger_mask), 1)

    def test_truncation_is_detected_instead_of_silently_dropping_trigger(self):
        with self.assertRaises(TriggerTruncationError):
            bind_trigger_prompt(
                self.tokenizer,
                "a long prefix [trigger]",
                self.literal,
                max_length=5,
            )

    def test_chat_template_literal_duplication_is_rejected(self):
        tokenizer = _DuplicatingTemplateTokenizer(self.literal)
        with self.assertRaises(TriggerConflictError):
            bind_trigger_prompt(tokenizer, "[trigger]", self.literal)

    def test_fast_tokenizer_and_atomic_id_validation(self):
        with self.assertRaises(TriggerTokenizerError):
            bind_trigger_prompt(_SlowTokenizer(self.literal), "[trigger]", self.literal)
        self.assertEqual(validate_atomic_token_id(self.tokenizer, self.literal), self.tokenizer.literal_id)
        with self.assertRaises(TriggerAtomicityError):
            validate_atomic_token_id(self.tokenizer, self.literal, expected_token_id=999)
        with self.assertRaises(TriggerAtomicityError):
            validate_atomic_token_id(_FastTokenizer(self.literal, atomic=False), self.literal)

    def test_batch_padding_masks_and_metadata(self):
        batch = bind_trigger_batch(
            self.tokenizer,
            ["[trigger]", "longer [trigger] and [trigger]"],
            self.literal,
            require_atomic=True,
            metadata={"phase": "a1"},
        )
        self.assertEqual(batch.input_ids.shape, batch.attention_mask.shape)
        self.assertEqual(batch.input_ids.shape, batch.trigger_mask.shape)
        self.assertEqual(batch.input_ids.shape, batch.token_indices.shape)
        self.assertEqual(batch.input_ids.dtype, torch.long)
        self.assertEqual(batch.trigger_mask.dtype, torch.bool)
        self.assertEqual(batch.trigger_mask.sum(dim=1).tolist(), [1, 2])
        self.assertEqual(batch.metadata["batch_size"], 2)
        self.assertEqual(batch.metadata["occurrence_counts"], [1, 2])
        self.assertEqual(batch.metadata["phase"], "a1")

    def test_virtual_tokens_expand_each_occurrence_and_keep_additive_mask(self):
        binding = bind_trigger_prompt(
            self.tokenizer,
            "[trigger] then [trigger]",
            self.literal,
            require_atomic=True,
            virtual_tokens=4,
        )
        self.assertEqual(binding.virtual_tokens, 4)
        self.assertEqual(len(binding.token_indices), 8)
        self.assertEqual(sum(binding.trigger_mask), 8)
        self.assertEqual(
            [binding.virtual_token_indices[index] for index in binding.token_indices],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )
        self.assertEqual(
            [binding.occurrence_indices[index] for index in binding.token_indices],
            [0, 0, 0, 0, 1, 1, 1, 1],
        )

    def test_runtime_metadata_is_json_serializable(self):
        batch = bind_trigger_batch(
            self.tokenizer,
            ["[trigger]", "[trigger] and [trigger]"],
            self.literal,
            require_atomic=True,
            virtual_tokens=2,
        )
        encoded = json.dumps(batch.runtime_metadata())
        self.assertIn('"architecture_version": 8', encoded)
        self.assertEqual(batch.trigger_mask.sum(dim=1).tolist(), [2, 4])

    def test_all_runtime_modes_expose_expected_flags(self):
        full = get_activator_runtime_state("full")
        self.assertTrue(full.embedding_enabled and full.internal_enabled and full.tap_enabled)
        self.assertTrue(get_activator_runtime_state("embedding_only").embedding_enabled)
        self.assertTrue(get_activator_runtime_state("tap_only").tap_enabled)
        self.assertTrue(get_activator_runtime_state("internal_only").internal_enabled)
        self.assertTrue(get_activator_runtime_state("activator_bypass").activator_bypassed)
        self.assertTrue(get_activator_runtime_state("stock_literal").stock_literal)
        with self.assertRaises(ActivatorModeError):
            get_activator_runtime_state("unknown")

    def test_runtime_context_is_nested_and_exception_safe_for_objects(self):
        runtime = _Runtime("full")
        with activator_runtime_mode(runtime, "embedding_only"):
            self.assertEqual(runtime.runtime_mode, "embedding_only")
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with activator_runtime_mode(runtime, "tap_only"):
                    self.assertEqual(runtime.runtime_mode, "tap_only")
                    raise RuntimeError("boom")
            self.assertEqual(runtime.runtime_mode, "embedding_only")
        self.assertEqual(runtime.runtime_mode, "full")

    def test_runtime_context_restores_mapping_and_removes_new_attribute(self):
        runtime = {"runtime_mode": "stock_literal"}
        with activator_runtime_mode(runtime, "activator_bypass"):
            self.assertEqual(runtime["runtime_mode"], "activator_bypass")
        self.assertEqual(runtime["runtime_mode"], "stock_literal")

        class Empty:
            pass

        empty = Empty()
        with activator_runtime_mode(empty, "internal_only"):
            self.assertEqual(empty.runtime_mode, "internal_only")
        self.assertFalse(hasattr(empty, "runtime_mode"))


if __name__ == "__main__":
    unittest.main()
