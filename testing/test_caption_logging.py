import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from toolkit.caption_logging import (
    emit_caption_log,
    format_caption_debug_log,
    reset_caption_log_listener,
    set_caption_log_listener,
    should_log_captions,
)


ROOT = Path(__file__).resolve().parents[1]


def _make_file_item(
    path,
    caption,
    *,
    caption_short=None,
    is_reg=False,
):
    return SimpleNamespace(
        path=path,
        caption=caption,
        caption_short=caption_short,
        is_reg=is_reg,
    )


def _make_batch(file_items, *, cached_embedding=False):
    return SimpleNamespace(
        file_items=list(file_items),
        prompt_embeds=object() if cached_embedding else None,
    )


def _load_logging_config_class():
    """Load LoggingConfig without importing the training dependency stack."""
    source_path = ROOT / 'toolkit' / 'config_modules.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'LoggingConfig'
    )
    namespace = {'math': math}
    module = ast.Module(body=[class_node], type_ignores=[])
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['LoggingConfig']


class CaptionLoggingCadenceTest(unittest.TestCase):
    def test_disabled_interval_never_logs(self):
        for completed_step in (0, 1, 100, 200):
            with self.subTest(completed_step=completed_step):
                self.assertFalse(should_log_captions(completed_step, 0))

    def test_interval_100_logs_only_on_completed_steps_100_and_200(self):
        expected = {
            0: False,
            1: False,
            99: False,
            100: True,
            101: False,
            199: False,
            200: True,
            201: False,
        }
        for completed_step, should_log in expected.items():
            with self.subTest(completed_step=completed_step):
                self.assertEqual(
                    should_log_captions(completed_step, 100),
                    should_log,
                )

    def test_first_completed_step_is_step_one(self):
        self.assertFalse(should_log_captions(0, 1))
        self.assertTrue(should_log_captions(1, 1))


class CaptionDebugFormattingTest(unittest.TestCase):
    def test_formats_two_microbatches_and_all_items(self):
        batches = [
            _make_batch(
                [
                    _make_file_item(
                        'dataset/a.png',
                        'long A',
                        caption_short='short A',
                    ),
                    _make_file_item(
                        'dataset/b.png',
                        'same B',
                        caption_short='same B',
                        is_reg=True,
                    ),
                ],
            ),
            _make_batch(
                [
                    _make_file_item('dataset/c.png', 'long C'),
                    _make_file_item('dataset/d.png', 'long D'),
                ],
                cached_embedding=True,
            ),
        ]

        message = format_caption_debug_log(100, batches)

        self.assertIsNotNone(message)
        lines = message.splitlines()
        self.assertEqual(len(lines), 4)

        self.assertIn('[caption-debug] step=100', lines[0])
        self.assertIn('microbatch=1/2', lines[0])
        self.assertIn('item=1/2', lines[0])
        self.assertIn('reg=false', lines[0])
        self.assertIn('cached_embedding=false', lines[0])
        self.assertIn(f'file={json.dumps("a.png")}', lines[0])
        self.assertIn(f'caption={json.dumps("long A")}', lines[0])
        self.assertIn(f'caption_short={json.dumps("short A")}', lines[0])

        self.assertIn('microbatch=1/2', lines[1])
        self.assertIn('item=2/2', lines[1])
        self.assertIn('reg=true', lines[1])
        self.assertNotIn('caption_short=', lines[1])

        self.assertIn('microbatch=2/2', lines[2])
        self.assertIn('item=1/2', lines[2])
        self.assertIn('cached_embedding=true', lines[2])
        self.assertIn(f'caption={json.dumps("long C")}', lines[2])

        self.assertIn('microbatch=2/2', lines[3])
        self.assertIn('item=2/2', lines[3])
        self.assertIn(f'file={json.dumps("d.png")}', lines[3])

    def test_json_escapes_path_caption_and_short_caption(self):
        path = 'dataset/a "quoted".png'
        basename = 'a "quoted".png'
        caption = 'first line\nsecond "line" \\ tail'
        caption_short = 'short\n"caption"'
        batch = _make_batch(
            [
                _make_file_item(
                    path,
                    caption,
                    caption_short=caption_short,
                )
            ]
        )

        message = format_caption_debug_log(1, [batch])

        self.assertEqual(len(message.splitlines()), 1)
        self.assertIn(
            f'file={json.dumps(basename, ensure_ascii=False)}',
            message,
        )
        self.assertIn(f'caption={json.dumps(caption, ensure_ascii=False)}', message)
        self.assertIn(
            f'caption_short={json.dumps(caption_short, ensure_ascii=False)}',
            message,
        )
        self.assertNotIn('first line\nsecond', message)

    def test_returns_none_when_there_are_no_file_items(self):
        self.assertIsNone(format_caption_debug_log(100, []))
        self.assertIsNone(format_caption_debug_log(100, [_make_batch([])]))


class CaptionLogListenerTest(unittest.TestCase):
    def test_context_listener_is_restored_when_token_is_reset(self):
        outer_messages = []
        inner_messages = []
        outer_token = set_caption_log_listener(outer_messages.append)
        try:
            emit_caption_log('outer before')
            inner_token = set_caption_log_listener(inner_messages.append)
            try:
                emit_caption_log('inner')
            finally:
                reset_caption_log_listener(inner_token)
            emit_caption_log('outer after')
        finally:
            reset_caption_log_listener(outer_token)

        self.assertEqual(outer_messages, ['outer before', 'outer after'])
        self.assertEqual(inner_messages, ['inner'])

    def test_listener_failure_is_non_fatal(self):
        def failing_listener(_message):
            raise RuntimeError('listener failed')

        token = set_caption_log_listener(failing_listener)
        try:
            emitted = emit_caption_log('caption debug message')
        finally:
            reset_caption_log_listener(token)

        self.assertFalse(emitted)


class TrainingLoopCaptionLoggingWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = ROOT / 'jobs' / 'process' / 'BaseSDTrainProcess.py'
        tree = ast.parse(cls.source_path.read_text(encoding='utf-8'), filename=str(cls.source_path))
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'BaseSDTrainProcess'
        )
        cls.run_method = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == 'run'
        )

    def test_uses_completed_step_and_all_accumulated_batches_once(self):
        completed_step_assignments = [
            node
            for node in ast.walk(self.run_method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == 'completed_step'
                for target in node.targets
            )
        ]
        self.assertEqual(len(completed_step_assignments), 1)
        self.assertEqual(
            ast.unparse(completed_step_assignments[0].value),
            'self.step_num + 1',
        )

        format_calls = [
            node
            for node in ast.walk(self.run_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'format_caption_debug_log'
        ]
        emit_calls = [
            node
            for node in ast.walk(self.run_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'emit_caption_log'
        ]
        self.assertEqual(len(format_calls), 1)
        self.assertEqual(len(emit_calls), 1)
        self.assertEqual(
            [ast.unparse(argument) for argument in format_calls[0].args],
            ['completed_step', 'batch_list'],
        )

    def test_skips_oom_and_non_main_process_before_cleanup(self):
        guard = next(
            node
            for node in ast.walk(self.run_method)
            if isinstance(node, ast.If)
            and 'should_log_captions' in ast.unparse(node.test)
        )
        guard_source = ast.unparse(guard.test)
        self.assertIn('not did_oom', guard_source)
        self.assertIn('self.accelerator.is_main_process', guard_source)
        self.assertIn(
            'should_log_captions(completed_step, caption_log_interval)',
            guard_source,
        )

        train_call = next(
            node
            for node in ast.walk(self.run_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'hook_train_loop'
        )
        cleanup_call = next(
            node
            for node in ast.walk(self.run_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'cleanup'
        )
        self.assertLess(train_call.lineno, guard.lineno)
        self.assertLess(guard.lineno, cleanup_call.lineno)


class LoggingConfigCaptionIntervalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.LoggingConfig = _load_logging_config_class()

    def test_defaults_to_disabled_and_accepts_none(self):
        self.assertEqual(self.LoggingConfig().log_captions_every_n_steps, 0)
        self.assertEqual(
            self.LoggingConfig(log_captions_every_n_steps=None).log_captions_every_n_steps,
            0,
        )

    def test_accepts_integer_strings_and_integer_valued_floats(self):
        for raw_value in ('100', '100.0', 100, 100.0):
            with self.subTest(raw_value=raw_value):
                config = self.LoggingConfig(log_captions_every_n_steps=raw_value)
                self.assertEqual(config.log_captions_every_n_steps, 100)
                self.assertIsInstance(config.log_captions_every_n_steps, int)

    def test_rejects_bool_fraction_negative_and_non_finite_values(self):
        invalid_values = (
            True,
            False,
            1.5,
            '1.5',
            -1,
            '-1',
            float('nan'),
            float('inf'),
            float('-inf'),
            'nan',
            'inf',
        )
        for raw_value in invalid_values:
            with self.subTest(raw_value=raw_value):
                with self.assertRaises(ValueError):
                    self.LoggingConfig(log_captions_every_n_steps=raw_value)


class FluxTrainUiCaptionLoggingWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = ROOT / 'flux_train_ui.py'
        cls.source = cls.source_path.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.source, filename=str(cls.source_path))
        cls.start_training = next(
            node
            for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == 'start_training'
        )

    def test_start_training_signature_and_config_assignment(self):
        argument_names = [argument.arg for argument in self.start_training.args.args]
        separator_index = argument_names.index('secondary_separator')
        logging_index = argument_names.index('log_captions_every_n_steps')
        dataset_index = argument_names.index('dataset_folder')
        self.assertEqual(logging_index, separator_index + 1)
        self.assertEqual(dataset_index, logging_index + 1)

        config_assignments = [
            node
            for node in ast.walk(self.start_training)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(part, ast.Constant)
                and part.value == 'log_captions_every_n_steps'
                for part in ast.walk(node)
            )
        ]
        self.assertTrue(config_assignments)
        assignment_source = '\n'.join(
            ast.unparse(assignment) for assignment in config_assignments
        )
        self.assertIn('int(log_captions_every_n_steps', assignment_source)

    def test_number_control_and_click_input_order(self):
        number_assignment = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == 'log_captions_every_n_steps'
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
        )
        keyword_values = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in number_assignment.value.keywords
            if keyword.arg in {'value', 'minimum', 'step'}
        }
        self.assertEqual(
            keyword_values,
            {'value': 0, 'minimum': 0, 'step': 1},
        )

        then_call = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'then'
            and any(
                keyword.arg == 'fn'
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == 'start_training'
                for keyword in node.keywords
            )
        )
        inputs_node = next(
            keyword.value for keyword in then_call.keywords if keyword.arg == 'inputs'
        )
        input_names = [
            element.id for element in inputs_node.elts if isinstance(element, ast.Name)
        ]
        separator_index = input_names.index('secondary_separator')
        logging_index = input_names.index('log_captions_every_n_steps')
        dataset_index = input_names.index('dataset_folder')
        self.assertEqual(logging_index, separator_index + 1)
        self.assertEqual(dataset_index, logging_index + 1)

    def test_training_is_a_live_generator_with_scoped_listener(self):
        self.assertTrue(
            any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(self.start_training))
        )
        called_names = {
            node.func.id
            for node in ast.walk(self.start_training)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn('set_caption_log_listener', called_names)
        self.assertIn('reset_caption_log_listener', called_names)

    def test_gradio_rejects_invalid_mixed_weights_before_writing_config(self):
        function_source = ast.unparse(self.start_training)
        self.assertIn('math.isfinite', function_source)
        self.assertIn("caption_mode == 'mixed' and mixed_weight_total <= 0", function_source)
        validation_line = next(
            node.lineno
            for node in ast.walk(self.start_training)
            if isinstance(node, ast.If)
            and "caption_mode == 'mixed'" in ast.unparse(node.test)
        )
        config_line = next(
            node.lineno
            for node in ast.walk(self.start_training)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(part, ast.Constant) and part.value == 'mixed_weights'
                for part in ast.walk(node)
            )
        )
        self.assertLess(validation_line, config_line)

    def test_ai_captioning_filters_uploaded_text_sidecars(self):
        run_captioning = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'run_captioning'
        )
        function_source = ast.unparse(run_captioning)
        self.assertIn('_is_image_file(image)', function_source)
        self.assertIn('enumerate(uploaded_images)', function_source)


if __name__ == '__main__':
    unittest.main()
