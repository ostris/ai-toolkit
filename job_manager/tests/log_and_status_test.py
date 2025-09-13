
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from job_manager.job_manager import JobManager
from collections import OrderedDict
import time
from typing import Dict

def main():
    # Example callbacks for status and logs (replace with queue pushes)
    def status_callback(status: Dict):
        print(f"[STATUS] {status}")

    def log_callback(line: str):
        print(f"[LOG] {line}")

    # Job configuration from provided YAML, adapted for UITrainer
    job_config = OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', 'job_manager_test'),
            ('process', [
                OrderedDict([
                    ('type', 'ui_trainer'),
                    ('training_folder', '/root/aakashvarma/ai-toolkit/job_manager_test'),
                    ('log_dir', '/root/aakashvarma/ai-toolkit/job_manager_test/job_manager_test/tensorboard'),
                    ('device', 'cuda'),
                    ('trigger_word', None),
                    ('performance_log_every', 1000),
                    ('network', OrderedDict([
                        ('type', 'lora'),
                        ('linear', 32),
                        ('linear_alpha', 32),
                        ('conv', 16),
                        ('conv_alpha', 16),
                        ('lokr_full_rank', True),
                        ('lokr_factor', -1),
                        ('network_kwargs', OrderedDict([
                            ('only_if_contains', ["0.*"])
                        ]))
                    ])),
                    ('save', OrderedDict([
                        ('dtype', 'bf16'),
                        ('save_every', 800),
                        ('start_saving_after', 500),
                        ('max_step_saves_to_keep', 4),
                        ('save_format', 'diffusers'),
                        ('push_to_hub', False)
                    ])),
                    ('datasets', [
                        OrderedDict([
                            ('folder_path', '/root/swetha/ai-toolkit/datasets/alia_bhatt'),
                            ('control_path', None),
                            ('mask_path', None),
                            ('mask_min_value', 0.1),
                            ('trigger_token', '[AB]'),
                            ('initializer_concept', 'A 34 year old Indian woman, 5 feet 1 inch tall, fair skin, slim build, slim jawline, dimples on cheeks'),
                            ('default_caption', ''),
                            ('caption_ext', 'txt'),
                            ('caption_dropout_rate', 0),
                            ('cache_latents_to_disk', False),
                            ('is_reg', False),
                            ('network_weight', 1),
                            ('resolution', [1024, 1024]),
                            ('controls', []),
                            ('shrink_video_to_frames', True),
                            ('num_frames', 1),
                            ('do_i2v', True)
                        ])
                    ]),
                    ('train', OrderedDict([
                        ('batch_size', 2),
                        ('bypass_guidance_embedding', False),
                        ('steps', 7000),
                        ('gradient_accumulation', 1),
                        ('train_unet', True),
                        ('train_text_encoder', False),
                        ('gradient_checkpointing', True),
                        ('noise_scheduler', 'flowmatch'),
                        ('optimizer', 'adamw'),
                        ('timestep_type', 'weighted'),
                        ('content_or_style', 'balanced'),
                        ('optimizer_params', OrderedDict([
                            ('weight_decay', 0.0001)
                        ])),
                        ('enable_ti', True),
                        ('enable_ttb', False),
                        ('early_stopping_num_epochs', 30),
                        ('unload_text_encoder', False),
                        ('cache_text_embeddings', False),
                        ('lr', 0.0001),
                        ('ema_config', OrderedDict([
                            ('use_ema', False),
                            ('ema_decay', 0.99)
                        ])),
                        ('skip_first_sample', False),
                        ('disable_sampling', False),
                        ('dtype', 'bf16'),
                        ('diff_output_preservation', False),
                        ('diff_output_preservation_multiplier', 1),
                        ('diff_output_preservation_class', 'person')
                    ])),
                    ('model', OrderedDict([
                        ('name_or_path', 'Qwen/Qwen-Image'),
                        ('quantize', False),
                        ('qtype', 'qfloat8'),
                        ('quantize_te', False),
                        ('qtype_te', 'qfloat8'),
                        ('arch', 'qwen_image'),
                        ('low_vram', False),
                        ('model_kwargs', OrderedDict())
                    ])),
                    ('sample', OrderedDict([
                        ('sampler', 'flowmatch'),
                        ('sample_every', 250),
                        ('width', 1664),
                        ('height', 928),
                        ('samples', [
                            OrderedDict([
                                ('prompt', '(([RC] man)) and (([AB] woman)) having fun at a beach, high detail, intricate, cinematic lighting, photo realistic'),
                                ('neg', ''),
                                ('seed', 42),
                                ('walk_seed', True),
                                ('guidance_scale', 4),
                                ('sample_steps', 25),
                                ('num_frames', 1),
                                ('fps', 1)
                            ])
                        ])
                    ]))
                ])
            ])
        ])),
        ('meta', OrderedDict([
            ('name', 'job_manager_test'),
            ('version', '1.0')
        ]))
    ])

    # Initialize job manager
    manager = JobManager(toolkit_root="/root/aakashvarma/ai-toolkit", db_path="/root/aakashvarma/ai-toolkit/aitk_db.db")

    # Start a job
    try:
        job_id = manager.start_training(
            job_config=job_config,
            job_name="job_manager_test",
            gpu_ids="0",
            status_callback=status_callback,
            log_callback=log_callback
        )
        print(f"Started job with ID: {job_id}")

        # Monitor status and logs for 60 seconds (for testing)
        time.sleep(60)
        
        # Get status
        status = manager.get_job_status(job_id)
        print(f"Job status: {status}")

        # Get recent logs
        logs = manager.get_rolling_logs(job_id, max_lines=10)
        print("Recent logs:")
        for line in logs:
            print(line)

        # Stop job
        if manager.stop_job(job_id):
            print(f"Job {job_id} stopped successfully")
        else:
            print(f"Failed to stop job {job_id}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
