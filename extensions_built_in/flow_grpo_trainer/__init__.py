from toolkit.extension import Extension


class FlowGRPOTrainerExtension(Extension):
    uid = "flow_grpo_trainer"
    name = "Flow GRPO Trainer"

    @classmethod
    def get_process(cls):
        from .FlowGRPOTrainer import FlowGRPOTrainer

        return FlowGRPOTrainer


AI_TOOLKIT_EXTENSIONS = [
    FlowGRPOTrainerExtension,
]
