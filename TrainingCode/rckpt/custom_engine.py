# custom_engine.py
from deepspeed.runtime.engine import DeepSpeedEngine

class CustomDeepSpeedEngine(DeepSpeedEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("This is new CustomDeepSpeedEngine")
        
    def rckpt_save_checkpoint(self, save_dir, tag, client_state={}, exclude_frozen_parameters=False):
        print("This is new save_checkpoint function")
        super().save_checkpoint(save_dir, client_state, tag)