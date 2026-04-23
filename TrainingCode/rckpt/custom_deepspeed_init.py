# custom_init.py
import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from rckpt.custom_engine import CustomDeepSpeedEngine

# 保存原始 initialize 函数
_original_initialize = deepspeed.initialize

def custom_initialize(*args, **kwargs):
    # 调用原始初始化函数
    engine, optimizer, training_dataloader, lr_scheduler = _original_initialize(*args, **kwargs)
    
    # 动态将 Engine 替换为自定义类
    engine.__class__ = CustomDeepSpeedEngine
    
    return engine, optimizer, training_dataloader, lr_scheduler

