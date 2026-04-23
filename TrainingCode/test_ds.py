import sys
sys.path.insert(0, "/home/lff/RCKPT/DeepSpeed")

# 清除可能的缓存
if 'deepspeed' in sys.modules:
    del sys.modules['deepspeed']
if 'deepspeed.ops' in sys.modules:
    del sys.modules['deepspeed.ops']

try:
    import deepspeed
    print(f"Loaded DeepSpeed from: {deepspeed.__file__}")
except ImportError:
    print("DeepSpeed not found!")