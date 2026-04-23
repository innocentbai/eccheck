import os
import torch
import torch.distributed as dist
import concurrent.futures
from collections import OrderedDict
from threading import Lock
from typing import Union
import deepspeed

class DistributedAsyncUpdater:
    def __init__(self, 
                 model: Union[torch.nn.Module, torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel],
                 learning_rate: float,
                 num_threads: int = None,
                 rank: int = 0):
        """
        多GPU适用的异步参数更新器
        Args:
            model: 支持普通Module/DataParallel/DistributedDataParallel
            learning_rate: 学习率
            num_threads: 存储线程数（默认CPU核心数半数）
            rank: 主进程编号（DDP模式使用）
        """
        # 提取实际模型（适配并行包装器）
        self.raw_model = model.module if isinstance(model, (torch.nn.parallel.DataParallel, 
                                                          torch.nn.parallel.DistributedDataParallel)) else model
        self.lr = learning_rate
        self.rank = rank
        
        # 获取主设备参数（确保在正确设备上）
        self.param_dict = OrderedDict(reversed(list(self.raw_model.named_parameters())))
        self.updated_params = {}
        self.param_snapshots = {}
        
        # 异步存储配置
        num_threads = num_threads or min(4, os.cpu_count())
        self.storage_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
        self.snapshot_lock = Lock()
        
        # 注册梯度钩子（仅在主设备）
        if self._is_main_process():
            self._register_update_hooks()
            print(f"初始化完成，存储线程: {num_threads}, 学习率: {learning_rate}")

    def _is_main_process(self) -> bool:
        """判断当前是否为主进程"""
        return self.rank == 0

    def _compute_and_store(self, grad: torch.Tensor, param_name: str) -> torch.Tensor:
        """多GPU安全的更新值计算与存储"""
        if not self._is_main_process():
            return grad  # 仅主进程处理
        
        # 获取主设备参数
        param = self.param_dict[param_name]
        
        # 确保梯度来自聚合结果
        if param.grad is not None:
            grad = param.grad  # 使用已聚合的梯度                  #聚合完了么?
            
        # 计算更新值
        with torch.no_grad():
            updated = param.data - self.lr * grad
        
        # 保存更新值（后续统一应用）
        self.updated_params[param_name] = updated.detach().clone()#额外开销?
        # 提交异步存储（仅在主进程）
        self._async_snapshot(param_name, updated.cpu()) #pin memory?
        return grad

    def _async_snapshot(self, param_name: str, data: torch.Tensor):
        """跨设备安全的存储操作"""
        if not self._is_main_process():
            return
        
        # 确保数据在CPU并克隆
        cpu_data = data.clone() if data.device.type != 'cpu' else data.clone()#又存一份?
        self.storage_pool.submit(
            self._safe_store_snapshot,
            param_name,
            cpu_data
        )

    def _safe_store_snapshot(self, name: str, data: torch.Tensor):
        """分布式环境下的线程安全存储"""
        with self.snapshot_lock:
            if name not in self.param_snapshots:
                self.param_snapshots[name] = []     #阻碍训练
            self.param_snapshots[name].append(data) #放线程里?

    def _register_update_hooks(self):
        """为所有参数注册梯度钩子（仅主进程）"""
        for name, param in self.param_dict.items():
            if param.requires_grad:
                param.register_hook(
                    lambda grad, name=name: self._compute_and_store(grad, name)
                )

    def apply_updates(self):
        """应用更新并同步到所有设备"""
        # 仅主进程执行更新
        if self._is_main_process():
            with torch.no_grad():
                for name, param in self.param_dict.items():
                    if name in self.updated_params:
                        param.data.copy_(self.updated_params[name])
            
            # 清空缓存
            self.updated_params.clear()

        # DDP模式下的跨进程同步
        if isinstance(self.raw_model, torch.nn.parallel.DistributedDataParallel):
            for param in self.raw_model.parameters():
                dist.broadcast(param.data, src=0)

    def get_snapshots(self, param_name: str) -> list:
        """获取参数历史（主进程独占）"""
        if not self._is_main_process():
            raise RuntimeError("仅主进程可访问参数历史")
        
        with self.snapshot_lock:
            return [t.clone() for t in self.param_snapshots.get(param_name, [])]

    def cleanup(self):
        """资源释放"""
        self.storage_pool.shutdown()

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 初始化分布式环境（示例）
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    # 创建模型并包装
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 10)
    ).cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # 初始化更新器
    updater = DistributedAsyncUpdater(ddp_model, learning_rate=0.01, rank=rank)
    
    # 训练步骤
    inputs = torch.randn(32, 1024).cuda()
    labels = torch.randint(0, 10, (32,)).cuda()
    
    # 前向传播
    outputs = ddp_model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    
    # 反向传播（自动触发梯度聚合与存储）
    loss.backward()
    
    # 应用更新并同步
    updater.apply_updates()
    
    # 主进程打印结果
    if rank == 0:
        print("第一层参数版本数:", len(updater.get_snapshots('0.weight')))
    updater.cleanup()