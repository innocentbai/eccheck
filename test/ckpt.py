from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import copy
 
# 定义模型
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(6, 6)
        self.fc1 = nn.Linear(6 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 6 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyDataset(Dataset):
    def __init__(self):
        # 示例：生成随机数据
        self.data = torch.randn(8, 3, 32, 32)  # 输入形状 (batch, channels, height, width)
        self.labels = torch.randint(0, 10, (8,))  # 分类任务，标签范围 0-9
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def _compare(s1, s2):        
    if s1['state'].keys() != s2['state'].keys():
        return False, "state的key不一致"

    for key in s1['state'].keys():
        val1 = s1['state'][key]
        val2 = s2['state'][key]
        if val1.keys() != val2.keys():
            return False, "state的属性类型不一致"
        
    for key in s1['state'].keys():
        val1 = s1['state'][key]
        val2 = s2['state'][key]
        for kk in val1.keys():
            vv1 = val1[kk]
            vv2 = val2[kk]
            if isinstance(vv1, torch.Tensor):
                if not torch.equal(vv1.detach().cpu(), vv2.detach().cpu()):
                    return False, f"state的属性torch值不同!"
            else:
                if vv1 != vv2:
                    return False, f"state的属性int值不同!"
            
    if len(s1['param_groups']) != len(s2['param_groups']):
        return False, "param_groups的长度不一致"
    
    for p1, p2 in zip(s1['param_groups'], s2['param_groups']):
        if p1.keys() != p2.keys():
            return False, "param_groups内的属性不一致"
        for key in p1.keys():
            vv1 = p1[key]
            vv2 = p2[key]
            if isinstance(vv1, torch.Tensor):
                if not torch.equal(vv1.detach().cpu(), vv2.detach().cpu()):
                    return False, f"param_groups属性的torch值不同!"
            else:
                if vv1 != vv2:
                    return False, f"param_groups属性的值不同!"
    
    return True, "optimizer.state_dict完全相同"

def _compare_p(s1, s2):        
    if s1.keys() != s2.keys():
        return False, "model的key不一致"

    for key in s1.keys():
        val1 = s1[key]
        val2 = s2[key]
        if isinstance(val1, torch.Tensor):
            if not torch.equal(val1.detach().cpu(), val2.detach().cpu()):
                return False, f"model的属性torch值不同!"
        else:
            if val1 != val2:
                return False, f"state的属性int值不同!"
    
    return True, "model.state_dict完全相同"

def _to_cpu(ele):
    if torch.is_tensor(ele):
        snapshot = ele.cpu().detach().pin_memory()
    elif isinstance(ele, dict):
        snapshot = {}
        for k, v in ele.items():
            snapshot[k] = None
            snapshot[k] = _to_cpu(v)
    elif isinstance(ele, list):
        snapshot = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
            snapshot[idx] = _to_cpu(v)
    else:
        snapshot = copy.deepcopy(ele)
    return snapshot

def copy_to_cpu(ele, snapshot):
    if torch.is_tensor(ele):
        snapshot.copy_(ele.detach(), non_blocking=True)
    elif isinstance(ele, dict):
        for k, v in ele.items():
            copy_to_cpu(v, snapshot[k])
    elif isinstance(ele, list):
        for idx, v in enumerate(ele):
            copy_to_cpu(v, snapshot[idx])

# 初始化模型、数据加载器和优化器
def main():
    # 1. 定义模型并转移到 GPU
    model = TheModelClass()
    model = model.cuda()
    
    # 2. 定义数据加载器
    train_dataset = MyDataset()
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    
    # 4. 训练循环
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        index = 0
        for inputs, labels in train_loader:
            # 将数据转移到 GPU
            inputs, labels = inputs.cuda(), labels.cuda()  # [[8]]
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, index={index}")   


    #print(type(optimizer.state_dict()['state']))
    #print(type(model.state_dict()))
    # # 打印模型的状态字典
    print("Model's named_parameters:")
    for name, param in model.named_parameters():
        print(name, "\t", param)

    # # 打印优化器的状态字典
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])



    s1 = _to_cpu(model.state_dict())
    parameter_names = {v: k for k, v in sorted(model.named_parameters())}
    memo = set()
    for module_prefix, module in model.named_modules():
        members = module._buffers.items()
        for k, v in members:
            if v is None or v in memo or k in module._non_persistent_buffers_set:
                continue
            memo.add(v)
            n = module_prefix + ('.' if module_prefix else '') + k
            copy_to_cpu(v, s1[n])
    for group in optimizer.param_groups:
        for p in group['params']:
            copy_to_cpu(p, s1[parameter_names[p]])
    s1 = OrderedDict([(k, s1[k]) for k in s1.keys()])
    result, message = _compare_p(s1 ,model.state_dict())
    print(f"比较结果: {result},{message}")

    s1 = _to_cpu(optimizer.state_dict())
    start_index = 0
    param_mappings = {}
    def pack_group(group):
        nonlocal start_index
        packed = {k: v for k, v in group.items() if k != 'params'}
        param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                                if id(p) not in param_mappings})
        packed['params'] = [param_mappings[id(p)] for p in group['params']]
        start_index += len(packed['params'])
        return packed
    s1['param_groups'] = [pack_group(g) for g in optimizer.param_groups]
    s1['state'] = {param_mappings[id(i)] : p for i, p in optimizer.state.items()}
    result, message = _compare(s1 ,optimizer.state_dict())
    print(f"比较结果: {result},{message}")
 
    # s1_index = 0
    # s1['state'] = {i : p for i, p in enumerate(s1['state'].values(), s1_index)}

    # s1_index = 0
    # for group in s1['param_groups']:
    #     group['params'] = [i for i, p in enumerate(group['params'], s1_index)]
    #     s1_index += len(group['params'])



if __name__ == "__main__":
    main()
