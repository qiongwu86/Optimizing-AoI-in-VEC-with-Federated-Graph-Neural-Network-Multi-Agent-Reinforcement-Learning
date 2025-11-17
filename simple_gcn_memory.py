import torch
import numpy as np
import gc
from collections import deque

class SimpleGCNMemory:
    """
    简化版本的GCN内存管理类，避免复杂的预分配机制
    """
    def __init__(self, memory_size, input_size, outputsize):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.input_size = input_size
        self.outputsize = outputsize
        
        # 使用deque自动管理大小
        self.memory_gcn = deque(maxlen=memory_size)
        self.memory_gcn_ = deque(maxlen=memory_size)
        self.memory_r = deque(maxlen=memory_size)
        self.G_data = deque(maxlen=memory_size)
        self.G_data_ = deque(maxlen=memory_size)

    def Addremember(self, TraingcnData, r, TraingcnData_, G_data, G_data_):
        """
        添加记忆数据
        """
        # 确保数据是detached的，避免计算图累积
        if isinstance(TraingcnData, torch.Tensor):
            TraingcnData = TraingcnData.detach().clone()
        if isinstance(TraingcnData_, torch.Tensor):
            TraingcnData_ = TraingcnData_.detach().clone()
        
        # 直接添加到deque（自动管理大小）
        self.memory_gcn.append(TraingcnData)
        self.memory_gcn_.append(TraingcnData_)
        self.memory_r.append(r)
        self.G_data.append(G_data)
        self.G_data_.append(G_data_)
        
        self.memory_counter += 1
        
        # 定期清理内存
        if self.memory_counter % 1000 == 0:
            gc.collect()

    def get_batch(self, batch_size):
        """
        获取批次数据
        """
        if len(self.memory_gcn) < batch_size:
            return None, None, None, None, None
        
        # 随机选择索引
        indices = np.random.choice(len(self.memory_gcn), size=batch_size, replace=False)
        
        # 收集批次数据
        batch_gcn = []
        batch_gcn_ = []
        batch_r = []
        batch_G_data = []
        batch_G_data_ = []
        
        for idx in indices:
            batch_gcn.append(self.memory_gcn[idx])
            batch_gcn_.append(self.memory_gcn_[idx])
            batch_r.append(self.memory_r[idx])
            batch_G_data.append(self.G_data[idx])
            batch_G_data_.append(self.G_data_[idx])
        
        return batch_gcn, batch_gcn_, batch_r, batch_G_data, batch_G_data_

    def clear(self):
        """清空所有内存"""
        self.memory_gcn.clear()
        self.memory_gcn_.clear()
        self.memory_r.clear()
        self.G_data.clear()
        self.G_data_.clear()
        self.memory_counter = 0
        gc.collect()

    def __len__(self):
        return len(self.memory_gcn)

    def __del__(self):
        """析构函数"""
        try:
            self.clear()
        except:
            pass


class SimpleMemory:
    """
    简化版本的普通内存管理类
    """
    def __init__(self, memory_size, input_size, outputsize):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.input_size = input_size
        self.outputsize = outputsize
        
        # 使用numpy数组
        self.memory = np.zeros((memory_size, input_size + outputsize + 1 + input_size + 1), dtype=np.float32)
        self.valid_indices = deque(maxlen=memory_size)

    def Addremember(self, TrainData):
        """
        添加训练数据
        """
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.array(TrainData, dtype=np.float32)
        self.valid_indices.append(idx)
        self.memory_counter += 1
        
        # 定期清理内存
        if self.memory_counter % 1000 == 0:
            gc.collect()

    def get_batch(self, batch_size):
        """
        获取批次数据
        """
        if len(self.valid_indices) < batch_size:
            return None
        
        # 从有效索引中随机选择
        indices = np.random.choice(list(self.valid_indices), size=batch_size, replace=False)
        return self.memory[indices]

    def clear(self):
        """清空内存"""
        self.memory.fill(0)
        self.valid_indices.clear()
        self.memory_counter = 0
        gc.collect()

    def __len__(self):
        return len(self.valid_indices)

    def __del__(self):
        """析构函数"""
        try:
            self.clear()
        except:
            pass

