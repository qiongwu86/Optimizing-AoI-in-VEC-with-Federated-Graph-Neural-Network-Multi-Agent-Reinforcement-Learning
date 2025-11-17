import torch
import numpy as np
import gc
from collections import deque

class OptimizedGCNMemory:
    """
    优化版本的GCN内存管理类，解决内存泄漏问题
    """
    def __init__(self, memory_size, input_size, outputsize):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.input_size = input_size
        self.outputsize = outputsize
        
        # 使用deque而不是list，自动管理大小
        self.memory_gcn = deque(maxlen=memory_size)
        self.memory_gcn_ = deque(maxlen=memory_size)
        self.memory_r = deque(maxlen=memory_size)
        self.G_data = deque(maxlen=memory_size)
        self.G_data_ = deque(maxlen=memory_size)
        
        # 预分配tensor以减少内存碎片
        self._preallocated_tensors = []
        self._preallocate_tensors()

    def _preallocate_tensors(self):
        """预分配tensor以减少内存碎片"""
        for _ in range(min(100, self.memory_size // 10)):  # 预分配少量tensor
            self._preallocated_tensors.append(
                torch.zeros(self.input_size + self.outputsize, dtype=torch.float32)
            )

    def _get_preallocated_tensor(self):
        """获取预分配的tensor"""
        if self._preallocated_tensors:
            return self._preallocated_tensors.pop()
        else:
            return torch.zeros(self.input_size + self.outputsize, dtype=torch.float32)

    def _return_preallocated_tensor(self, tensor):
        """归还预分配的tensor"""
        if len(self._preallocated_tensors) < 100:
            tensor.zero_()  # 清零tensor
            self._preallocated_tensors.append(tensor)

    def Addremember(self, TraingcnData, r, TraingcnData_, G_data, G_data_):
        """
        添加记忆数据，优化内存使用
        """
        # 确保数据是detached的，避免计算图累积
        if isinstance(TraingcnData, torch.Tensor):
            TraingcnData = TraingcnData.detach().clone()
        if isinstance(TraingcnData_, torch.Tensor):
            TraingcnData_ = TraingcnData_.detach().clone()
        
        # 直接使用实际数据大小，不使用预分配tensor
        # 这样可以避免大小不匹配的问题
        self.memory_gcn.append(TraingcnData)
        self.memory_gcn_.append(TraingcnData_)
        self.memory_r.append(r)
        self.G_data.append(G_data)
        self.G_data_.append(G_data_)
        
        self.memory_counter += 1
        
        # 定期清理内存
        if self.memory_counter % 1000 == 0:
            self._cleanup_memory()

    def _cleanup_memory(self):
        """定期清理内存"""
        # 强制垃圾回收
        gc.collect()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 限制deque大小，防止无限增长
        if len(self.memory_gcn) > self.memory_size * 1.2:
            # 删除最旧的数据
            excess = len(self.memory_gcn) - self.memory_size
            for _ in range(excess):
                if self.memory_gcn:
                    self.memory_gcn.popleft()
                if self.memory_gcn_:
                    self.memory_gcn_.popleft()
                if self.memory_r:
                    self.memory_r.popleft()
                if self.G_data:
                    self.G_data.popleft()
                if self.G_data_:
                    self.G_data_.popleft()

    def get_batch(self, batch_size):
        """
        获取批次数据，优化内存使用
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
        
        # 清理预分配的tensor
        if hasattr(self, '_preallocated_tensors'):
            self._preallocated_tensors.clear()
            try:
                self._preallocate_tensors()
            except:
                # 如果预分配失败，忽略错误
                pass
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.memory_gcn)

    def __del__(self):
        """析构函数，确保内存被正确释放"""
        try:
            self.clear()
        except:
            # 忽略析构函数中的错误，避免程序崩溃
            pass


class OptimizedMemory:
    """
    优化版本的普通内存管理类
    """
    def __init__(self, memory_size, input_size, outputsize):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.input_size = input_size
        self.outputsize = outputsize
        
        # 使用numpy数组而不是list，更高效
        self.memory = np.zeros((memory_size, input_size + outputsize + 1 + input_size + 1), dtype=np.float32)
        
        # 使用deque管理索引
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
        self.clear()
