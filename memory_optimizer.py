import torch
import gc
import psutil
import os
import sys
from contextlib import contextmanager
import time

class MemoryOptimizer:
    """
    内存优化工具类，用于监控和优化内存使用
    """
    
    def __init__(self, log_file="memory_optimizer.log"):
        self.log_file = log_file
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.cleanup_threshold = 0.8  # 内存使用超过80%时触发清理
        
    def get_memory_usage(self):
        """获取当前内存使用情况"""
        process = psutil.Process(os.getpid())       # 获取当前Python进程的ID, psutil.Process() 创建一个Process对象来监控这个特定进程
        memory_info = process.memory_info()
        return {       # 返回内存使用情况
            'rss': memory_info.rss / 1024 / 1024,  # MB 实际使用的物理内存（常驻内存）
            'vms': memory_info.vms / 1024 / 1024,  # MB 虚拟内存使用量
            'percent': process.memory_percent()    # 返回内存使用百分比
        }
    
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                'cached': torch.cuda.memory_reserved() / 1024 / 1024,  # MB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024 / 1024,  # MB
            }
        return None
    
    def log_memory_usage(self, context=""):
        """记录内存使用情况"""
        cpu_memory = self.get_memory_usage()
        gpu_memory = self.get_gpu_memory_usage()
        
        # 更新峰值内存
        if cpu_memory['rss'] > self.peak_memory['rss']:
            self.peak_memory = cpu_memory.copy()
        
        log_message = f"[{context}] CPU Memory: {cpu_memory['rss']:.2f}MB ({cpu_memory['percent']:.1f}%), " # 记录CPU内存使用情况
        if gpu_memory:
            log_message += f"GPU Memory: {gpu_memory['allocated']:.2f}MB allocated, {gpu_memory['cached']:.2f}MB cached"
        else:
            log_message += "GPU: Not available" # 如果GPU不可用，记录GPU不可用
        
        print(log_message)
        
        # 写入日志文件
        with open(self.log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {log_message}\n") # 将内存使用情况写入日志文件
    
    def cleanup_memory(self, force=False):
        """清理内存"""
        current_memory = self.get_memory_usage() # 获取当前内存使用情况
        
        if force or current_memory['percent'] > self.cleanup_threshold * 100:
            print(f"Cleaning up memory... Current usage: {current_memory['percent']:.1f}%")
            
            # Python垃圾回收
            collected = gc.collect() # 执行垃圾回收，返回清理的对象数量; 只会清理没有任何引用（reference）指向它的对象。
            print(f"Garbage collected {collected} objects") # 打印清理的对象数量
            
            # GPU内存清理
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache() # 清空GPU缓存 torch.cuda.empty_cache() 只是清理 PyTorch 的 GPU 内存缓存池（cache pool）。
                print("GPU cache cleared") # 打印GPU缓存已清空
            
            # 记录清理后的内存使用
            after_memory = self.get_memory_usage() # 获取清理后的内存使用情况
            print(f"Memory after cleanup: {after_memory['rss']:.2f}MB ({after_memory['percent']:.1f}%)") # 打印清理后的内存使用情况
            
            return True
        return False
    
    @contextmanager
    def memory_context(self, context_name="", cleanup_after=True):
        """内存监控上下文管理器"""
        self.log_memory_usage(f"Before {context_name}")
        
        try:
            yield self # 暂停函数执行，把控制权交还给 with 块中的用户代码 ; self 被传回去，方便用户在 with 块内访问这个对象；
        finally:       # finally 保证无论代码是否抛出异常，都会执行；
            if cleanup_after: # 如果 cleanup_after=True，则执行 cleanup_memory() 来释放缓存或垃圾回收；
                self.cleanup_memory() # 释放缓存或垃圾回收；    finally 块中执行 cleanup_memory() 来释放缓存或垃圾回收；
            self.log_memory_usage(f"After {context_name}") # 记录内存使用情况；
    
    def monitor_training_step(self, step_func, *args, **kwargs):
        """监控训练步骤的内存使用"""
        with self.memory_context("training_step"):
            return step_func(*args, **kwargs)
    
    def get_memory_summary(self):
        """获取内存使用摘要"""
        current = self.get_memory_usage()
        gpu = self.get_gpu_memory_usage()
        
        summary = {
            'initial_memory_mb': self.initial_memory['rss'],
            'current_memory_mb': current['rss'],
            'peak_memory_mb': self.peak_memory['rss'],
            'memory_increase_mb': current['rss'] - self.initial_memory['rss'],
            'memory_percent': current['percent']
        }
        
        if gpu:
            summary.update({
                'gpu_allocated_mb': gpu['allocated'],
                'gpu_cached_mb': gpu['cached'],
                'gpu_max_allocated_mb': gpu['max_allocated']
            })
        
        return summary


class GradientOptimizer:
    """
    梯度优化工具类，解决梯度累积和计算图保留问题
    """
    
    @staticmethod
    def detach_tensors(*tensors):
        """分离tensor的计算图"""
        return [t.detach() if isinstance(t, torch.Tensor) else t for t in tensors]
    
    @staticmethod
    def clear_gradients(model):
        """清理模型梯度"""
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
    
    @staticmethod
    def zero_gradients(model):
        """清零模型梯度"""
        if hasattr(model, 'zero_grad'):
            model.zero_grad()
        elif hasattr(model, 'parameters'):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
    
    @staticmethod
    def clip_gradients(model, max_norm=1.0):
        """梯度裁剪"""
        if hasattr(model, 'parameters'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def cleanup_computation_graph(*tensors):
        """清理计算图"""
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                tensor.detach_()
                del tensor
        gc.collect()


class ModelOptimizer:
    """
    模型优化工具类
    """
    
    @staticmethod
    def move_to_cpu(model):
        """将模型移动到CPU"""
        if isinstance(model, torch.nn.Module):
            model.cpu()
        return model
    
    @staticmethod
    def move_to_gpu(model, device=None):
        """将模型移动到GPU"""
        if isinstance(model, torch.nn.Module):
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
        return model
    
    @staticmethod
    def save_model_optimized(model, path, optimizer=None):
        """优化保存模型"""
        # 移动到CPU以节省GPU内存
        try:
            # 尝试获取设备信息
            original_device = next(model.parameters()).device
            model.cpu()
        except (AttributeError, StopIteration):
            # 如果模型没有parameters方法或没有参数，尝试使用cpu方法
            if hasattr(model, 'cpu'):
                model.cpu()
                original_device = None
            else:
                # 如果都没有，直接保存
                original_device = None
        
        save_dict = {
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, path)
        
        # 移回原设备（如果知道原设备的话）
        if original_device is not None and hasattr(model, 'to'):
            model.to(original_device)
    
    @staticmethod
    def load_model_optimized(model, path, optimizer=None, device=None):
        """优化加载模型"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model, optimizer


# 使用示例和装饰器
def memory_monitor(func):
    """内存监控装饰器"""
    def wrapper(*args, **kwargs):
        optimizer = MemoryOptimizer()
        with optimizer.memory_context(func.__name__):
            result = func(*args, **kwargs)
        return result
    return wrapper

def gradient_cleanup(func):
    """梯度清理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        finally:
            GradientOptimizer.cleanup_computation_graph()
        return result
    return wrapper
