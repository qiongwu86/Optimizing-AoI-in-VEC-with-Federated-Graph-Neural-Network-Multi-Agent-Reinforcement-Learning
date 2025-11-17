"""
并行化训练配置文件
用于调整多核并行化的参数
"""

import multiprocessing as mp
import os

class ParallelConfig:
    """并行化配置类"""
    
    def __init__(self):
        # CPU核心配置
        self.max_workers = mp.cpu_count()
        self.vehicle_workers = min(self.max_workers, 8)  # 车辆处理最多8个核心
        self.training_workers = min(self.max_workers, 4)  # 训练最多4个核心
        
        # 批量处理配置
        self.vehicle_batch_size = 16
        self.training_batch_size = 8
        
        # 内存优化配置
        self.memory_cleanup_interval = 5000
        self.gc_interval = 1000
        
        # 性能监控配置
        self.performance_log_interval = 1000
        self.memory_log_interval = 5000
        
        # 线程池配置
        self.thread_pool_max_workers = min(self.max_workers, 16)
        
        # 进程池配置（用于CPU密集型任务）
        self.process_pool_max_workers = min(self.max_workers, 8)
        
        # 环境变量配置
        self.set_environment_variables()
    
    def set_environment_variables(self):
        """设置环境变量以优化性能"""
        # PyTorch多线程配置
        os.environ['OMP_NUM_THREADS'] = str(self.max_workers)
        os.environ['MKL_NUM_THREADS'] = str(self.max_workers)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.max_workers)
        
        # 禁用PyTorch的自动并行化（避免冲突）
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        
        # 设置进程优先级
        if os.name == 'nt':  # Windows
            os.environ['PYTHONUNBUFFERED'] = '1'
    
    def get_optimal_workers(self, task_type="vehicle"):
        """根据任务类型获取最优的工作进程数"""
        if task_type == "vehicle":
            return self.vehicle_workers
        elif task_type == "training":
            return self.training_workers
        elif task_type == "thread":
            return self.thread_pool_max_workers
        elif task_type == "process":
            return self.process_pool_max_workers
        else:
            return self.max_workers
    
    def print_config(self):
        """打印配置信息"""
        print("=== 并行化配置 ===")
        print(f"CPU核心数: {self.max_workers}")
        print(f"车辆处理核心数: {self.vehicle_workers}")
        print(f"训练核心数: {self.training_workers}")
        print(f"线程池最大工作数: {self.thread_pool_max_workers}")
        print(f"进程池最大工作数: {self.process_pool_max_workers}")
        print(f"车辆批处理大小: {self.vehicle_batch_size}")
        print(f"训练批处理大小: {self.training_batch_size}")
        print(f"内存清理间隔: {self.memory_cleanup_interval}")
        print(f"垃圾回收间隔: {self.gc_interval}")
        print("==================")

# 全局配置实例
parallel_config = ParallelConfig()

# 性能基准测试
class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_vehicle_processing(self, vehicles, num_iterations=100):
        """基准测试车辆处理性能"""
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        # 串行处理基准
        start_time = time.time()
        for _ in range(num_iterations):
            for veh in vehicles:
                veh.x += 1  # 简单操作
        serial_time = time.time() - start_time
        
        # 并行处理基准
        def process_vehicle(veh):
            veh.x += 1
            return veh
        
        start_time = time.time()
        for _ in range(num_iterations):
            with ThreadPoolExecutor(max_workers=parallel_config.vehicle_workers) as executor:
                list(executor.map(process_vehicle, vehicles))
        parallel_time = time.time() - start_time
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 1
        
        self.benchmark_results['vehicle_processing'] = {
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'vehicles_count': len(vehicles)
        }
        
        return speedup
    
    def print_benchmark_results(self):
        """打印基准测试结果"""
        print("=== 性能基准测试结果 ===")
        for test_name, results in self.benchmark_results.items():
            print(f"{test_name}:")
            print(f"  串行时间: {results['serial_time']:.4f}s")
            print(f"  并行时间: {results['parallel_time']:.4f}s")
            print(f"  加速比: {results['speedup']:.2f}x")
            if 'vehicles_count' in results:
                print(f"  车辆数量: {results['vehicles_count']}")
        print("========================")

# 使用示例和测试
if __name__ == "__main__":
    # 打印配置
    parallel_config.print_config()
    
    # 运行基准测试
    benchmark = PerformanceBenchmark()
    
    # 创建测试车辆
    class MockVehicle:
        def __init__(self, x=0):
            self.x = x
    
    test_vehicles = [MockVehicle(i) for i in range(50)]
    
    # 运行基准测试
    speedup = benchmark.benchmark_vehicle_processing(test_vehicles, 10)
    benchmark.print_benchmark_results()
    
    print(f"车辆处理加速比: {speedup:.2f}x")
