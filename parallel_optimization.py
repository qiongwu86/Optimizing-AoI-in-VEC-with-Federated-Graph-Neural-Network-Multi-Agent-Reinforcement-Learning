"""
多核并行化优化方案
针对 trainGCN_MDSAC_1_final_optimized.py 的训练加速
"""

import multiprocessing as mp
from multiprocessing import Pool, Manager
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import partial
import time

class ParallelVehicleProcessor:
    """车辆并行处理器"""
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        print(f"初始化并行处理器，使用 {self.num_workers} 个核心")
    
    def parallel_update_vehicle_locations(self, vehicles, slotT, rsuX, rsuY, gridW, maxRoadLen):
        """并行更新车辆位置"""
        def update_single_vehicle(veh_data):
            veh, slotT, rsuX, rsuY, gridW, maxRoadLen = veh_data
            veh.x += veh.v * slotT
            veh.dis = np.sqrt((rsuX - veh.x) ** 2 + (rsuY - veh.y) ** 2 + 10**2)
            veh.Loc = np.array([veh.x, veh.y, 0])
            veh.updateChannel()
            veh.GenerateTask()
            veh.node = int(veh.lane * int(maxRoadLen/gridW) + veh.x//gridW)
            veh.GenTaskNextSlot -= 1
            return veh
        
        # 准备数据
        vehicle_data = [(veh, slotT, rsuX, rsuY, gridW, maxRoadLen) for veh in vehicles]
        
        # 使用线程池并行处理（因为主要是数值计算）
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            updated_vehicles = list(executor.map(update_single_vehicle, vehicle_data))
        
        return updated_vehicles
    
    def parallel_compute_sinr(self, vehicles, allPowerGain, uploadpower):
        """并行计算SINR"""
        def compute_single_sinr(veh_data):
            veh, allPowerGain, uploadpower = veh_data
            veh.compute_sinr(allPowerGain, uploadpower[0])
            return veh.channel
        
        vehicle_data = [(veh, allPowerGain, uploadpower) for veh in vehicles]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            channels = list(executor.map(compute_single_sinr, vehicle_data))
        
        return channels
    
    def parallel_train_vehicles(self, vehicles):
        """并行训练车辆模型"""
        def train_single_vehicle(veh):
            if (veh.model.memory.memory_counter > veh.update_counter) and (veh.slot % veh.trainingTimeslot == 0):
                print(f"并行训练车辆 {veh.record_id}")
                veh.model.update(veh.Training_number)
                veh.need_local_aggregate = True
                veh.global_aggregate = True
                veh.trainNumber += 1
            return veh
        
        # 筛选需要训练的车辆
        vehicles_to_train = [veh for veh in vehicles 
                           if (veh.model.memory.memory_counter > veh.update_counter) and 
                              (veh.slot % veh.trainingTimeslot == 0)]
        
        if not vehicles_to_train:
            return vehicles
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            trained_vehicles = list(executor.map(train_single_vehicle, vehicles_to_train))
        
        # 更新原车辆列表
        for trained_veh in trained_vehicles:
            for i, veh in enumerate(vehicles):
                if veh.record_id == trained_veh.record_id:
                    vehicles[i] = trained_veh
                    break
        
        return vehicles

class ParallelEnvironmentOptimizer:
    """环境并行优化器"""
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        self.vehicle_processor = ParallelVehicleProcessor(num_workers)
    
    def optimized_updateVehLoc(self, env):
        """优化的车辆位置更新"""
        return self.vehicle_processor.parallel_update_vehicle_locations(
            env.Vehicle, env.slotT, env.rsuX, env.rsuY, env.gridW, env.maxRoadLen
        )
    
    def optimized_updateSINR(self, env):
        """优化的SINR更新"""
        allPowerGain = 0
        Channel_list = np.zeros((len(env.Vehicle)))
        
        # 并行计算所有车辆的动作和功率增益
        def compute_vehicle_action(veh):
            veh.Translate = False
            _, action_pre, action = veh.model.select_action(veh.s0, veh.record_id)
            
            if np.isnan(action):
                print(f"警告: 车辆 {veh.record_id} 的动作为 NaN")
                action = 0.0
            
            veh.action_pre = action_pre.item()
            veh.action = action.item()
            return veh.action * veh.channel
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            power_gains = list(executor.map(compute_vehicle_action, env.Vehicle))
        
        allPowerGain = sum(power_gains)
        uploadpower = env.getuploadPower(allPowerGain, env.uploadveh)
        
        # 并行计算SINR
        channels = self.vehicle_processor.parallel_compute_sinr(env.Vehicle, allPowerGain, uploadpower)
        
        # 更新车辆状态
        for i, veh in enumerate(env.Vehicle):
            Channel_list[i] = channels[i]
            veh.env_AOI = env.reward + env.penalty
        
        return Channel_list
    
    def optimized_Trainlocalmodel(self, env):
        """优化的本地模型训练"""
        return self.vehicle_processor.parallel_train_vehicles(env.Vehicle)

class BatchTrainingOptimizer:
    """批量训练优化器"""
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
    
    def batch_process_vehicles(self, vehicles, process_func, *args):
        """批量处理车辆"""
        results = []
        for i in range(0, len(vehicles), self.batch_size):
            batch = vehicles[i:i + self.batch_size]
            batch_results = [process_func(veh, *args) for veh in batch]
            results.extend(batch_results)
        return results

class MemoryOptimizedTraining:
    """内存优化的训练循环"""
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        self.parallel_env = ParallelEnvironmentOptimizer(num_workers)
        self.batch_optimizer = BatchTrainingOptimizer()
    
    def optimized_step(self, env):
        """优化的环境步进"""
        # 1. 并行更新车辆位置
        env.Vehicle = self.parallel_env.optimized_updateVehLoc(env)
        
        # 2. 销毁超出范围的车辆（串行，因为需要修改列表）
        penalty = 0
        for veh in env.Vehicle.copy():
            if veh.x >= env.maxRoadLen:
                if veh.queue:
                    aoi_mean = np.mean(veh.AOI_list) if veh.AOI_list else 0.0
                    penalty += aoi_mean
                env.Vehicle.remove(veh)
        
        if env.Vehicle:
            env.penalty += penalty / len(env.Vehicle)
        
        # 3. 生成新车辆（串行）
        env.generateVeh()
        
        # 4. 更新图结构（串行，因为涉及全局状态）
        V2Vlink_indx = env.V2Vlink()
        env.updateGrapth(V2Vlink_indx)
        env.getOverlayG()
        
        # 5. 并行训练本地模型
        if env.slot >= 0:
            env.Vehicle = self.parallel_env.optimized_Trainlocalmodel(env)
        
        # 6. 联邦聚合（串行，因为涉及全局状态）
        env.localAsyFederated()
        env.globalAsyFederated()
        
        # 7. 并行更新SINR
        Channel_list = self.parallel_env.optimized_updateSINR(env)
        
        # 8. 更新内存和其他状态
        env.updateGCNMemory()
        
        # 9. GCN训练（串行）
        gnnloss = None
        gnnCriticloss = None
        if env.slot % 1000 == 0:
            gnnloss, gnnCriticloss = env.updateGCN()
        
        env.slot += 1
        env.GenNextSlot -= 1
        
        # 10. 更新车辆slot（并行）
        def update_vehicle_slot(veh):
            veh.slot += 1
            return veh
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            env.Vehicle = list(executor.map(update_vehicle_slot, env.Vehicle))
        
        # 11. 计算奖励
        mean_veh_reward = []
        for veh in env.Vehicle:
            mean_veh_reward.append(veh.reward)
        
        mean_veh_reward_value = np.mean(mean_veh_reward) if mean_veh_reward else 0.0
        if np.isnan(mean_veh_reward_value):
            mean_veh_reward_value = 0.0
        
        env.destroyAOI = 0
        return env.reward, mean_veh_reward_value, env.penalty, len(env.Vehicle), env.destroyAOI, gnnloss, gnnCriticloss

# 使用示例
def create_optimized_training_script():
    """创建优化后的训练脚本"""
    return """
# 在 trainGCN_MDSAC_1_final_optimized.py 中添加以下代码：

from parallel_optimization import MemoryOptimizedTraining

# 在 main() 函数中初始化优化器
def main():
    # ... 现有初始化代码 ...
    
    # 添加并行优化器
    parallel_trainer = MemoryOptimizedTraining(num_workers=mp.cpu_count())
    
    # 在训练循环中替换 env.step()
    for simSlot in range(int(TotalSlot)):
        # 替换原来的 env.step() 调用
        AOI, mean_veh_reward, penalty, n_veh, destroyAOI, gnnloss, gnnCriticloss = parallel_trainer.optimized_step(env)
        
        # ... 其余代码保持不变 ...
"""

if __name__ == "__main__":
    print("并行优化模块已准备就绪")
    print(f"检测到 {mp.cpu_count()} 个CPU核心")
