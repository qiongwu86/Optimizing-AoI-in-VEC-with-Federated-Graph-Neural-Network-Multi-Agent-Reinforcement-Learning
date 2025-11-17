import numpy as np
import torch
# from EnvironmentGCN_SAC import VEC_env
from EnvironmentGCN_SAC_new_global2_test import  VEC_env
# from MADDPG import DDPG_model
# from Environment import My_GCN
from GCN_DRL_SAC_FL_fix import DRL_SAC
from GCN_DRL_SAC_FL_fix import *
import matplotlib.pyplot as plt
import os
# from memory_profiler import profile
from torch.utils.tensorboard import SummaryWriter
import sys
# from memory_profiler import memory_usage
import copy
import math
import gc  # 添加垃圾回收模块
import time  # 添加时间模块
import psutil  # 添加系统监控模块
import threading  # 添加线程模块
import csv  # 用于保存CSV数据
from datetime import datetime  # 用于时间戳
from scipy import stats  # 用于计算置信区间

# 导入优化工具
from memory_optimizer import MemoryOptimizer, GradientOptimizer, ModelOptimizer
from simple_gcn_memory import SimpleGCNMemory, SimpleMemory

# 性能分析
# 1. 和周围所有车辆聚合, without GCN
# 2. RSU 异步聚合
# 3. 其他比如 DQN
# 4.

def main():
    # 初始化内存优化器（用于最终内存统计）
    memory_optimizer = MemoryOptimizer("training_memory.log")
    
    num_lane = 4 # four lane
    slotT = 0.02
    T = 20 * 1000 # 1000K =
    TotalSlot = T/slotT/100
    plot_T = 10000 # 10 K
    save_T = 10000
    TotalEpi = 1
    rsuW = 250
    gridW = 50      # 网格数量也是可以做性能分析的
    # vehGenRate = [8,8,8,8] # sec, 性能,
    vehNum = 15
    # vehGenRate = [2, 2, 2, 4]  # sec, 性能,
    # VehSpeed =  [120,100,90,60] # 性能, km/h
    # VehSpeed = [60, 90, 100, 120]  # 性能, km/h
    VehSpeed = [30, 35, 40, 50]  # 性能, km/h
    TaskGenRate = 0.2 # slot, 性能
    vehGenRate = [8, 8, 8, 8]
    TaskSize = 10 # 性能 大小----------------------------------------------------------------------------------------------------------------
    p = 0.0 # 性能
    GCN_batchsize = 64
    TransmissionRange = 100
    maxP = 20 # w
    slideW = 100
    w = 1
    k= 0.2
    penalty_factor = 0.98  # 修改：更快的penalty衰减，防止penalty累积
    powerFactor = 1
    param_noise_var = 1e-4
    critic_noise_var = 0.1
    GCN_factor = 0.3
    policy_rate = 1e-4
    critic_rate = 1e-3
    alpha_lr = 1e-4
    rateDecay = 0.99  # 修改：更快的学习率衰减，防止后期过拟合
    reward_scale=0.1
    
    # 添加penalty倍数配置变量
    # 可以设置为不同的值进行仿真对比：
    penalty_multiplier = 10  # 减少penalty影响
    # penalty_multiplier = 0.5  # 减少penalty影响
    # penalty_multiplier = 1.0  # 原始设置
    # penalty_multiplier = 2.0  # 增加penalty影响
    # penalty_multiplier = 5.0  # 大幅增加penalty影响
  
    # DRL模型隐藏层大小配置（64, 128, 256）
    DRL_hidden_size = 256  # 可以设置为 64, 128, 或 256

    DRL_model = DRL_SAC(state_dim=6,
                        action_dim=1,
                        max_action=maxP,
                        policy_rate = policy_rate, 
                        critic_rate = critic_rate, 
                        alpha_lr = alpha_lr,
                        reward_scale=reward_scale,
                        hidden_size=DRL_hidden_size) # --------------------------------------------------------------------------------------------

    "--------------------------------------节点特征数量, 取哪些特征-------------------------------------------------"
    node_feature_size = 5 # vehicle number,...
    # vehicle
    # init env
    # DDPG 和 GNN　一个　循环里训练算了
    env = VEC_env(lane = num_lane,vehGenRate = vehGenRate,
                  slotT = slotT,VehSpeed =VehSpeed,rsuW=rsuW,
                  plot=False,TaskGenRate=TaskGenRate,TaskSize =TaskSize,
                  TransmissionRange = TransmissionRange,gridW = gridW,node_feature_size = node_feature_size, out_feature = 1,
                  DRL_model=  DRL_model,GCN_batchsize = GCN_batchsize,p = p,maxP =maxP,slideW = slideW,w = w,k=k,
                  penalty_factor = penalty_factor,powerFactor=powerFactor,
                  param_noise_var = param_noise_var,
                  GCN_factor = GCN_factor,
                  critic_noise_var =critic_noise_var,
                  rateDecay = rateDecay,
                  GCN_critic = ConcatMlp,
                  reward_scale=reward_scale,
                  penalty_multiplier=penalty_multiplier  # 传递penalty倍数参数
                  )
                  
    # ==================== 模型加载配置 ====================
    # 训练好的模型路径（根据DRL_hidden_size自动调整）
    trained_model_base_path = f"/Users/wang/Documents/研究生资料/自己论文/第二篇/FinalConcise_20251103/训练结果2025101102/学习率0.9/模型大小/{DRL_hidden_size}/MyCode1_p_0.1"
    trained_DRL_model_path = os.path.join(trained_model_base_path, "SAC", "SAC_train__")

    
    # 检查模型文件是否存在
    load_trained_model = True  # 是否加载训练好的模型
    if load_trained_model:
        if not os.path.exists(trained_DRL_model_path):
            print(f"⚠️ Warning: DRL model file not found at {trained_DRL_model_path}")
            print("   Continuing without loading DRL model...")
            load_trained_model = False

    
    # 加载训练好的模型
    if load_trained_model:
        print("=" * 60)
        print("Loading trained models...")
        print(f"DRL Model Path: {trained_DRL_model_path}")

        print("=" * 60)
        
        try:
            # 加载DRL模型
            if os.path.exists(trained_DRL_model_path):
                ModelOptimizer.load_model_optimized(env.DRL_model, trained_DRL_model_path)
                print("✅ DRL model loaded successfully!")
            else:
                print("❌ DRL model file not found, skipping...")
            

                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("   Continuing with randomly initialized models...")
            import traceback
            traceback.print_exc()
        
        print("=" * 60)
    
    # ==================== 测试数据保存配置 ====================
    test_data_base_path = "/Users/wang/Documents/研究生资料/自己论文/第二篇/FinalConcise_20251103/Test/Hardware"
    if not os.path.exists(test_data_base_path):
        os.makedirs(test_data_base_path, exist_ok=True)
        print(f"Created test data directory: {test_data_base_path}")
    
    # 创建数据文件路径（根据模型大小、penalty_multiplier和gridW命名）
    file_name_suffix = f"{DRL_hidden_size}_{penalty_multiplier}_{gridW}"
    
    # 平均值和统计文件
    vehicle_data_avg_file = os.path.join(test_data_base_path, f"vehicle_data_{file_name_suffix}_averaged.csv")
    system_data_avg_file = os.path.join(test_data_base_path, f"system_data_{file_name_suffix}_averaged.csv")
    
    summary_file = os.path.join(test_data_base_path, f"test_summary_{file_name_suffix}.txt")
    
    # 如果统计文件和总结文件已存在，先删除
    for file_path in [vehicle_data_avg_file, system_data_avg_file, summary_file]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted existing file: {file_path}")
    
    # ==================== 运行10次并收集数据 ====================
    num_runs = 10  # 运行10次取平均
    
    test_start_time = time.time()  # 记录测试开始时间
    test_start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print("\n" + "=" * 60)
    print(f"Starting {num_runs} test runs to compute average values...")
    print("=" * 60)
    
    # 初始化用于追踪内存的变量
    previous_run_end_memory = test_start_memory
    
    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")
        
        # 记录本次运行开始时的内存
        run_start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"Memory at start of Run {run_idx + 1}: {run_start_memory:.2f} MB")
        if run_idx > 0:
            memory_increase = run_start_memory - previous_run_end_memory
            if memory_increase > 0:
                print(f"  ⚠️ Memory increase from previous run end: {memory_increase:.2f} MB")
            elif memory_increase < 0:
                print(f"  ✅ Memory decrease from previous run end: {abs(memory_increase):.2f} MB")
            else:
                print(f"  ✅ Memory unchanged from previous run end")
        
        # 创建本次运行的数据文件（带run_idx）
        vehicle_data_file = os.path.join(test_data_base_path, f"vehicle_data_{file_name_suffix}_run_{run_idx + 1}.csv")
        system_data_file = os.path.join(test_data_base_path, f"system_data_{file_name_suffix}_run_{run_idx + 1}.csv")
        
        # 如果文件已存在，先删除
        for file_path in [vehicle_data_file, system_data_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {file_path}")
        
        # 创建本次运行的数据文件并写入表头
        with open(vehicle_data_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Run_ID', 'Slot', 'Vehicle_ID', 'AOI', 'Power', 'Throughput', 'Decision_Time', 'Memory_MB'])
        
        with open(system_data_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Run_ID', 'Slot', 'System_AOI', 'Total_Vehicles', 'Total_Power', 'Total_Throughput', 'Slot_Time', 'Total_Memory_MB'])
        
        # 初始化本次运行的数据收集
        run_system_data = []  # 本次运行的系统数据
        run_vehicle_data = []  # 本次运行的车辆数据（按slot索引）
        
        # 可选：记录初始化后的内存使用（如果需要详细内存监控，可以取消注释）
        # memory_optimizer.log_memory_usage("After Environment Initialization and Model Loading")
        
        for episode in range(TotalEpi):
            env.reset(episode)
            env.episode  = episode
            
            # modelStart = env.DRL_model.actor.state_dict()['l1.weight'][:4][0]
            
            for simSlot in range(int(TotalSlot)):
                slot_start_time = time.time()
                
                # 定期内存清理（每1000个slot）
                if simSlot % 1000 == 0:
                    memory_optimizer.cleanup_memory()
                    print(f"Run {run_idx + 1}/{num_runs} - episode: {episode}, slot: {simSlot}, vehicle number: {len(env.Vehicle)}")
                
                # 执行环境步骤
                AOI, mean_veh_reward ,penalty,  n_veh, destroyAOI,  gnnloss, gnnCriticloss = env.step()
                
                # 记录slot时间和内存
                slot_time = time.time() - slot_start_time
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # ==================== 收集本次运行的数据 ====================
                # 初始化系统总功率和总吞吐量
                total_power = 0.0
                total_throughput = 0.0
                
                # 初始化当前slot的车辆数据字典
                slot_vehicle_data = {}
                
                # 收集每辆车的详细数据
                for veh in env.Vehicle:
                    # 计算单车AOI
                    veh_aoi = np.mean(veh.AOI_list) if veh.AOI_list else 0.0
                    
                    # 测量决策时间（测量下一次决策的时间）
                    decision_time = 0.0
                    if hasattr(veh, 'model') and hasattr(veh, 's0'):
                        try:
                            decision_start = time.time()
                            # 获取当前状态并计算action（模拟决策过程）
                            _ = veh.model.select_action(veh.s0, veh.record_id)
                            decision_time = time.time() - decision_start
                        except:
                            decision_time = 0.0
                    
                    # 获取车辆数据
                    vehicle_power = veh.action if hasattr(veh, 'action') else 0.0
                    # 吞吐量是实际发送的数据量（字节），不是传输速率
                    # Transimission_tasksize 在 Translate_task() 中设置，如果该slot成功传输任务则为 First_task.size，否则为 None
                    vehicle_throughput = 0.0
                    if hasattr(veh, 'Transimission_tasksize'):
                        if veh.Transimission_tasksize is not None:
                            vehicle_throughput = float(veh.Transimission_tasksize)
                        # 如果为 None，说明该slot没有传输任务，吞吐量为0
                    
                    # 累计系统总功率和总吞吐量（实际发送的数据量）
                    total_power += vehicle_power
                    total_throughput += vehicle_throughput
                    
                    # 获取车辆内存（如果需要单独测量）
                    veh_memory = current_memory / len(env.Vehicle) if len(env.Vehicle) > 0 else 0.0
                    
                    # 保存到当前slot的车辆数据字典（使用车辆ID作为key）
                    slot_vehicle_data[veh.record_id] = [
                        simSlot,
                        veh.record_id,
                        veh_aoi,
                        vehicle_power,
                        vehicle_throughput,
                        decision_time,
                        veh_memory
                    ]
                
                # 收集系统级数据（包含总功率和总吞吐量）
                system_row = [
                    run_idx + 1,  # Run_ID (从1开始)
                    simSlot,
                    AOI,  # 系统AOI
                    len(env.Vehicle),
                    total_power,  # 系统总功率
                    total_throughput,  # 系统总吞吐量
                    slot_time,
                    current_memory
                ]
                run_system_data.append(system_row)
                
                # 立即保存系统数据到CSV
                with open(system_data_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(system_row)
                
                # 保存当前slot的车辆数据
                run_vehicle_data.append(slot_vehicle_data)
                
                # 立即保存车辆数据到CSV
                vehicle_rows = []
                for veh_id, veh_data in slot_vehicle_data.items():
                    vehicle_row = [
                        run_idx + 1,  # Run_ID (从1开始)
                        veh_data[0],  # Slot
                        veh_data[1],  # Vehicle_ID (实际的车辆ID，不是-1)
                        veh_data[2],  # AOI
                        veh_data[3],  # Power
                        veh_data[4],  # Throughput
                        veh_data[5],  # Decision_Time
                        veh_data[6]   # Memory_MB
                    ]
                    vehicle_rows.append(vehicle_row)
                
                if vehicle_rows:
                    with open(vehicle_data_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(vehicle_rows)
        
        print(f"Run {run_idx + 1}/{num_runs} completed. Data saved to:")
        print(f"  {vehicle_data_file}")
        print(f"  {system_data_file}")
        
        # ==================== 清理本次运行的内存变量 ====================
        print(f"Cleaning up memory after Run {run_idx + 1}/{num_runs}...")
        memory_before_cleanup = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 清理本次运行的数据变量（数据已保存到文件）
        # 先清理嵌套的数据结构，确保所有嵌套的字典和列表都被清空
        try:
            # run_vehicle_data 是列表，每个元素是一个字典（slot_vehicle_data）
            if run_vehicle_data is not None:
                for slot_data_dict in run_vehicle_data:
                    if isinstance(slot_data_dict, dict):
                        slot_data_dict.clear()
                run_vehicle_data.clear()
        except Exception as e:
            print(f"  Warning: Error cleaning run_vehicle_data: {e}")
        
        try:
            # run_system_data 是列表，每个元素是一个列表（system_row）
            if run_system_data is not None:
                for system_row in run_system_data:
                    if isinstance(system_row, list):
                        system_row.clear()
                run_system_data.clear()
        except Exception as e:
            print(f"  Warning: Error cleaning run_system_data: {e}")
        
        # 然后删除变量
        try:
            del run_vehicle_data
        except NameError:
            # 变量可能已经被删除，忽略
            pass
        except Exception as e:
            print(f"  Warning: Error deleting run_vehicle_data: {e}")
        
        try:
            del run_system_data
        except NameError:
            # 变量可能已经被删除，忽略
            pass
        except Exception as e:
            print(f"  Warning: Error deleting run_system_data: {e}")
        
        # 1. 清理环境中的车辆对象（参考reset方法）
        if hasattr(env, 'Vehicle') and isinstance(env.Vehicle, list):
            for veh in list(env.Vehicle):  # 使用list()创建副本，避免迭代时修改
                try:
                    # 清理车辆相关的缓存和引用
                    if hasattr(veh, 'AOI_list') and isinstance(veh.AOI_list, (list, dict, set)):
                        veh.AOI_list.clear()
                    if hasattr(veh, 's0'):
                        try:
                            del veh.s0
                        except:
                            pass
                    # 清理车辆模型的replay buffer（如果存在）
                    if hasattr(veh, 'model'):
                        try:
                            # 检查是否是独立的模型实例还是共享的DRL_model
                            if hasattr(env, 'DRL_model') and veh.model is not env.DRL_model:
                                if hasattr(veh.model, 'memory'):
                                    # 这是独立的模型实例，可以清理其memory
                                    if hasattr(veh.model.memory, 'storage'):
                                        if isinstance(veh.model.memory.storage, (list, dict, set)):
                                            veh.model.memory.storage.clear()
                                    try:
                                        del veh.model.memory
                                    except:
                                        pass
                        except Exception as e:
                            pass  # 忽略模型清理错误
                    # 注意：不删除veh.model本身，因为下一轮会重新创建车辆
                except Exception as e:
                    veh_id = veh.record_id if hasattr(veh, 'record_id') else 'unknown'
                    print(f"  Warning: Error cleaning vehicle {veh_id}: {e}")
            # 清理车辆列表（参考reset方法）
            try:
                env.Vehicle.clear()
            except:
                pass
        
        # 2. 清理环境中的图结构和缓存（安全地检查类型）
        # 辅助函数：安全清理可清理对象
        def safe_clear(obj):
            """安全地清理对象，支持列表、字典、集合等"""
            if isinstance(obj, (list, dict, set)):
                obj.clear()
            elif hasattr(obj, 'clear'):
                try:
                    obj.clear()
                except:
                    pass
        
        if hasattr(env, 'v2vlink'):
            safe_clear(env.v2vlink)
        
        if hasattr(env, 'uploadveh_channel'):
            safe_clear(env.uploadveh_channel)
        
        if hasattr(env, 'destroyAOI'):
            safe_clear(env.destroyAOI)
        
        # 清理PyTorch张量（释放GPU/CPU内存）
        pytorch_tensors = ['node', 'last_node', 'edge', 'GCN_data', 'last_GCN_data', 'last_G_conv']
        for attr_name in pytorch_tensors:
            if hasattr(env, attr_name):
                try:
                    attr = getattr(env, attr_name)
                    # 检查是否是PyTorch张量或Data对象
                    if isinstance(attr, (torch.Tensor, torch.nn.utils.rnn.PackedSequence)):
                        delattr(env, attr_name)
                    elif hasattr(attr, '__class__') and 'torch' in str(type(attr)):
                        # 其他PyTorch对象（如Data对象）
                        try:
                            delattr(env, attr_name)
                        except:
                            setattr(env, attr_name, None)
                except Exception as e:
                    print(f"  Warning: Error cleaning {attr_name}: {e}")
        
        # 清理其他缓存（设置为None或空值）
        if hasattr(env, 'train_mask'):
            try:
                env.train_mask = None
            except:
                pass
        
        # 3. 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        
        # 4. 清理DRL_model的缓存（如果存在）
        if hasattr(env, 'DRL_model') and env.DRL_model is not None:
            try:
                # 清理DRL_model的replay buffer（如果存在）
                if hasattr(env.DRL_model, 'memory'):
                    if hasattr(env.DRL_model.memory, 'storage'):
                        if isinstance(env.DRL_model.memory.storage, (list, dict, set)):
                            env.DRL_model.memory.storage.clear()
                    # 注意：不删除memory，因为模型需要它
                
                # 清理模型的计算图缓存
                if hasattr(env.DRL_model, 'actor'):
                    env.DRL_model.actor.zero_grad(set_to_none=True)
                if hasattr(env.DRL_model, 'critic'):
                    if hasattr(env.DRL_model.critic, 'q1'):
                        env.DRL_model.critic.q1.zero_grad(set_to_none=True)
                    if hasattr(env.DRL_model.critic, 'q2'):
                        env.DRL_model.critic.q2.zero_grad(set_to_none=True)
            except Exception as e:
                print(f"  Warning: Error cleaning DRL_model cache: {e}")
        
        # 5. 清理内存优化器的缓存（但保留基本功能）
        memory_optimizer.cleanup_memory()
        
        # 6. 强制垃圾回收（多次调用以确保彻底清理）
        # 收集0代、1代、2代的垃圾
        collected = gc.collect()
        collected += gc.collect()  # 第二次调用以清理循环引用
        collected += gc.collect(2)  # 强制收集老一代的垃圾
        
        # 7. 再次清理PyTorch缓存（在垃圾回收后）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 8. 清理文件系统缓冲区（通过强制刷新）
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        # 9. 记录清理后的内存使用
        memory_after_cleanup = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_freed = memory_before_cleanup - memory_after_cleanup
        print(f"  Memory before cleanup: {memory_before_cleanup:.2f} MB")
        print(f"  Memory after cleanup: {memory_after_cleanup:.2f} MB")
        print(f"  Memory freed: {memory_freed:.2f} MB")
        
        # 记录本次运行的内存变化
        if run_idx == 0:
            run_memory_increase = memory_after_cleanup - run_start_memory
        else:
            run_memory_increase = memory_after_cleanup - run_start_memory
        print(f"  Net memory change for Run {run_idx + 1}: {run_memory_increase:.2f} MB")
        
        # 保存本次运行结束时的内存，用于下次比较
        previous_run_end_memory = memory_after_cleanup
        
        print(f"Memory cleanup completed for Run {run_idx + 1}/{num_runs}.")
    
    # ==================== 计算平均值、标准差和置信区间 ====================
    print("\n" + "=" * 60)
    print("Computing statistics (mean, std, confidence intervals) across all runs...")
    print("Reading data from individual run files...")
    print("=" * 60)
    
    # 用于计算置信区间的t值（95%置信区间，自由度=num_runs-1）
    confidence_level = 0.95
    alpha = 1 - confidence_level
    
    # ==================== 从所有运行文件中读取系统数据 ====================
    system_data_by_slot = {}  # {slot: [{'run_id': ..., 'AOI': ..., ...}, ...]}
    
    for run_idx in range(num_runs):
        system_data_file = os.path.join(test_data_base_path, f"system_data_{file_name_suffix}_run_{run_idx + 1}.csv")
        if not os.path.exists(system_data_file):
            print(f"Warning: System data file not found: {system_data_file}")
            continue
        
        with open(system_data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slot = int(row['Slot'])
                if slot not in system_data_by_slot:
                    system_data_by_slot[slot] = []
                system_data_by_slot[slot].append({
                    'run_id': int(row['Run_ID']),
                    'AOI': float(row['System_AOI']),
                    'n_veh': float(row['Total_Vehicles']),
                    'total_power': float(row['Total_Power']),
                    'total_throughput': float(row['Total_Throughput']),
                    'slot_time': float(row['Slot_Time']),
                    'memory': float(row['Total_Memory_MB'])
                })
    
    # 计算系统数据的统计信息
    num_slots = max(system_data_by_slot.keys()) + 1 if system_data_by_slot else 0
    averaged_system_data = []
    
    for slot_idx in range(num_slots):
        if slot_idx not in system_data_by_slot:
            continue
        
        slot_data_list = system_data_by_slot[slot_idx]
        slot_values = {
            'AOI': [d['AOI'] for d in slot_data_list],
            'n_veh': [d['n_veh'] for d in slot_data_list],
            'total_power': [d['total_power'] for d in slot_data_list],
            'total_throughput': [d['total_throughput'] for d in slot_data_list],
            'slot_time': [d['slot_time'] for d in slot_data_list],
            'memory': [d['memory'] for d in slot_data_list]
        }
        
        # 计算统计信息
        n = len(slot_values['AOI'])
        if n > 0:
            # 计算均值和标准差
            mean_aoi = np.mean(slot_values['AOI'])
            std_aoi = np.std(slot_values['AOI'], ddof=1) if n > 1 else 0.0
            
            mean_n_veh = np.mean(slot_values['n_veh'])
            std_n_veh = np.std(slot_values['n_veh'], ddof=1) if n > 1 else 0.0
            
            mean_power = np.mean(slot_values['total_power'])
            std_power = np.std(slot_values['total_power'], ddof=1) if n > 1 else 0.0
            
            mean_throughput = np.mean(slot_values['total_throughput'])
            std_throughput = np.std(slot_values['total_throughput'], ddof=1) if n > 1 else 0.0
            
            mean_time = np.mean(slot_values['slot_time'])
            std_time = np.std(slot_values['slot_time'], ddof=1) if n > 1 else 0.0
            
            mean_memory = np.mean(slot_values['memory'])
            std_memory = np.std(slot_values['memory'], ddof=1) if n > 1 else 0.0
            
            # 计算95%置信区间
            if n > 1:
                t_value = stats.t.ppf(1 - alpha/2, n - 1)
                ci_aoi = t_value * std_aoi / np.sqrt(n)
                ci_n_veh = t_value * std_n_veh / np.sqrt(n)
                ci_power = t_value * std_power / np.sqrt(n)
                ci_throughput = t_value * std_throughput / np.sqrt(n)
                ci_time = t_value * std_time / np.sqrt(n)
                ci_memory = t_value * std_memory / np.sqrt(n)
            else:
                ci_aoi = ci_n_veh = ci_power = ci_throughput = ci_time = ci_memory = 0.0
            
            # 保存统计信息：[Slot, Mean, Std, CI_Lower, CI_Upper] 对于每个指标
            # 为了简化，我们保存为多行，每行一个指标
            averaged_system_data.append({
                'slot': slot_idx,
                'AOI': {'mean': mean_aoi, 'std': std_aoi, 'ci_lower': mean_aoi - ci_aoi, 'ci_upper': mean_aoi + ci_aoi},
                'n_veh': {'mean': mean_n_veh, 'std': std_n_veh, 'ci_lower': mean_n_veh - ci_n_veh, 'ci_upper': mean_n_veh + ci_n_veh},
                'total_power': {'mean': mean_power, 'std': std_power, 'ci_lower': mean_power - ci_power, 'ci_upper': mean_power + ci_power},
                'total_throughput': {'mean': mean_throughput, 'std': std_throughput, 'ci_lower': mean_throughput - ci_throughput, 'ci_upper': mean_throughput + ci_throughput},
                'slot_time': {'mean': mean_time, 'std': std_time, 'ci_lower': mean_time - ci_time, 'ci_upper': mean_time + ci_time},
                'memory': {'mean': mean_memory, 'std': std_memory, 'ci_lower': mean_memory - ci_memory, 'ci_upper': mean_memory + ci_memory}
            })
    
    # ==================== 从所有运行文件中读取车辆数据 ====================
    vehicle_data_by_slot = {}  # {slot: [{'run_id': ..., 'AOI': ..., ...}, ...]}
    
    for run_idx in range(num_runs):
        vehicle_data_file = os.path.join(test_data_base_path, f"vehicle_data_{file_name_suffix}_run_{run_idx + 1}.csv")
        if not os.path.exists(vehicle_data_file):
            print(f"Warning: Vehicle data file not found: {vehicle_data_file}")
            continue
        
        with open(vehicle_data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slot = int(row['Slot'])
                if slot not in vehicle_data_by_slot:
                    vehicle_data_by_slot[slot] = []
                vehicle_data_by_slot[slot].append({
                    'run_id': int(row['Run_ID']),
                    'AOI': float(row['AOI']),
                    'Power': float(row['Power']),
                    'Throughput': float(row['Throughput']),
                    'Decision_Time': float(row['Decision_Time']),
                    'Memory': float(row['Memory_MB'])
                })
    
    # 计算车辆数据的统计信息
    # 对于车辆数据，我们按slot统计所有车辆的平均指标
    averaged_vehicle_data = []
    
    for slot_idx in range(num_slots):
        if slot_idx not in vehicle_data_by_slot:
            continue
        
        # 收集所有运行中该slot的所有车辆数据
        slot_data_list = vehicle_data_by_slot[slot_idx]
        slot_aoi_list = [d['AOI'] for d in slot_data_list]
        slot_power_list = [d['Power'] for d in slot_data_list]
        slot_throughput_list = [d['Throughput'] for d in slot_data_list]
        slot_decision_time_list = [d['Decision_Time'] for d in slot_data_list]
        slot_memory_list = [d['Memory'] for d in slot_data_list]
        
        # 计算该slot的平均车辆指标和统计信息
        if len(slot_aoi_list) > 0:
            n = len(slot_aoi_list)
            
            # 计算均值和标准差
            mean_aoi = np.mean(slot_aoi_list)
            std_aoi = np.std(slot_aoi_list, ddof=1) if n > 1 else 0.0
            
            mean_power = np.mean(slot_power_list)
            std_power = np.std(slot_power_list, ddof=1) if n > 1 else 0.0
            
            mean_throughput = np.mean(slot_throughput_list)
            std_throughput = np.std(slot_throughput_list, ddof=1) if n > 1 else 0.0
            
            mean_decision_time = np.mean(slot_decision_time_list)
            std_decision_time = np.std(slot_decision_time_list, ddof=1) if n > 1 else 0.0
            
            mean_memory = np.mean(slot_memory_list)
            std_memory = np.std(slot_memory_list, ddof=1) if n > 1 else 0.0
            
            # 计算95%置信区间
            if n > 1:
                t_value = stats.t.ppf(1 - alpha/2, n - 1)
                ci_aoi = t_value * std_aoi / np.sqrt(n)
                ci_power = t_value * std_power / np.sqrt(n)
                ci_throughput = t_value * std_throughput / np.sqrt(n)
                ci_decision_time = t_value * std_decision_time / np.sqrt(n)
                ci_memory = t_value * std_memory / np.sqrt(n)
            else:
                ci_aoi = ci_power = ci_throughput = ci_decision_time = ci_memory = 0.0
            
            averaged_vehicle_data.append({
                'slot': slot_idx,
                'AOI': {'mean': mean_aoi, 'std': std_aoi, 'ci_lower': mean_aoi - ci_aoi, 'ci_upper': mean_aoi + ci_aoi},
                'Power': {'mean': mean_power, 'std': std_power, 'ci_lower': mean_power - ci_power, 'ci_upper': mean_power + ci_power},
                'Throughput': {'mean': mean_throughput, 'std': std_throughput, 'ci_lower': mean_throughput - ci_throughput, 'ci_upper': mean_throughput + ci_throughput},
                'Decision_Time': {'mean': mean_decision_time, 'std': std_decision_time, 'ci_lower': mean_decision_time - ci_decision_time, 'ci_upper': mean_decision_time + ci_decision_time},
                'Memory': {'mean': mean_memory, 'std': std_memory, 'ci_lower': mean_memory - ci_memory, 'ci_upper': mean_memory + ci_memory}
            })
    
    # ==================== 保存平均值和统计信息 ====================
    # 写入系统数据统计CSV（包含均值、标准差和置信区间）
    with open(system_data_avg_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Slot', 'Metric', 'Mean', 'Std', 'CI_Lower', 'CI_Upper'])
        for slot_data in averaged_system_data:
            slot = slot_data['slot']
            for metric_name, metric_stats in slot_data.items():
                if metric_name != 'slot':
                    writer.writerow([
                        slot,
                        metric_name,
                        metric_stats['mean'],
                        metric_stats['std'],
                        metric_stats['ci_lower'],
                        metric_stats['ci_upper']
                    ])
    
    # 写入车辆数据统计CSV（包含均值、标准差和置信区间）
    with open(vehicle_data_avg_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Slot', 'Metric', 'Mean', 'Std', 'CI_Lower', 'CI_Upper'])
        for slot_data in averaged_vehicle_data:
            slot = slot_data['slot']
            for metric_name, metric_stats in slot_data.items():
                if metric_name != 'slot':
                    writer.writerow([
                        slot,
                        metric_name,
                        metric_stats['mean'],
                        metric_stats['std'],
                        metric_stats['ci_lower'],
                        metric_stats['ci_upper']
                    ])
    
    # ==================== 清理从文件读取的数据结构（统计信息已保存） ====================
    print("\nCleaning up data structures after statistics computation...")
    memory_before_cleanup_stats = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    peak_memory = memory_before_cleanup_stats  # 记录峰值内存（在清理前）
    
    # 清理从文件读取的数据结构（统计信息已保存到CSV文件）
    del system_data_by_slot
    del vehicle_data_by_slot
    system_data_by_slot = None
    vehicle_data_by_slot = None
    
    # 强制垃圾回收
    gc.collect()
    gc.collect()
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    memory_after_cleanup_stats = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    memory_freed_stats = memory_before_cleanup_stats - memory_after_cleanup_stats
    print(f"  Memory before cleanup: {memory_before_cleanup_stats:.2f} MB")
    print(f"  Memory after cleanup: {memory_after_cleanup_stats:.2f} MB")
    print(f"  Memory freed: {memory_freed_stats:.2f} MB")
    
    # ==================== 保存最终测试总结 ====================
    test_end_time = time.time()
    test_total_time = test_end_time - test_start_time
    
    # 使用清理后的内存作为最终内存（更准确）
    test_end_memory = memory_after_cleanup_stats
    test_total_memory = test_end_memory - test_start_memory
    memory_summary = memory_optimizer.get_memory_summary()
    
    # 计算一些统计信息（从averaged_system_data字典中提取，包括置信区间）
    if averaged_system_data:
        # 计算平均值
        avg_system_aoi = np.mean([slot_data['AOI']['mean'] for slot_data in averaged_system_data])
        std_system_aoi = np.mean([slot_data['AOI']['std'] for slot_data in averaged_system_data])
        avg_total_vehicles = np.mean([slot_data['n_veh']['mean'] for slot_data in averaged_system_data])
        std_total_vehicles = np.mean([slot_data['n_veh']['std'] for slot_data in averaged_system_data])
        avg_total_power = np.mean([slot_data['total_power']['mean'] for slot_data in averaged_system_data])
        std_total_power = np.mean([slot_data['total_power']['std'] for slot_data in averaged_system_data])
        avg_total_throughput = np.mean([slot_data['total_throughput']['mean'] for slot_data in averaged_system_data])
        std_total_throughput = np.mean([slot_data['total_throughput']['std'] for slot_data in averaged_system_data])
        
        # 计算置信区间（使用每个slot的CI上下界的平均值）
        avg_aoi_ci_lower = np.mean([slot_data['AOI']['ci_lower'] for slot_data in averaged_system_data])
        avg_aoi_ci_upper = np.mean([slot_data['AOI']['ci_upper'] for slot_data in averaged_system_data])
        avg_vehicles_ci_lower = np.mean([slot_data['n_veh']['ci_lower'] for slot_data in averaged_system_data])
        avg_vehicles_ci_upper = np.mean([slot_data['n_veh']['ci_upper'] for slot_data in averaged_system_data])
        avg_power_ci_lower = np.mean([slot_data['total_power']['ci_lower'] for slot_data in averaged_system_data])
        avg_power_ci_upper = np.mean([slot_data['total_power']['ci_upper'] for slot_data in averaged_system_data])
        avg_throughput_ci_lower = np.mean([slot_data['total_throughput']['ci_lower'] for slot_data in averaged_system_data])
        avg_throughput_ci_upper = np.mean([slot_data['total_throughput']['ci_upper'] for slot_data in averaged_system_data])
    else:
        avg_system_aoi = std_system_aoi = avg_total_vehicles = std_total_vehicles = avg_total_power = std_total_power = avg_total_throughput = std_total_throughput = 0.0
        avg_aoi_ci_lower = avg_aoi_ci_upper = avg_vehicles_ci_lower = avg_vehicles_ci_upper = avg_power_ci_lower = avg_power_ci_upper = avg_throughput_ci_lower = avg_throughput_ci_upper = 0.0
    
    # 保存测试总结到文件
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Test Summary Report (Statistics from 10 Runs)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Number of Runs: {num_runs}\n")
        f.write(f"Confidence Level: {confidence_level*100:.1f}%\n")
        f.write(f"Test Start Time: {datetime.fromtimestamp(test_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test End Time: {datetime.fromtimestamp(test_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Test Time: {test_total_time:.2f} seconds ({test_total_time/60:.2f} minutes)\n")
        f.write(f"Average Time per Run: {test_total_time/num_runs:.2f} seconds ({test_total_time/num_runs/60:.2f} minutes)\n")
        f.write(f"\n")
        f.write(f"Memory Statistics:\n")
        f.write(f"  Start Memory: {test_start_memory:.2f} MB\n")
        f.write(f"  Peak Memory Usage (before cleanup): {peak_memory:.2f} MB\n")
        f.write(f"  End Memory (after cleanup): {test_end_memory:.2f} MB\n")
        if 'memory_freed_stats' in locals():
            f.write(f"  Memory Freed by Cleanup: {memory_freed_stats:.2f} MB\n")
        f.write(f"  Total Memory Increase: {test_total_memory:.2f} MB\n")
        f.write(f"\n")
        f.write(f"Average Performance Metrics (across all slots and runs):\n")
        f.write(f"  Average System AOI: {avg_system_aoi:.4f} ± {std_system_aoi:.4f}\n")
        f.write(f"    95% Confidence Interval: [{avg_aoi_ci_lower:.4f}, {avg_aoi_ci_upper:.4f}]\n")
        f.write(f"  Average Total Vehicles: {avg_total_vehicles:.2f} ± {std_total_vehicles:.2f}\n")
        f.write(f"    95% Confidence Interval: [{avg_vehicles_ci_lower:.2f}, {avg_vehicles_ci_upper:.2f}]\n")
        f.write(f"  Average Total Power: {avg_total_power:.4f} ± {std_total_power:.4f} W\n")
        f.write(f"    95% Confidence Interval: [{avg_power_ci_lower:.4f}, {avg_power_ci_upper:.4f}] W\n")
        f.write(f"  Average Total Throughput: {avg_total_throughput:.4f} ± {std_total_throughput:.4f} bytes\n")
        f.write(f"    95% Confidence Interval: [{avg_throughput_ci_lower:.4f}, {avg_throughput_ci_upper:.4f}] bytes\n")
        f.write(f"\n")
        f.write(f"Data Files:\n")
        f.write(f"  Individual Run Files (separate file for each run):\n")
        for run_idx in range(num_runs):
            vehicle_file = os.path.join(test_data_base_path, f"vehicle_data_{file_name_suffix}_run_{run_idx + 1}.csv")
            system_file = os.path.join(test_data_base_path, f"system_data_{file_name_suffix}_run_{run_idx + 1}.csv")
            f.write(f"    Run {run_idx + 1}: {vehicle_file}, {system_file}\n")
        f.write(f"  Vehicle Data Statistics (Mean±Std, CI): {vehicle_data_avg_file}\n")
        f.write(f"  System Data Statistics (Mean±Std, CI): {system_data_avg_file}\n")
        f.write(f"  Summary File: {summary_file}\n")
        f.write(f"\n")
        f.write(f"Note: The statistics files contain mean, standard deviation, and {confidence_level*100:.0f}% confidence intervals for each metric at each slot.\n")
        f.write(f"      Individual run files contain data for each run separately for uncertainty quantification.\n")
        f.write(f"\n")
        f.write(f"Model Path:\n")
        f.write(f"  DRL Model: {trained_DRL_model_path}\n")
        f.write("=" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print("Test Summary (Statistics from 10 Runs)")
    print("=" * 60)
    print(f"Number of Runs: {num_runs}")
    print(f"Confidence Level: {confidence_level*100:.1f}%")
    print(f"Total Test Time: {test_total_time:.2f} seconds ({test_total_time/60:.2f} minutes)")
    print(f"Average Time per Run: {test_total_time/num_runs:.2f} seconds ({test_total_time/num_runs/60:.2f} minutes)")
    print(f"\nMemory Statistics:")
    print(f"  Start Memory: {test_start_memory:.2f} MB")
    print(f"  Peak Memory Usage (before cleanup): {peak_memory:.2f} MB")
    print(f"  End Memory (after cleanup): {test_end_memory:.2f} MB")
    if 'memory_freed_stats' in locals():
        print(f"  Memory Freed by Cleanup: {memory_freed_stats:.2f} MB")
    print(f"  Total Memory Increase: {test_total_memory:.2f} MB")
    print(f"\nAverage Performance Metrics:")
    print(f"  Average System AOI: {avg_system_aoi:.4f} ± {std_system_aoi:.4f}")
    print(f"    95% Confidence Interval: [{avg_aoi_ci_lower:.4f}, {avg_aoi_ci_upper:.4f}]")
    print(f"  Average Total Vehicles: {avg_total_vehicles:.2f} ± {std_total_vehicles:.2f}")
    print(f"    95% Confidence Interval: [{avg_vehicles_ci_lower:.2f}, {avg_vehicles_ci_upper:.2f}]")
    print(f"  Average Total Power: {avg_total_power:.4f} ± {std_total_power:.4f} W")
    print(f"    95% Confidence Interval: [{avg_power_ci_lower:.4f}, {avg_power_ci_upper:.4f}] W")
    print(f"  Average Total Throughput: {avg_total_throughput:.4f} ± {std_total_throughput:.4f} bytes")
    print(f"    95% Confidence Interval: [{avg_throughput_ci_lower:.4f}, {avg_throughput_ci_upper:.4f}] bytes")
    print(f"\nData saved to:")
    print(f"  Individual Run Files:")
    for run_idx in range(num_runs):
        vehicle_file = os.path.join(test_data_base_path, f"vehicle_data_{file_name_suffix}_run_{run_idx + 1}.csv")
        system_file = os.path.join(test_data_base_path, f"system_data_{file_name_suffix}_run_{run_idx + 1}.csv")
        print(f"    Run {run_idx + 1}: {vehicle_file}")
        print(f"              {system_file}")
    print(f"  Vehicle Data Statistics: {vehicle_data_avg_file}")
    print(f"  System Data Statistics: {system_data_avg_file}")
    print(f"  Summary File: {summary_file}")
    print("=" * 60)
    
    print("Test completed.")

if __name__ == "__main__":
    # 到时候也写成 arg
    main()
