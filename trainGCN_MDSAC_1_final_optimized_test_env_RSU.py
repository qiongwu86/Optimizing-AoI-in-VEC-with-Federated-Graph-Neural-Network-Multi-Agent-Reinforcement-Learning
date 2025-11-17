import numpy as np
import torch
# from EnvironmentGCN_SAC import VEC_env
from EnvironmentGCN_SAC_new_global2_test_env_RSU import  VEC_env
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
import csv  # 用于保存CSV数据
from scipy import stats  # 用于计算置信区间

# 性能分析
# 1. 和周围所有车辆聚合, without GCN
# 2. RSU 异步聚合
# 3. 其他比如 DQN
# 4.

def main():
    # ==================== 基础配置参数 ====================
    num_lane = 4 # four lane
    slotT = 0.02
    T = 20 * 1000 # 1000K =
    TotalSlot = T/slotT/50
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
  
    # ==================== RSU数量配置 ====================
    # 外层循环：遍历不同的RSU数量
    # 注意：当前环境文件只支持2个RSU，如果需要支持更多RSU，需要修改环境文件
    num_rsu_list = [5]
    
    # ==================== 算法配置 ====================
    # 内层循环：遍历不同的算法
    algorithms = {
        # 'GNN_No_CRITC_SAC': {
        #     'name': 'GNN_No_CRITC_SAC',
        #     'model_path': '/Users/wang/Documents/研究生资料/自己论文/第二篇/FinalConcise_20251103/训练结果2025101102/学习率0.9/对比算法/无gnn critic /MyCode1_p_0.1',
        #     'DRL_hidden_size': 64,
        #     'GCN_critic': ConcatMlp
        # },
        # 'LFSAC': {
        #     'name': 'LFSAC',
        #     'model_path': '/Users/wang/Documents/研究生资料/自己论文/第二篇/FinalConcise_20251103/训练结果2025101102/学习率0.9/对比算法/FL平均/MyCode1_p_0.1',
        #     'DRL_hidden_size': 64,
        #     'GCN_critic': ConcatMlp
        # },
        'FGNN_MASAC': {
            'name': 'FGNN_MASAC',
            'model_path': '/Users/wang/Documents/研究生资料/自己论文/第二篇/FinalConcise_20251103/训练结果2025101102/学习率0.9/模型大小/64/MyCode1_p_0.1',
            'DRL_hidden_size': 64,
            'GCN_critic': ConcatMlp
        }
    }
    
    # ==================== 外层循环：遍历RSU数量 ====================
    for num_rsu in num_rsu_list:
        print("\n" + "=" * 80)
        print(f"Starting tests for Number of RSUs = {num_rsu}")
        print("=" * 80)
        
        # ==================== 内层循环：遍历不同算法 ====================
        for alg_key, alg_config in algorithms.items():
            algorithm_name = alg_config['name']
            model_base_path = alg_config['model_path']
            alg_DRL_hidden_size = alg_config['DRL_hidden_size']
            GCN_critic_class = alg_config['GCN_critic']
            
            print("\n" + "=" * 80)
            print(f"Testing Algorithm: {algorithm_name}")
            print(f"Number of RSUs: {num_rsu}")
            print("=" * 80)
            
            # 运行单个算法的测试
            run_algorithm_test(
                num_lane=num_lane, slotT=slotT, TotalSlot=TotalSlot, TotalEpi=TotalEpi,
                rsuW=rsuW, gridW=gridW, VehSpeed=VehSpeed, TaskGenRate=TaskGenRate,
                vehGenRate=vehGenRate, TaskSize=TaskSize, p=p, GCN_batchsize=GCN_batchsize,
                TransmissionRange=TransmissionRange, maxP=maxP, slideW=slideW, w=w, k=k,
                penalty_factor=penalty_factor, powerFactor=powerFactor,
                param_noise_var=param_noise_var, critic_noise_var=critic_noise_var,
                GCN_factor=GCN_factor, policy_rate=policy_rate, critic_rate=critic_rate,
                alpha_lr=alpha_lr, rateDecay=rateDecay, reward_scale=reward_scale,
                penalty_multiplier=penalty_multiplier, DRL_hidden_size=alg_DRL_hidden_size,
                algorithm_name=algorithm_name, model_base_path=model_base_path,
                GCN_critic_class=GCN_critic_class, num_rsu=num_rsu
            )
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


def run_algorithm_test(
    num_lane, slotT, TotalSlot, TotalEpi, rsuW, gridW, VehSpeed, TaskGenRate,
    vehGenRate, TaskSize, p, GCN_batchsize, TransmissionRange, maxP, slideW, w, k,
    penalty_factor, powerFactor, param_noise_var, critic_noise_var, GCN_factor,
    policy_rate, critic_rate, alpha_lr, rateDecay, reward_scale, penalty_multiplier,
    DRL_hidden_size, algorithm_name, model_base_path, GCN_critic_class, num_rsu
):
    """运行单个算法的测试"""
    node_feature_size = 5  # vehicle number,...
    
    # ==================== 根据RSU数量创建对应数量的DRL模型 ====================
    # 所有RSU使用相同的模型大小（根据算法配置）
    DRL_models = []
    trained_DRL_model_paths = []
    
    # 为每个RSU创建DRL模型
    for i in range(num_rsu):
        DRL_model = DRL_SAC(state_dim=6,
                           action_dim=1,
                           max_action=maxP,
                           policy_rate = policy_rate, 
                           critic_rate = critic_rate, 
                           alpha_lr = alpha_lr,
                           reward_scale=reward_scale,
                           hidden_size=DRL_hidden_size)
        DRL_models.append(DRL_model)
        
        # 所有RSU使用相同的模型路径
        trained_DRL_model_path = os.path.join(model_base_path, "SAC", "SAC_train__")
        trained_DRL_model_paths.append(trained_DRL_model_path)
    
    # 初始化环境（使用DRL_models列表和num_rsu）
    # 注意：DRL_model参数用于向后兼容，实际使用DRL_models列表
    env = VEC_env(lane = num_lane,vehGenRate = vehGenRate,
                  slotT = slotT,VehSpeed =VehSpeed,rsuW=rsuW,
                  plot=False,TaskGenRate=TaskGenRate,TaskSize =TaskSize,
                  TransmissionRange = TransmissionRange,gridW = gridW,node_feature_size = node_feature_size, out_feature = 1,
                  DRL_model=DRL_models[0] if len(DRL_models) > 0 else None,  # 向后兼容参数，实际使用DRL_models列表
                  num_rsu=num_rsu,  # RSU数量
                  DRL_models=DRL_models,  # 所有RSU的模型列表（主要使用这个）
                  GCN_batchsize = GCN_batchsize,p = p,maxP =maxP,slideW = slideW,w = w,k=k,
                  penalty_factor = penalty_factor,powerFactor=powerFactor,
                  param_noise_var = param_noise_var,
                  GCN_factor = GCN_factor,
                  critic_noise_var =critic_noise_var,
                  rateDecay = rateDecay,
                  GCN_critic = GCN_critic_class,
                  reward_scale=reward_scale,
                  penalty_multiplier=penalty_multiplier
                  )

    
    # 检查模型文件是否存在
    load_trained_model = True  # 是否加载训练好的模型
    model_file_exists = False
    
    if load_trained_model:
        # 检查是否有任何模型文件存在
        for model_path in trained_DRL_model_paths:
            if os.path.exists(model_path):
                model_file_exists = True
                break
        
        if not model_file_exists:
            print(f"⚠️ Warning: No DRL model files found for any RSU")
            print("   Continuing without loading DRL models...")
            load_trained_model = False

    
    # 加载训练好的模型
    if load_trained_model:
        print("=" * 60)
        print(f"Loading trained models for {num_rsu} RSUs...")
        for i, model_path in enumerate(trained_DRL_model_paths):
            print(f"RSU{i+1} Model Path: {model_path}")
        print("=" * 60)
        
        try:
            # 加载所有RSU的DRL模型
            for i in range(num_rsu):
                model_path = trained_DRL_model_paths[i]
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location='cpu')

                    # ModelOptimizer.save_model_optimized 默认使用 'model_state_dict'
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model_state = checkpoint['model_state_dict']
                    else:
                        model_state = checkpoint

                    # DRL_SAC.state_dict() 返回 {policy, qf1, qf2, target_qf1, target_qf2}
                    if isinstance(model_state, dict) and 'policy' in model_state:
                        env.DRL_models[i].load_state_dict(model_state)
                    elif isinstance(model_state, dict) and 'actor' in model_state:
                        # 兼容旧格式（使用actor和critic键名）
                        env.DRL_models[i].policy.load_state_dict(model_state['actor'])
                        if 'critic' in model_state:
                            env.DRL_models[i].qf1.load_state_dict(model_state['critic'])
                            env.DRL_models[i].qf2.load_state_dict(model_state['critic'])
                    else:
                        try:
                            env.DRL_models[i].policy.load_state_dict(model_state)
                        except Exception:
                            print(f"⚠️ RSU{i+1} 无法识别模型格式，跳过加载...")
                            continue

                    if isinstance(checkpoint, dict) and 'alpha' in checkpoint:
                        env.DRL_models[i].alpha = checkpoint['alpha']
                    print(f"✅ RSU{i+1} DRL model loaded successfully!")
                else:
                    print(f"❌ RSU{i+1} DRL model file not found at {model_path}, skipping...")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("   Continuing with randomly initialized models...")
            import traceback
            traceback.print_exc()
        
        print("=" * 60)
    
    # ==================== 测试数据保存配置 ====================
    # 创建保存目录结构：/Test/RSU/{num_rsu}/
    test_data_base_path = f"/Users/wang/Documents/研究生资料/自己论文/第二篇/FinalConcise_20251103/Test/RSU/{num_rsu}"
    if not os.path.exists(test_data_base_path):
        os.makedirs(test_data_base_path, exist_ok=True)
        print(f"Created test data directory: {test_data_base_path}")
    
    # 创建数据文件路径（使用算法名称和RSU数量）
    file_name_suffix = f"{algorithm_name}_{penalty_multiplier}_{gridW}"
    
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
    num_runs = 1  # 运行10次取平均
    
    print("\n" + "=" * 60)
    print(f"Starting {num_runs} test runs to compute average values...")
    print("=" * 60)
    
    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")
        
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
            writer.writerow(['Run_ID', 'Slot', 'Vehicle_ID', 'RSU_ID', 'AOI', 'Power', 'Throughput'])
        
        # 动态生成系统数据CSV表头（包含所有RSU的统计）
        system_data_header = ['Run_ID', 'Slot', 'System_AOI', 'Total_Vehicles', 'Total_Power', 'Total_Throughput']
        for i in range(1, num_rsu + 1):
            system_data_header.extend([
                f'RSU{i}_AOI', f'RSU{i}_Vehicles', f'RSU{i}_Power', f'RSU{i}_Throughput'
            ])
        
        with open(system_data_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(system_data_header)
        
        # 初始化本次运行的数据收集
        run_system_data = []  # 本次运行的系统数据
        run_vehicle_data = []  # 本次运行的车辆数据（按slot索引）
        
        for episode in range(TotalEpi):
            env.reset(episode)
            env.episode  = episode
            
            # modelStart = env.DRL_model.actor.state_dict()['l1.weight'][:4][0]
            
            for simSlot in range(int(TotalSlot)):
                print(f"Run {run_idx + 1}/{num_runs} - episode: {episode}, slot: {simSlot}, vehicle number: {len(env.Vehicle)}")
                
                # 执行环境步骤
                AOI, mean_veh_reward ,penalty,  n_veh, destroyAOI,  gnnloss, gnnCriticloss = env.step()
                
                # ==================== 收集本次运行的数据 ====================
                # 初始化系统总功率和总吞吐量
                total_power = 0.0
                total_throughput = 0.0
                
                # 初始化每个RSU的统计（使用字典存储）
                rsu_stats = {}
                for i in range(1, num_rsu + 1):
                    rsu_stats[i] = {
                        'power': 0.0,
                        'throughput': 0.0,
                        'aoi_sum': 0.0,
                        'vehicle_count': 0
                    }
                
                # 初始化当前slot的车辆数据字典
                slot_vehicle_data = {}
                
                # 收集每辆车的详细数据
                for veh in env.Vehicle:
                    # 计算单车AOI
                    veh_aoi = np.mean(veh.AOI_list) if veh.AOI_list else 0.0
                    
                    # 获取车辆数据
                    vehicle_power = veh.action if hasattr(veh, 'action') else 0.0
                    # 吞吐量是实际发送的数据量（字节），不是传输速率
                    # Transimission_tasksize 在 Translate_task() 中设置，如果该slot成功传输任务则为 First_task.size，否则为 None
                    vehicle_throughput = 0.0
                    if hasattr(veh, 'Transimission_tasksize'):
                        if veh.Transimission_tasksize is not None:
                            vehicle_throughput = float(veh.Transimission_tasksize)
                        # 如果为 None，说明该slot没有传输任务，吞吐量为0
                    
                    # 获取车辆当前连接的RSU
                    current_rsu = veh.current_rsu if hasattr(veh, 'current_rsu') else 1
                    
                    # 累计系统总功率和总吞吐量（实际发送的数据量）
                    total_power += vehicle_power
                    total_throughput += vehicle_throughput
                    
                    # 根据RSU分别统计
                    if current_rsu in rsu_stats:
                        rsu_stats[current_rsu]['power'] += vehicle_power
                        rsu_stats[current_rsu]['throughput'] += vehicle_throughput
                        rsu_stats[current_rsu]['aoi_sum'] += veh_aoi
                        rsu_stats[current_rsu]['vehicle_count'] += 1
                    
                    # 保存到当前slot的车辆数据字典（使用车辆ID作为key）
                    slot_vehicle_data[veh.record_id] = [
                        simSlot,
                        veh.record_id,
                        current_rsu,  # RSU_ID
                        veh_aoi,
                        vehicle_power,
                        vehicle_throughput
                    ]
                
                # 计算每个RSU的平均AOI
                rsu_avg_aois = {}
                for i in range(1, num_rsu + 1):
                    if rsu_stats[i]['vehicle_count'] > 0:
                        rsu_avg_aois[i] = rsu_stats[i]['aoi_sum'] / rsu_stats[i]['vehicle_count']
                    else:
                        rsu_avg_aois[i] = 0.0
                
                # 收集系统级数据（包含总功率和总吞吐量，以及每个RSU的统计）
                system_row = [
                    run_idx + 1,  # Run_ID (从1开始)
                    simSlot,
                    AOI,  # 系统AOI
                    len(env.Vehicle),  # 总车辆数
                    total_power,  # 系统总功率
                    total_throughput  # 系统总吞吐量
                ]
                
                # 添加每个RSU的统计
                for i in range(1, num_rsu + 1):
                    system_row.extend([
                        rsu_avg_aois[i],  # RSU平均AOI
                        rsu_stats[i]['vehicle_count'],  # RSU车辆数
                        rsu_stats[i]['power'],  # RSU总功率
                        rsu_stats[i]['throughput']  # RSU总吞吐量
                    ])
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
                        veh_data[2],  # RSU_ID
                        veh_data[3],  # AOI
                        veh_data[4],  # Power
                        veh_data[5]   # Throughput
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
            pass
        except Exception as e:
            print(f"  Warning: Error deleting run_vehicle_data: {e}")
        
        try:
            del run_system_data
        except NameError:
            pass
        except Exception as e:
            print(f"  Warning: Error deleting run_system_data: {e}")
        
        # 清理环境中的车辆对象（参考reset方法）
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
        
        # 强制垃圾回收（清理循环引用）
        gc.collect()
        gc.collect()
    
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
                
                # 构建数据字典（包含系统级和所有RSU的数据）
                data_dict = {
                    'run_id': int(row['Run_ID']),
                    'AOI': float(row['System_AOI']),
                    'n_veh': float(row['Total_Vehicles']),
                    'total_power': float(row['Total_Power']),
                    'total_throughput': float(row['Total_Throughput'])
                }
                
                # 动态读取所有RSU的数据
                for i in range(1, num_rsu + 1):
                    data_dict[f'rsu{i}_aoi'] = float(row.get(f'RSU{i}_AOI', 0))
                    data_dict[f'rsu{i}_n_veh'] = float(row.get(f'RSU{i}_Vehicles', 0))
                    data_dict[f'rsu{i}_power'] = float(row.get(f'RSU{i}_Power', 0))
                    data_dict[f'rsu{i}_throughput'] = float(row.get(f'RSU{i}_Throughput', 0))
                
                system_data_by_slot[slot].append(data_dict)
    
    # 计算系统数据的统计信息
    num_slots = max(system_data_by_slot.keys()) + 1 if system_data_by_slot else 0
    averaged_system_data = []
    
    for slot_idx in range(num_slots):
        if slot_idx not in system_data_by_slot:
            continue
        
        slot_data_list = system_data_by_slot[slot_idx]
        
        # 动态构建slot_values字典（包含系统级和所有RSU的数据）
        slot_values = {
            'AOI': [d['AOI'] for d in slot_data_list],
            'n_veh': [d['n_veh'] for d in slot_data_list],
            'total_power': [d['total_power'] for d in slot_data_list],
            'total_throughput': [d['total_throughput'] for d in slot_data_list]
        }
        
        # 添加所有RSU的数据
        for i in range(1, num_rsu + 1):
            slot_values[f'rsu{i}_aoi'] = [d.get(f'rsu{i}_aoi', 0) for d in slot_data_list]
            slot_values[f'rsu{i}_n_veh'] = [d.get(f'rsu{i}_n_veh', 0) for d in slot_data_list]
            slot_values[f'rsu{i}_power'] = [d.get(f'rsu{i}_power', 0) for d in slot_data_list]
            slot_values[f'rsu{i}_throughput'] = [d.get(f'rsu{i}_throughput', 0) for d in slot_data_list]
        
        # 计算统计信息
        n = len(slot_values['AOI'])
        if n > 0:
            # 计算系统整体的均值和标准差
            mean_aoi = np.mean(slot_values['AOI'])
            std_aoi = np.std(slot_values['AOI'], ddof=1) if n > 1 else 0.0
            
            mean_n_veh = np.mean(slot_values['n_veh'])
            std_n_veh = np.std(slot_values['n_veh'], ddof=1) if n > 1 else 0.0
            
            mean_power = np.mean(slot_values['total_power'])
            std_power = np.std(slot_values['total_power'], ddof=1) if n > 1 else 0.0
            
            mean_throughput = np.mean(slot_values['total_throughput'])
            std_throughput = np.std(slot_values['total_throughput'], ddof=1) if n > 1 else 0.0
            
            # 计算95%置信区间（系统整体）
            if n > 1:
                t_value = stats.t.ppf(1 - alpha/2, n - 1)
                ci_aoi = t_value * std_aoi / np.sqrt(n)
                ci_n_veh = t_value * std_n_veh / np.sqrt(n)
                ci_power = t_value * std_power / np.sqrt(n)
                ci_throughput = t_value * std_throughput / np.sqrt(n)
            else:
                ci_aoi = ci_n_veh = ci_power = ci_throughput = 0.0
            
            # 保存系统整体统计信息
            slot_stats = {
                'slot': slot_idx,
                'AOI': {'mean': mean_aoi, 'std': std_aoi, 'ci_lower': mean_aoi - ci_aoi, 'ci_upper': mean_aoi + ci_aoi},
                'n_veh': {'mean': mean_n_veh, 'std': std_n_veh, 'ci_lower': mean_n_veh - ci_n_veh, 'ci_upper': mean_n_veh + ci_n_veh},
                'total_power': {'mean': mean_power, 'std': std_power, 'ci_lower': mean_power - ci_power, 'ci_upper': mean_power + ci_power},
                'total_throughput': {'mean': mean_throughput, 'std': std_throughput, 'ci_lower': mean_throughput - ci_throughput, 'ci_upper': mean_throughput + ci_throughput}
            }
            
            # 动态计算每个RSU的统计信息
            for i in range(1, num_rsu + 1):
                mean_rsu_aoi = np.mean(slot_values[f'rsu{i}_aoi'])
                std_rsu_aoi = np.std(slot_values[f'rsu{i}_aoi'], ddof=1) if n > 1 else 0.0
                
                mean_rsu_n_veh = np.mean(slot_values[f'rsu{i}_n_veh'])
                std_rsu_n_veh = np.std(slot_values[f'rsu{i}_n_veh'], ddof=1) if n > 1 else 0.0
                
                mean_rsu_power = np.mean(slot_values[f'rsu{i}_power'])
                std_rsu_power = np.std(slot_values[f'rsu{i}_power'], ddof=1) if n > 1 else 0.0
                
                mean_rsu_throughput = np.mean(slot_values[f'rsu{i}_throughput'])
                std_rsu_throughput = np.std(slot_values[f'rsu{i}_throughput'], ddof=1) if n > 1 else 0.0
                
                # 计算95%置信区间
                if n > 1:
                    ci_rsu_aoi = t_value * std_rsu_aoi / np.sqrt(n)
                    ci_rsu_n_veh = t_value * std_rsu_n_veh / np.sqrt(n)
                    ci_rsu_power = t_value * std_rsu_power / np.sqrt(n)
                    ci_rsu_throughput = t_value * std_rsu_throughput / np.sqrt(n)
                else:
                    ci_rsu_aoi = ci_rsu_n_veh = ci_rsu_power = ci_rsu_throughput = 0.0
                
                # 保存RSU统计信息
                slot_stats[f'rsu{i}_aoi'] = {'mean': mean_rsu_aoi, 'std': std_rsu_aoi, 'ci_lower': mean_rsu_aoi - ci_rsu_aoi, 'ci_upper': mean_rsu_aoi + ci_rsu_aoi}
                slot_stats[f'rsu{i}_n_veh'] = {'mean': mean_rsu_n_veh, 'std': std_rsu_n_veh, 'ci_lower': mean_rsu_n_veh - ci_rsu_n_veh, 'ci_upper': mean_rsu_n_veh + ci_rsu_n_veh}
                slot_stats[f'rsu{i}_power'] = {'mean': mean_rsu_power, 'std': std_rsu_power, 'ci_lower': mean_rsu_power - ci_rsu_power, 'ci_upper': mean_rsu_power + ci_rsu_power}
                slot_stats[f'rsu{i}_throughput'] = {'mean': mean_rsu_throughput, 'std': std_rsu_throughput, 'ci_lower': mean_rsu_throughput - ci_rsu_throughput, 'ci_upper': mean_rsu_throughput + ci_rsu_throughput}
            
            averaged_system_data.append(slot_stats)
    
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
                    'Throughput': float(row['Throughput'])
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
            
            # 计算95%置信区间
            if n > 1:
                t_value = stats.t.ppf(1 - alpha/2, n - 1)
                ci_aoi = t_value * std_aoi / np.sqrt(n)
                ci_power = t_value * std_power / np.sqrt(n)
                ci_throughput = t_value * std_throughput / np.sqrt(n)
            else:
                ci_aoi = ci_power = ci_throughput = 0.0
            
            averaged_vehicle_data.append({
                'slot': slot_idx,
                'AOI': {'mean': mean_aoi, 'std': std_aoi, 'ci_lower': mean_aoi - ci_aoi, 'ci_upper': mean_aoi + ci_aoi},
                'Power': {'mean': mean_power, 'std': std_power, 'ci_lower': mean_power - ci_power, 'ci_upper': mean_power + ci_power},
                'Throughput': {'mean': mean_throughput, 'std': std_throughput, 'ci_lower': mean_throughput - ci_throughput, 'ci_upper': mean_throughput + ci_throughput}
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
    
    # ==================== 计算总体统计信息（在所有数据上）====================
    print("\nComputing overall statistics across all slots and runs...")
    
    # 从 system_data_by_slot 收集所有数据点（用于计算总体统计）
    all_aoi_values = []
    all_n_veh_values = []
    all_power_values = []
    all_throughput_values = []
    
    # 为每个RSU创建数据列表
    all_rsu_values = {}
    for i in range(1, num_rsu + 1):
        all_rsu_values[i] = {
            'aoi': [],
            'n_veh': [],
            'power': [],
            'throughput': []
        }
    
    for slot_idx in range(num_slots):
        if slot_idx in system_data_by_slot:
            slot_data_list = system_data_by_slot[slot_idx]
            all_aoi_values.extend([d['AOI'] for d in slot_data_list])
            all_n_veh_values.extend([d['n_veh'] for d in slot_data_list])
            all_power_values.extend([d['total_power'] for d in slot_data_list])
            all_throughput_values.extend([d['total_throughput'] for d in slot_data_list])
            
            # 收集所有RSU的数据
            for i in range(1, num_rsu + 1):
                all_rsu_values[i]['aoi'].extend([d.get(f'rsu{i}_aoi', 0) for d in slot_data_list])
                all_rsu_values[i]['n_veh'].extend([d.get(f'rsu{i}_n_veh', 0) for d in slot_data_list])
                all_rsu_values[i]['power'].extend([d.get(f'rsu{i}_power', 0) for d in slot_data_list])
                all_rsu_values[i]['throughput'].extend([d.get(f'rsu{i}_throughput', 0) for d in slot_data_list])
    
    # 计算总体统计信息（基于所有数据点）
    if len(all_aoi_values) > 0:
        # 计算系统整体总体均值和标准差
        avg_system_aoi = np.mean(all_aoi_values)
        std_system_aoi = np.std(all_aoi_values, ddof=1) if len(all_aoi_values) > 1 else 0.0
        
        avg_total_vehicles = np.mean(all_n_veh_values)
        std_total_vehicles = np.std(all_n_veh_values, ddof=1) if len(all_n_veh_values) > 1 else 0.0
        
        avg_total_power = np.mean(all_power_values)
        std_total_power = np.std(all_power_values, ddof=1) if len(all_power_values) > 1 else 0.0
        
        avg_total_throughput = np.mean(all_throughput_values)
        std_total_throughput = np.std(all_throughput_values, ddof=1) if len(all_throughput_values) > 1 else 0.0
        
        # 动态计算每个RSU的总体均值和标准差
        rsu_stats = {}
        for i in range(1, num_rsu + 1):
            rsu_stats[i] = {
                'avg_aoi': np.mean(all_rsu_values[i]['aoi']) if len(all_rsu_values[i]['aoi']) > 0 else 0.0,
                'std_aoi': np.std(all_rsu_values[i]['aoi'], ddof=1) if len(all_rsu_values[i]['aoi']) > 1 else 0.0,
                'avg_vehicles': np.mean(all_rsu_values[i]['n_veh']) if len(all_rsu_values[i]['n_veh']) > 0 else 0.0,
                'std_vehicles': np.std(all_rsu_values[i]['n_veh'], ddof=1) if len(all_rsu_values[i]['n_veh']) > 1 else 0.0,
                'avg_power': np.mean(all_rsu_values[i]['power']) if len(all_rsu_values[i]['power']) > 0 else 0.0,
                'std_power': np.std(all_rsu_values[i]['power'], ddof=1) if len(all_rsu_values[i]['power']) > 1 else 0.0,
                'avg_throughput': np.mean(all_rsu_values[i]['throughput']) if len(all_rsu_values[i]['throughput']) > 0 else 0.0,
                'std_throughput': np.std(all_rsu_values[i]['throughput'], ddof=1) if len(all_rsu_values[i]['throughput']) > 1 else 0.0
            }
        
        # 计算95%置信区间（基于所有数据点）
        n_total = len(all_aoi_values)
        if n_total > 1:
            t_value = stats.t.ppf(1 - alpha/2, n_total - 1)
            
            # 系统整体置信区间
            se_aoi = std_system_aoi / np.sqrt(n_total)
            se_n_veh = std_total_vehicles / np.sqrt(n_total)
            se_power = std_total_power / np.sqrt(n_total)
            se_throughput = std_total_throughput / np.sqrt(n_total)
            
            margin_aoi = t_value * se_aoi
            margin_n_veh = t_value * se_n_veh
            margin_power = t_value * se_power
            margin_throughput = t_value * se_throughput
            
            avg_aoi_ci_lower = avg_system_aoi - margin_aoi
            avg_aoi_ci_upper = avg_system_aoi + margin_aoi
            avg_vehicles_ci_lower = avg_total_vehicles - margin_n_veh
            avg_vehicles_ci_upper = avg_total_vehicles + margin_n_veh
            avg_power_ci_lower = avg_total_power - margin_power
            avg_power_ci_upper = avg_total_power + margin_power
            avg_throughput_ci_lower = avg_total_throughput - margin_throughput
            avg_throughput_ci_upper = avg_total_throughput + margin_throughput
            
            # 动态计算每个RSU的置信区间
            for i in range(1, num_rsu + 1):
                n_rsu = len(all_rsu_values[i]['aoi'])
                if n_rsu > 1:
                    t_value_rsu = stats.t.ppf(1 - alpha/2, n_rsu - 1)
                    se_rsu_aoi = rsu_stats[i]['std_aoi'] / np.sqrt(n_rsu)
                    se_rsu_power = rsu_stats[i]['std_power'] / np.sqrt(n_rsu)
                    se_rsu_throughput = rsu_stats[i]['std_throughput'] / np.sqrt(n_rsu)
                    
                    margin_rsu_aoi = t_value_rsu * se_rsu_aoi
                    margin_rsu_power = t_value_rsu * se_rsu_power
                    margin_rsu_throughput = t_value_rsu * se_rsu_throughput
                    
                    rsu_stats[i]['aoi_ci_lower'] = rsu_stats[i]['avg_aoi'] - margin_rsu_aoi
                    rsu_stats[i]['aoi_ci_upper'] = rsu_stats[i]['avg_aoi'] + margin_rsu_aoi
                    rsu_stats[i]['power_ci_lower'] = rsu_stats[i]['avg_power'] - margin_rsu_power
                    rsu_stats[i]['power_ci_upper'] = rsu_stats[i]['avg_power'] + margin_rsu_power
                    rsu_stats[i]['throughput_ci_lower'] = rsu_stats[i]['avg_throughput'] - margin_rsu_throughput
                    rsu_stats[i]['throughput_ci_upper'] = rsu_stats[i]['avg_throughput'] + margin_rsu_throughput
                else:
                    rsu_stats[i]['aoi_ci_lower'] = rsu_stats[i]['aoi_ci_upper'] = 0.0
                    rsu_stats[i]['power_ci_lower'] = rsu_stats[i]['power_ci_upper'] = 0.0
                    rsu_stats[i]['throughput_ci_lower'] = rsu_stats[i]['throughput_ci_upper'] = 0.0
        else:
            avg_aoi_ci_lower = avg_aoi_ci_upper = avg_vehicles_ci_lower = avg_vehicles_ci_upper = avg_power_ci_lower = avg_power_ci_upper = avg_throughput_ci_lower = avg_throughput_ci_upper = 0.0
            for i in range(1, num_rsu + 1):
                rsu_stats[i]['aoi_ci_lower'] = rsu_stats[i]['aoi_ci_upper'] = 0.0
                rsu_stats[i]['power_ci_lower'] = rsu_stats[i]['power_ci_upper'] = 0.0
                rsu_stats[i]['throughput_ci_lower'] = rsu_stats[i]['throughput_ci_upper'] = 0.0
    else:
        avg_system_aoi = std_system_aoi = avg_total_vehicles = std_total_vehicles = avg_total_power = std_total_power = avg_total_throughput = std_total_throughput = 0.0
        avg_aoi_ci_lower = avg_aoi_ci_upper = avg_vehicles_ci_lower = avg_vehicles_ci_upper = avg_power_ci_lower = avg_power_ci_upper = avg_throughput_ci_lower = avg_throughput_ci_upper = 0.0
        rsu_stats = {}
        for i in range(1, num_rsu + 1):
            rsu_stats[i] = {
                'avg_aoi': 0.0, 'std_aoi': 0.0, 'avg_vehicles': 0.0, 'std_vehicles': 0.0,
                'avg_power': 0.0, 'std_power': 0.0, 'avg_throughput': 0.0, 'std_throughput': 0.0,
                'aoi_ci_lower': 0.0, 'aoi_ci_upper': 0.0,
                'power_ci_lower': 0.0, 'power_ci_upper': 0.0,
                'throughput_ci_lower': 0.0, 'throughput_ci_upper': 0.0
            }
    
    # ==================== 清理从文件读取的数据结构（统计信息已保存） ====================
    print("\nCleaning up data structures after statistics computation...")
    
    # 清理临时数据
    del all_aoi_values
    del all_n_veh_values
    del all_power_values
    del all_throughput_values
    del all_rsu_values
    
    # 清理从文件读取的数据结构（统计信息已保存到CSV文件）
    del system_data_by_slot
    del vehicle_data_by_slot
    system_data_by_slot = None
    vehicle_data_by_slot = None
    
    # 强制垃圾回收
    gc.collect()
    gc.collect()
    
    # ==================== 保存最终测试总结 ====================
    
    # 保存测试总结到文件
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Test Summary Report (Statistics from 10 Runs)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Number of RSUs: {num_rsu}\n")
        f.write(f"Number of Runs: {num_runs}\n")
        f.write(f"Confidence Level: {confidence_level*100:.1f}%\n")
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
        # 动态写入每个RSU的性能指标
        for i in range(1, num_rsu + 1):
            f.write(f"RSU{i} Performance Metrics (Hidden Size: {DRL_hidden_size}):\n")
            f.write(f"  Average RSU{i} AOI: {rsu_stats[i]['avg_aoi']:.4f} ± {rsu_stats[i]['std_aoi']:.4f}\n")
            f.write(f"    95% Confidence Interval: [{rsu_stats[i]['aoi_ci_lower']:.4f}, {rsu_stats[i]['aoi_ci_upper']:.4f}]\n")
            f.write(f"  Average RSU{i} Vehicles: {rsu_stats[i]['avg_vehicles']:.2f} ± {rsu_stats[i]['std_vehicles']:.2f}\n")
            f.write(f"  Average RSU{i} Power: {rsu_stats[i]['avg_power']:.4f} ± {rsu_stats[i]['std_power']:.4f} W\n")
            f.write(f"    95% Confidence Interval: [{rsu_stats[i]['power_ci_lower']:.4f}, {rsu_stats[i]['power_ci_upper']:.4f}] W\n")
            f.write(f"  Average RSU{i} Throughput: {rsu_stats[i]['avg_throughput']:.4f} ± {rsu_stats[i]['std_throughput']:.4f} bytes\n")
            f.write(f"    95% Confidence Interval: [{rsu_stats[i]['throughput_ci_lower']:.4f}, {rsu_stats[i]['throughput_ci_upper']:.4f}] bytes\n")
            f.write(f"\n")
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
        f.write(f"Model Paths:\n")
        for i in range(1, num_rsu + 1):
            if i <= len(trained_DRL_model_paths):
                f.write(f"  RSU{i} DRL Model: {trained_DRL_model_paths[i-1]}\n")
        f.write(f"  Number of RSUs: {num_rsu}\n")
        f.write("=" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print(f"Test Summary for {algorithm_name} (Number of RSUs = {num_rsu})")
    print("=" * 60)
    print(f"Number of Runs: {num_runs}")
    print(f"Confidence Level: {confidence_level*100:.1f}%")
    print(f"\nAverage Performance Metrics:")
    print(f"  Average System AOI: {avg_system_aoi:.4f} ± {std_system_aoi:.4f}")
    print(f"    95% Confidence Interval: [{avg_aoi_ci_lower:.4f}, {avg_aoi_ci_upper:.4f}]")
    print(f"  Average Total Vehicles: {avg_total_vehicles:.2f} ± {std_total_vehicles:.2f}")
    print(f"    95% Confidence Interval: [{avg_vehicles_ci_lower:.2f}, {avg_vehicles_ci_upper:.2f}]")
    print(f"  Average Total Power: {avg_total_power:.4f} ± {std_total_power:.4f} W")
    print(f"    95% Confidence Interval: [{avg_power_ci_lower:.4f}, {avg_power_ci_upper:.4f}] W")
    print(f"  Average Total Throughput: {avg_total_throughput:.4f} ± {std_total_throughput:.4f} bytes")
    print(f"    95% Confidence Interval: [{avg_throughput_ci_lower:.4f}, {avg_throughput_ci_upper:.4f}] bytes")
    # 动态打印每个RSU的性能指标
    for i in range(1, num_rsu + 1):
        print(f"\nRSU{i} Performance Metrics (Hidden Size: {DRL_hidden_size}):")
        print(f"  Average RSU{i} AOI: {rsu_stats[i]['avg_aoi']:.4f} ± {rsu_stats[i]['std_aoi']:.4f}")
        print(f"    95% Confidence Interval: [{rsu_stats[i]['aoi_ci_lower']:.4f}, {rsu_stats[i]['aoi_ci_upper']:.4f}]")
        print(f"  Average RSU{i} Vehicles: {rsu_stats[i]['avg_vehicles']:.2f} ± {rsu_stats[i]['std_vehicles']:.2f}")
        print(f"  Average RSU{i} Power: {rsu_stats[i]['avg_power']:.4f} ± {rsu_stats[i]['std_power']:.4f} W")
        print(f"    95% Confidence Interval: [{rsu_stats[i]['power_ci_lower']:.4f}, {rsu_stats[i]['power_ci_upper']:.4f}] W")
        print(f"  Average RSU{i} Throughput: {rsu_stats[i]['avg_throughput']:.4f} ± {rsu_stats[i]['std_throughput']:.4f} bytes")
        print(f"    95% Confidence Interval: [{rsu_stats[i]['throughput_ci_lower']:.4f}, {rsu_stats[i]['throughput_ci_upper']:.4f}] bytes")
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
    
    print(f"\n✅ Test completed for {algorithm_name} with Number of RSUs = {num_rsu}.")

if __name__ == "__main__":
    # 到时候也写成 arg
    main()
