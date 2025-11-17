import numpy as np
import torch
# from EnvironmentGCN_SAC import VEC_env
from EnvironmentGCN_SAC_new_global2 import  VEC_env
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

# 导入优化工具
from memory_optimizer import MemoryOptimizer, GradientOptimizer, ModelOptimizer
from simple_gcn_memory import SimpleGCNMemory, SimpleMemory

class TrainingMonitor:
    """训练监控类，记录详细的训练时间和内存信息"""
    
    def __init__(self, log_file="training_monitor.log"):
        self.log_file = log_file
        self.start_time = time.time()
        self.episode_start_time = None
        self.slot_start_time = None
        
        # 车辆训练统计
        self.vehicle_training_stats = {}
        self.vehicle_memory_stats = {}
        
        # 整体统计
        self.total_training_time = 0
        self.total_memory_usage = []
        self.peak_memory = 0
        
        # 进程信息
        self.process = psutil.Process()
        
        # 创建日志文件
        with open(self.log_file, 'w') as f:
            f.write("Training Monitor Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
    
    def start_episode(self, episode):
        """开始新的episode"""
        self.episode_start_time = time.time()
        self.vehicle_training_stats[episode] = {}
        self.vehicle_memory_stats[episode] = {}
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpisode {episode} started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def start_slot(self, slot):
        """开始新的slot"""
        self.slot_start_time = time.time()
    
    def record_vehicle_training(self, episode, slot, vehicle_id, training_time, memory_usage):
        """记录单辆车的训练信息"""
        if episode not in self.vehicle_training_stats:
            self.vehicle_training_stats[episode] = {}
        if slot not in self.vehicle_training_stats[episode]:
            self.vehicle_training_stats[episode][slot] = {}
        
        self.vehicle_training_stats[episode][slot][vehicle_id] = {
            'training_time': training_time,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        }
        
        # 记录到日志文件
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode}, Slot {slot}, Vehicle {vehicle_id}: "
                   f"Training Time: {training_time:.4f}s, Memory: {memory_usage:.2f}MB\n")
    
    def record_slot_summary(self, episode, slot, total_vehicles, slot_time, memory_usage):
        """记录slot的汇总信息"""
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode}, Slot {slot} Summary: "
                   f"Vehicles: {total_vehicles}, Slot Time: {slot_time:.4f}s, "
                   f"Memory: {memory_usage:.2f}MB\n")
    
    def record_memory_usage(self, context=""):
        """记录当前内存使用情况"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024 # 将内存使用情况转换为MB
        self.total_memory_usage.append(memory_mb) # 将内存使用情况添加到总内存使用情况列表中
        
        if memory_mb > self.peak_memory:
            self.peak_memory = memory_mb
        
        with open(self.log_file, 'a') as f:
            f.write(f"Memory Usage {context}: {memory_mb:.2f}MB\n")
        
        return memory_mb
    
    def get_vehicle_average_stats(self, episode):
        """获取车辆平均统计信息"""
        if episode not in self.vehicle_training_stats:
            return None
        
        total_training_time = 0
        total_memory = 0
        total_vehicles = 0
        
        for slot_data in self.vehicle_training_stats[episode].values():
            for vehicle_data in slot_data.values():
                total_training_time += vehicle_data['training_time']
                total_memory += vehicle_data['memory_usage']
                total_vehicles += 1
        
        if total_vehicles == 0:
            return None
        
        return {
            'avg_training_time': total_training_time / total_vehicles,
            'avg_memory_usage': total_memory / total_vehicles,
            'total_vehicles': total_vehicles
        }
    
    def get_overall_stats(self):
        """获取整体统计信息"""
        total_time = time.time() - self.start_time
        avg_memory = sum(self.total_memory_usage) / len(self.total_memory_usage) if self.total_memory_usage else 0
        
        return {
            'total_training_time': total_time,
            'average_memory_usage': avg_memory,
            'peak_memory_usage': self.peak_memory,
            'current_memory_usage': self.total_memory_usage[-1] if self.total_memory_usage else 0
        }
    
    def save_summary(self):
        """保存训练总结"""
        overall_stats = self.get_overall_stats()
        
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Training Time: {overall_stats['total_training_time']:.2f} seconds\n")
            f.write(f"Average Memory Usage: {overall_stats['average_memory_usage']:.2f} MB\n")
            f.write(f"Peak Memory Usage: {overall_stats['peak_memory_usage']:.2f} MB\n")
            f.write(f"Current Memory Usage: {overall_stats['current_memory_usage']:.2f} MB\n")
            f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")

# 性能分析
# 1. 和周围所有车辆聚合, without GCN
# 2. RSU 异步聚合
# 3. 其他比如 DQN
# 4.

def main():
    # 初始化内存优化器
    memory_optimizer = MemoryOptimizer("training_memory.log")
    memory_optimizer.log_memory_usage("Program Start")
    
    # 初始化训练监控器
    training_monitor = TrainingMonitor("detailed_training_monitor.log")
    training_monitor.record_memory_usage("Program Start")
    
    num_lane = 4 # four lane
    slotT = 0.02
    T = 20 * 1000 # 1000K =
    TotalSlot = T/slotT
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
  

    DRL_model = DRL_SAC(state_dim=6,
                        action_dim=1,
                        max_action=maxP,
                        policy_rate = policy_rate, 
                        critic_rate = critic_rate, 
                        alpha_lr = alpha_lr,reward_scale=reward_scale) # --------------------------------------------------------------------------------------------

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

    # 优化：替换环境中的内存管理类 
    if hasattr(env, 'GCN_model') and hasattr(env.GCN_model, 'GCN_memory'):
        # 保存原有数据
        old_memory = env.GCN_model.GCN_memory # 保存原有GCN内存
        old_data = {
            'memory_counter': old_memory.memory_counter,    # 保存原有GCN内存计数器
            'G_data': list(old_memory.G_data) if hasattr(old_memory, 'G_data') else [],    # 保存原有GCN数据
            'G_data_': list(old_memory.G_data_) if hasattr(old_memory, 'G_data_') else [],    # 保存原有GCN数据
            'memory_r': list(old_memory.memory_r) if hasattr(old_memory, 'memory_r') else []    # 保存原有GCN数据
        } # 保存原有GCN数据
        
        # 创建新的简化内存管理
        env.GCN_model.GCN_memory = SimpleGCNMemory(  # 创建新的简化GCN内存管理
            old_memory.memory_size,     # 保存原有GCN旧的容量
            4 * (int(rsuW * 2 /gridW)) * node_feature_size,     # 保存原有GCN数据大小
            node_feature_size * 1  # out_feature     # 保存原有GCN数据大小
        )     # 创建新的简化GCN内存管理
        
        # 迁移数据
        for i in range(len(old_data['G_data'])):    # 迁移原有GCN数据到新的简化GCN内存管理
            env.GCN_model.GCN_memory.Addremember( 
                torch.zeros(env.num_nodes * node_feature_size),  #  迁移原有GCN数据到新的简化GCN内存管理占位符
                old_data['memory_r'][i] if i < len(old_data['memory_r']) else 0,
                torch.zeros(env.num_nodes * node_feature_size),  #  迁移原有GCN数据到新的简化GCN内存管理占位符
                old_data['G_data'][i] if i < len(old_data['G_data']) else None,
                old_data['G_data_'][i] if i < len(old_data['G_data_']) else None
            )
        
        # 清理旧内存
        del old_memory, old_data    # 清理旧内存
        gc.collect()    # 清理垃圾回收

    R_E = []
    AOI_E = []
    GCNLoss_E = []
    numberVeh_AOI_E = {}
    losses_E = {}
    destory_AOI_E = []
    veh_n_E = []

    #mean veh
    R_vehmean_E =[]
    AOI_vehmean_E = []
    # losses_policy_E = []
    # losses_qf1_E = []
    # losses_qf2_E = []
    log_pi_E =[]
    q_new_actions_E  = []

    losses_policy_E = []
    losses_qf1_E = []
    losses_qf2_E = []
    q1_E = []
    q2_E = []
    q_target_E = []
    train_r_E = []
    target_q_values_E = []

    dir = "./save/MyCode1_p_0.1/"
    data_dir = dir + "TRAIN_DATA/"
    DRL_dir = dir + "SAC/"
    GCN_dir = dir + "GCN/"
    fig_dir = dir + "fig/"
    tensorboard_dir = dir + "tensorboard2"

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(DRL_dir):
        os.mkdir(DRL_dir)
    if not os.path.exists(GCN_dir):
        os.mkdir(GCN_dir)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    
    # 优化：减少TensorBoard记录频率，避免内存累积
    writer = SummaryWriter(tensorboard_dir, max_queue=5, flush_secs=60)
    
    memory_optimizer.log_memory_usage("After Environment Initialization")
    
    for episode in range(TotalEpi):
        with memory_optimizer.memory_context(f"Episode {episode}"):
            # 开始episode监控
            training_monitor.start_episode(episode)
            training_monitor.record_memory_usage(f"Episode {episode} Start")
            
            env.reset(episode)
            env.episode  = episode
            # env.globalaggreNumber = 0

            # 改为累积和+计数的方式，避免列表无限增长
            R_T_sum = 0.0  # 总reward累积和
            R_T_count = 0   # 计数
            
            AOE_T_sum = 0.0  # AOI累积和
            AOE_T_count = 0   # 计数（注意：需要定期清零）
            
            penalty_T_sum = 0.0
            penalty_T_count = 0
            
            mean_veh_reward_T_sum = 0.0
            mean_veh_reward_T_count = 0
            
            gnnloss_T_sum = 0.0
            gnnloss_T_count = 0
            
            gnnCriticloss_T_sum = 0.0
            gnnCriticloss_T_count = 0
            
            # 这些只需要保存最后一个值，不需要列表
            q1_last = 0.0
            q2_last = 0.0
            q_target_last = 0.0
            train_r_last = 0.0
            target_q_values_last = 0.0
            losses_policy_last = 0.0
            losses_qf1_last = 0.0
            losses_qf2_last = 0.0
            gnnloss_last = 0.0
            gnnCriticloss_last = 0.0
            # modelStart = env.DRL_model.actor.state_dict()['l1.weight'][:4][0]
            
            for simSlot in range(int(TotalSlot)):
                # 开始slot监控
                training_monitor.start_slot(simSlot)
                slot_start_time = time.time()
                
                # 定期内存监控和清理
                if simSlot % 5000 == 0:
                    memory_optimizer.log_memory_usage(f"Slot {simSlot}")
                    memory_optimizer.cleanup_memory()
                    training_monitor.record_memory_usage(f"Slot {simSlot}")

                if simSlot % plot_T == 0 :
                     # 清零AOE_T的累积和（对应原来的AOE_T = []）
                     AOE_T_sum = 0.0
                     AOE_T_count = 0

                print("episode:", episode , "slot : ", simSlot, ", vehicle number : ", len(env.Vehicle))
                # if simSlot % 1000 == 0:
                    # print("0")
                
                # 使用内存监控的训练步骤
                AOI, mean_veh_reward ,penalty,  n_veh, destroyAOI,  gnnloss, gnnCriticloss = memory_optimizer.monitor_training_step(
                    env.step
                )
                
                # 记录slot时间
                
                slot_time = time.time() - slot_start_time
                current_memory = training_monitor.record_memory_usage(f"Slot {simSlot} End")
                training_monitor.record_slot_summary(episode, simSlot, len(env.Vehicle), slot_time, current_memory)

                # 使用累积和方式，而不是append，避免内存持续增长
                R_T_sum += (AOI + penalty)
                R_T_count += 1
                
                AOE_T_sum += AOI
                AOE_T_count += 1
                
                penalty_T_sum += penalty
                penalty_T_count += 1
                
                mean_veh_reward_T_sum += mean_veh_reward
                mean_veh_reward_T_count += 1
                
                if gnnloss is not None:
                    gnnloss_T_sum += gnnloss
                    gnnloss_T_count += 1
                    gnnloss_last = gnnloss  # 同时保存最后一个值

                if gnnCriticloss is not None:
                    gnnCriticloss_T_sum += gnnCriticloss
                    gnnCriticloss_T_count += 1
                    gnnCriticloss_last = gnnCriticloss  # 同时保存最后一个值

                if simSlot % plot_T == 0:
                    # 记录每辆车的训练时间和内存
                    for veh in env.Vehicle:
                        veh_training_start = time.time()
                        _ = veh.model.getGradient(veh.Training_number)
                        veh_training_time = time.time() - veh_training_start
                        
                        # 获取车辆内存使用情况
                        veh_memory = training_monitor.record_memory_usage(f"Vehicle {veh.record_id}")
                        
                        # 记录车辆训练信息
                        training_monitor.record_vehicle_training(
                            episode, simSlot, veh.record_id, 
                            veh_training_time, veh_memory
                        )
                    
                    meanLoss, q1, q2, q_target, train_r, target_q_values = env.getMeanLoss()
                    # 只保存最后一个值，不需要列表
                    losses_policy_last = meanLoss["policy_loss"]
                    losses_qf1_last = meanLoss["qf1_loss"]
                    losses_qf2_last = meanLoss["qf2_loss"]

                    q1_last = q1
                    q2_last = q2
                    q_target_last = q_target
                    train_r_last = train_r * reward_scale
                    target_q_values_last = target_q_values
                    
                    # 优化：减少TensorBoard记录频率，避免内存累积
                    # 只在特定间隔记录，并使用更轻量的记录方式
                    if simSlot % (plot_T * 2) == 0:  # 减少记录频率
                        # 计算平均值：总和/计数
                        aoi_avg = AOE_T_sum / AOE_T_count if AOE_T_count > 0 else 0
                        reward_avg = R_T_sum / R_T_count if R_T_count > 0 else 0
                        penalty_avg = penalty_T_sum / penalty_T_count if penalty_T_count > 0 else 0
                        mean_veh_reward_avg = mean_veh_reward_T_sum / mean_veh_reward_T_count if mean_veh_reward_T_count > 0 else 0
                        
                        writer.add_scalar('AOI', aoi_avg, simSlot)
                        writer.add_scalar('Total_Reward', reward_avg, simSlot)
                        writer.add_scalar('Penalty', penalty_avg, simSlot)
                        writer.add_scalar('Mean_Vehicle_Reward', mean_veh_reward_avg, simSlot)
                        
                        # 使用最后一个值
                        writer.add_scalar('Q_Values/q1_mean', q1_last, simSlot)
                        writer.add_scalar('Q_Values/q2_mean', q2_last, simSlot)
                        
                        # Q值异常检测
                        if abs(q1_last) > 1000 or abs(q2_last) > 1000:
                            print(f"⚠️ WARNING: Q values too large at slot {simSlot}")
                            print(f"   Q1: {q1_last:.2f}, Q2: {q2_last:.2f}")
                            writer.add_text('Warnings/Q_Value_Anomaly', 
                                           f'Q1={q1_last:.2f}, Q2={q2_last:.2f} at slot {simSlot}', 
                                           simSlot)
                        
                        # 记录reward相关的组合数据
                        writer.add_scalars('Reward_Components', {
                            'AOI': aoi_avg,
                            'Penalty': penalty_avg,
                            'Total_Reward': reward_avg,
                            'Mean_Vehicle_Reward': mean_veh_reward_avg
                        }, simSlot)
                        
                        # 记录penalty倍数信息（作为文本标签）
                        writer.add_text('Config/Penalty_Multiplier', f'penalty_multiplier = {penalty_multiplier}', simSlot)
                        
                        # 记录训练时间和内存信息
                        writer.add_scalar('Training_Time/Slot_Time', slot_time, simSlot)
                        writer.add_scalar('Memory_Usage/Current_Memory', current_memory, simSlot)
                        
                        # 记录车辆平均训练统计
                        vehicle_stats = training_monitor.get_vehicle_average_stats(episode)
                        if vehicle_stats:
                            writer.add_scalar('Vehicle_Stats/Avg_Training_Time', vehicle_stats['avg_training_time'], simSlot)
                            writer.add_scalar('Vehicle_Stats/Avg_Memory_Usage', vehicle_stats['avg_memory_usage'], simSlot)
                            writer.add_scalar('Vehicle_Stats/Total_Vehicles', vehicle_stats['total_vehicles'], simSlot)
                        
                        writer.add_scalars('Q value', {
                            'q1_E': q1_last,
                            'q2_E': q2_last,
                            "q_target_E": q_target_last,
                            "train_r_E": train_r_last,
                            "target_q_values_E": target_q_values_last
                        }, simSlot)

                        writer.add_scalars('Q loss', {
                            'q1_loss': losses_qf1_last,
                            'q2_loss': losses_qf2_last
                        }, simSlot)
                        writer.add_scalar('policy loss', losses_policy_last, simSlot)
                        
                        # GCN loss使用平均值
                        gnnloss_avg = gnnloss_T_sum / gnnloss_T_count if gnnloss_T_count > 0 else gnnloss_last
                        gnnCriticloss_avg = gnnCriticloss_T_sum / gnnCriticloss_T_count if gnnCriticloss_T_count > 0 else gnnCriticloss_last
                        writer.add_scalar('gnn loss', gnnloss_avg, simSlot)
                        writer.add_scalar('gnn critic loss', gnnCriticloss_avg, simSlot)
                        
                        # 只在更长的间隔记录直方图
                        if simSlot % (plot_T * 10) == 0:
                            # 记录关键参数的直方图，减少记录数量
                            for name, param in env.DRL_model.policy.named_parameters():
                                if 'weight' in name:  # 只记录权重参数
                                    # 使用detach避免计算图累积
                                    param_detached = param.detach().cpu()
                                    writer.add_histogram("actor" + name, param_detached.numpy(), simSlot)
                                    del param_detached  # 立即删除临时变量
                            
                            # 清理临时变量
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # 优化：定期清理内存
                    if simSlot % 5000 == 0:
                        gc.collect()  # 强制垃圾回收
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # 清理GPU缓存
                    
                    # 修改1：调整探索衰减，定期降低学习率
                    if simSlot % 50000 == 0 and simSlot > 0:  # 每50k步衰减一次
                        # 衰减policy学习率
                        for param_group in env.DRL_model.policy_optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] *= 0.9
                            new_lr = param_group['lr']
                            print(f"Policy LR decayed: {old_lr:.6f} -> {new_lr:.6f}")
                        
                        # 衰减Q1网络学习率
                        for param_group in env.DRL_model.qf1_optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] *= 0.9
                            new_lr = param_group['lr']
                            print(f"Q1 LR decayed: {old_lr:.6f} -> {new_lr:.6f}")
                        
                        # 衰减Q2网络学习率
                        for param_group in env.DRL_model.qf2_optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] *= 0.9
                            new_lr = param_group['lr']
                            print(f"Q2 LR decayed: {old_lr:.6f} -> {new_lr:.6f}")
                            
                if simSlot % save_T == 0:
                    # 使用优化的模型保存
                    ModelOptimizer.save_model_optimized(env.DRL_model, DRL_dir + "SAC_train_" + "_")
                    ModelOptimizer.save_model_optimized(env.GCN_model, GCN_dir + "GCN_train_" + "_")

                def change_values_and_check_memory():
                    # 保存每次循环的内存使用情况
                    memory_usages = {attr: [] for attr in env.__dict__.keys()}
                    for attr in env.__dict__.keys():
                        memory_usages[attr].append(sys.getsizeof(getattr(env, attr)))
                    return memory_usages

                if simSlot % 10000 == 0:
                    # 使用 memory_usage 函数检查内存使用情况
                    # mem_usage = memory_usage(change_values_and_check_memory)
                    mem_usage = change_values_and_check_memory()
                    # 打开文件并写入内存使用情况
                    memory_dir = dir + 'memory_usage.txt'
                    with open(memory_dir, 'a+') as f:
                        for attr, usages in mem_usage.items():
                            f.write(f" {simSlot} : Memory usage of attribute {attr}: {usages}\n")
                        f.write(f"-----------------------------------------------------------\n")
                        
                    # 优化：定期清理环境中的累积数据
                    if hasattr(env, 'GCN_model') and hasattr(env.GCN_model, 'GCN_memory'):
                        # 清理GCN内存中的旧数据
                        gcn_memory = env.GCN_model.GCN_memory
                        if len(gcn_memory.G_data) > gcn_memory.memory_size * 1.2:
                            # 使用优化内存管理的清理方法
                            gcn_memory._cleanup_memory()
                            print(f"Cleaned GCN memory at slot {simSlot}")
            
            # 优化：在episode结束时清理内存
            print(f"Episode {episode} completed. Cleaning up memory...")
            memory_optimizer.cleanup_memory(force=True)
            
            # 记录episode结束时的统计信息
            episode_stats = training_monitor.get_vehicle_average_stats(episode)
            if episode_stats:
                print(f"Episode {episode} Vehicle Stats:")
                print(f"  Average Training Time: {episode_stats['avg_training_time']:.4f}s")
                print(f"  Average Memory Usage: {episode_stats['avg_memory_usage']:.2f}MB")
                print(f"  Total Vehicles: {episode_stats['total_vehicles']}")
            
            training_monitor.record_memory_usage(f"Episode {episode} End")
            
            # 清理环境中的临时数据
            if hasattr(env, 'Vehicle'):
                for veh in env.Vehicle:
                    if hasattr(veh, 'model') and hasattr(veh.model, 'memory'):
                        # 清理车辆内存
                        if hasattr(veh.model.memory, 'memory'):
                            veh.model.memory.memory_counter = min(veh.model.memory.memory_counter, veh.model.memory.memory_size)
    
    # 优化：关闭TensorBoard writer
    writer.close()
    
    # 最终内存报告
    memory_summary = memory_optimizer.get_memory_summary()
    print("=== Final Memory Summary ===")
    for key, value in memory_summary.items():
        print(f"{key}: {value}")
    
    # 保存训练监控总结
    training_monitor.save_summary()
    
    # 打印最终训练统计
    overall_stats = training_monitor.get_overall_stats()
    print("\n=== Final Training Statistics ===")
    print(f"Total Training Time: {overall_stats['total_training_time']:.2f} seconds")
    print(f"Average Memory Usage: {overall_stats['average_memory_usage']:.2f} MB")
    print(f"Peak Memory Usage: {overall_stats['peak_memory_usage']:.2f} MB")
    print(f"Current Memory Usage: {overall_stats['current_memory_usage']:.2f} MB")
    
    print("Training completed and memory cleaned up.")

if __name__ == "__main__":
    # 到时候也写成 arg
    main()
