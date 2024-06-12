import numpy as np
import torch
# from EnvironmentGCN_SAC import VEC_env
from EnvironmentGCN_SAC_test_game import  VEC_env
# from MADDPG import DDPG_model
# from Environment import My_GCN
from GCN_DRL_SAC_FL_fix import DRL_SAC
from GCN_DRL_SAC_FL_fix import *
import matplotlib.pyplot as plt
import os
# from memory_profiler import profile
# from torch.utils.tensorboard import SummaryWriter
import sys
# from memory_profiler import memory_usage
import copy
import math
# 性能分析
# 1. 和周围所有车辆聚合, without GCN
# 2. RSU 异步聚合
# 3. 其他比如 DQN
# 4.


def main():
    num_lane = 4 # four lane
    slotT = 0.02
    T = 2 * 1000 # 1000K =
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
    VehSpeed = [60, 90, 100, 120]  # 性能, km/h
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
    penalty_factor = 0.9999
    powerFactor = 1
    param_noise_var = 1e-4
    critic_noise_var = 0.1
    GCN_factor = 0.3
    policy_rate = 1e-4
    critic_rate = 1e-3
    alpha_lr = 1e-4
    rateDecay = 0.9999
    reward_scale=0.1

    dir = "./save/MyCode1/"
    # data_dir = dir + "TRAIN_DATA/"
    DRL_dir = dir + "SAC/"


    DRL_model = DRL_SAC(state_dim=6,action_dim=1,max_action=maxP,policy_rate = policy_rate, critic_rate = critic_rate ,alpha_lr = alpha_lr,reward_scale=reward_scale) # --------------------------------------------------------------------------------------------
    DRL_model.load(DRL_dir + "SAC_train_" + "_")
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
                  reward_scale=reward_scale
                  )
    R_T = []
    AOE_T =[]
    veh_n_T = []
    numberVeh_AOI_T = {}
    AOE_T = []
    Power_T = []
    Queue_T = []
    N_veh_T = []
    thoughout_T = []
    for simSlot in range(int(TotalSlot)):
        AOI, mean_power,mean_Q,n_veh,thoughout= env.step()
        print("Slot : ",simSlot,
              " n_veh : ", n_veh ,
              " AOI : ", AOI,
              " Power : ", mean_power,
              " Queue : ", mean_Q,
              " Thoughout",thoughout)
        AOE_T.append(AOI)
        Power_T.append(mean_power)
        Queue_T.append(mean_Q)
        N_veh_T.append(n_veh)
        thoughout_T.append(thoughout)

    plt.plot(AOE_T)
    plt.show()






if __name__ == "__main__":
    # 到时候也写成 arg
    main()
