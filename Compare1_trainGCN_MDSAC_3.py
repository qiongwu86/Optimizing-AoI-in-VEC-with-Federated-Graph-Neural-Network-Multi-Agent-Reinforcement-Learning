"MADDPG : 用这个，这个训练是考虑其他agent的"
"平均车辆 22 辆 "
import numpy as np
import torch
# from EnvironmentGCN_SAC import VEC_env
from Compare3_EnvironmentGCN_SAC_new_global import  VEC_env
from tqdm import tqdm
# from MADDPG import DDPG_model
# from Environment import My_GCN
from GCN_DRL_SAC_FL_fix import DRL_SAC
from GCN_DRL_SAC_FL_fix import *
import matplotlib.pyplot as plt
import os
from memory_profiler import profile
from torch.utils.tensorboard import SummaryWriter
from maddpg.agent import Agent
import argparse
import sys
from memory_profiler import memory_usage
import copy
import math
# 性能分析
# 1. 和周围所有车辆聚合, without GCN
# 2. RSU 异步聚合
# 3. 其他比如 DQN
# 4.

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="maddpg", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=1010000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--type", type=int, default = 0, help="0:MADDPG, 1 : IDMADDPG")
    parser.add_argument("--lr-actor", type=float, default=1e-3, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-2, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.2, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(2e3), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./save/Compare3_MyCode/model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
def main():
    num_lane = 4 # four lane
    slotT = 0.02
    T = 101 * 1000 # 1000K =
    TotalSlot = T/slotT
    plot_T = 10000 # 10 K
    save_T = 10000
    TotalEpi = 1
    rsuW = 250
    gridW = 50      # 网格数量也是可以做性能分析的
    # vehGenRate = [8,8,8,8] # sec, 性能,

    # vehGenRate = [2, 2, 2, 4]  # sec, 性能,
    # VehSpeed =  [120,100,90,60] # 性能, km/h
    VehSpeed = [60, 90, 100, 120]  # 性能, km/h
    VehSpeed = [30, 35, 40, 50]  # 性能, km/h
    TaskGenRate = 0.2 # slot, 性能

    "固定--------------------------------------数量----------------------------------------------"
    # vehGenRate = [8, 8, 8, 8]

    vehNum = 22

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

    agents = []
    args = get_args()
    args.n_agents = vehNum  # 需要操控的玩家个数 3个，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args.obs_shape = [ 6 for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
    action_shape = []
    for i in range(vehNum):
        action_shape.append(1)
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    args.high_action = maxP
    args.low_action = 0
    for i in range(vehNum):
        agent = Agent(i, args)  # 建立agent
        agents.append(agent)  # 添加到list里面




    # DRL_model = DRL_SAC(state_dim=6,action_dim=1,max_action=maxP,policy_rate = policy_rate, critic_rate = critic_rate ,alpha_lr = alpha_lr,reward_scale=reward_scale) # --------------------------------------------------------------------------------------------

    "--------------------------------------节点特征数量, 取哪些特征-------------------------------------------------"
    node_feature_size = 5 # vehicle number,...
    # vehicle
    # init env
    # DDPG 和 GNN　一个　循环里训练算了
    env = VEC_env(lane = num_lane,vehGenRate = args.n_agents,
                  slotT = slotT,VehSpeed =VehSpeed,rsuW=rsuW,
                  plot=False,TaskGenRate=TaskGenRate,TaskSize =TaskSize,
                  TransmissionRange = TransmissionRange,gridW = gridW,node_feature_size = node_feature_size, out_feature = 1,
                  DRL_model=  agents,GCN_batchsize = GCN_batchsize,p = p,maxP =maxP,slideW = slideW,w = w,k=k,
                  penalty_factor = penalty_factor,powerFactor=powerFactor,
                  param_noise_var = param_noise_var,
                  GCN_factor = GCN_factor,
                  critic_noise_var =critic_noise_var,
                  rateDecay = rateDecay,
                  GCN_critic = ConcatMlp,
                  reward_scale=reward_scale,
                  args = args
                  )



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



    dir = "./save/Compare3_MyCode/"
    data_dir = dir + "TRAIN_DATA/"
    DRL_dir = dir + "SAC/"
    GCN_dir = dir + "GCN/"
    fig_dir = dir + "fig/"
    tensorboard_dir = dir + "tensorboard2"
    writer = SummaryWriter(tensorboard_dir)

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
    # writer = SummaryWriter(tensorboard_dir)
    epi = 0
    for episode in range(int(TotalSlot/args.max_episode_len) + 1):
        # env.reset(episode)
        env.reset(epi)
        env.episode  = episode
        AOE_T = []
        for simSlot in range(args.max_episode_len):
            AOI, mean_veh_reward ,penalty,  n_veh, exceed_n, _, _ = env.step()
            AOE_T.append(AOI)
            print("episode:", episode , "slot : ", simSlot, ", exceed number : ", exceed_n,' AOI', np.mean(AOE_T))

        writer.add_scalar('AOI', np.mean(AOE_T), episode)
        epi += 1





        #     R_T.append(AOI+penalty) # reward








if __name__ == "__main__":
    # 到时候也写成 arg
    main()
