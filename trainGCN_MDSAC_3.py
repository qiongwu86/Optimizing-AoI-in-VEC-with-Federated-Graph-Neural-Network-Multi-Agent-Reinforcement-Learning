import numpy as np
import torch
# from EnvironmentGCN_SAC import VEC_env
"改变车辆的数量 vehGenRate = [10, 10, 10, 10]"
from EnvironmentGCN_SAC_new_global2 import  VEC_env
# from MADDPG import DDPG_model
# from Environment import My_GCN
from GCN_DRL_SAC_FL_fix import DRL_SAC
from GCN_DRL_SAC_FL_fix import *
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.tensorboard import SummaryWriter
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
    vehGenRate = [10, 10, 10, 10]
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

    DRL_model = DRL_SAC(state_dim=6,action_dim=1,max_action=maxP,policy_rate = policy_rate, critic_rate = critic_rate ,alpha_lr = alpha_lr,reward_scale=reward_scale) # --------------------------------------------------------------------------------------------

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

    losses_qf1_E = []
    losses_policy_E = []
    losses_qf2_E = []
    q1_E = []
    q2_E = []
    q_target_E = []
    train_r_E = []
    target_q_values_E = []



    dir = "./save/MyCode3/"
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
    writer = SummaryWriter(tensorboard_dir)
    for episode in range(TotalEpi):
        env.reset(episode)
        env.episode  = episode
        # env.globalaggreNumber = 0

        R_T = []
        AOE_T =[]
        GCNLoss_T = []
        veh_n_T = []
        numberVeh_AOI_T = {}
        losses_policy_T = []
        losses_qf1_T = []
        losses_qf2_T = []
        # mean veh
        R_vehmean_T = []
        AOI_vehmean_T = []
        losses_vehmean_T = {}

        losses_policy_T = []
        gnnloss_T = []
        gnnCriticloss_T =[]
        losses_qf1_T = []
        losses_qf2_T = []
        q1_T = []
        q2_T = []
        q_target_T = []
        train_r_T = []
        target_q_values_T = []
        # modelStart = env.DRL_model.actor.state_dict()['l1.weight'][:4][0]
        for simSlot in range(int(TotalSlot)):

            if simSlot % plot_T == 0 :
                 AOE_T = []

            print("episode:", episode , "slot : ", simSlot, ", vehicle number : ", len(env.Vehicle))
            # if simSlot % 1000 == 0:
                # print("0")
            AOI, mean_veh_reward ,penalty,  n_veh, destroyAOI,  gnnloss, gnnCriticloss = env.step()

        #     R_T.append(AOI+penalty) # reward
            AOE_T.append(AOI)
            if gnnloss is not None:
                gnnloss_T.append(gnnloss)

            if gnnCriticloss is not None:
                gnnCriticloss_T.append(gnnCriticloss)

            if simSlot % plot_T == 0:
                for veh in env.Vehicle:
                    _ = veh.model.getGradient(veh.Training_number)
                # AOI_E.append(np.mean(AOE_T))
                meanLoss, q1, q2, q_target, train_r, target_q_values = env.getMeanLoss()
                losses_policy_T.append(meanLoss["policy_loss"])
                losses_qf1_T.append(meanLoss["qf1_loss"])
                losses_qf2_T.append(meanLoss["qf2_loss"])

                q1_T.append(q1)
                q2_T.append(q2)
                q_target_T.append(q_target)
                train_r_T.append(train_r * reward_scale)
                target_q_values_T.append(target_q_values)
                for name, param in env.DRL_model.policy.named_parameters():
                            writer.add_histogram("actor" + name, param.clone().cpu().data.numpy(), simSlot)
                            if param.requires_grad:
                                grad_sum = None
                                count = 0
                                for veh in env.Vehicle:
                                    for namev, paramv in veh.model.policy.named_parameters():
                                        if namev == name:
                                            if paramv.grad is not None:
                                                if grad_sum is None:
                                                    grad_sum = paramv.grad.clone()
                                                else:
                                                    grad_sum += paramv.grad.clone()
                                                count += 1
                                if len(env.Vehicle) > 0 :
                                    writer.add_histogram("actor" + name + '_grad', grad_sum / len(env.Vehicle), simSlot)
                                else:
                                    writer.add_histogram("actor" + name + '_grad', 0, simSlot)

                for name, param in env.DRL_model.qf1.named_parameters():
                            writer.add_histogram("Q1" + name, param.clone().cpu().data.numpy(), simSlot)
                            if param.requires_grad:
                                grad_sum = None
                                count = 0
                                for veh in env.Vehicle:
                                    for namev, paramv in veh.model.qf1.named_parameters():
                                        if namev == name:
                                            if paramv.grad is not None:
                                                if grad_sum is None:
                                                    grad_sum = paramv.grad.clone()
                                                else:
                                                    grad_sum += paramv.grad.clone()
                                                count += 1
                                if len(env.Vehicle) > 0:
                                    writer.add_histogram("Q1" + name + '_grad', grad_sum / count, simSlot)
                                else:
                                    writer.add_histogram("Q1" + name + '_grad', 0, simSlot)

                for name, param in env.DRL_model.qf2.named_parameters():
                            writer.add_histogram("Q2" + name, param.clone().cpu().data.numpy(), simSlot)
                            if param.requires_grad:
                                grad_sum = None
                                count = 0
                                for veh in env.Vehicle:
                                    for namev, paramv in veh.model.qf2.named_parameters():
                                        if namev == name:
                                            if paramv.grad is not None:
                                                if grad_sum is None:
                                                    grad_sum = paramv.grad.clone()
                                                else:
                                                    grad_sum += paramv.grad.clone()
                                                count += 1
                                if len(env.Vehicle) > 0:
                                    writer.add_histogram("Q2" + name + '_grad', grad_sum / len(env.Vehicle), simSlot)
                                else:
                                    writer.add_histogram("Q2" + name + '_grad', 0, simSlot)

                for name, param in env.DRL_model.target_qf1.named_parameters():
                            writer.add_histogram("target_qf1" + name, param.clone().cpu().data.numpy(), simSlot)

                for name, param in env.DRL_model.target_qf2.named_parameters():
                            writer.add_histogram("target_qf2" + name, param.clone().cpu().data.numpy(), simSlot)

                for name, param in env.GCN_model.GCN_critic.named_parameters():
                            writer.add_histogram("GNN critic" + name, param.clone().cpu().data.numpy(), simSlot)
                for name, param in env.GCN_model.GCN_target_critic.named_parameters():
                            writer.add_histogram("GNN target critic" + name, param.clone().cpu().data.numpy(), simSlot)
                for name, param in env.GCN_model.conv1.named_parameters():
                            writer.add_histogram("GNN conv1" + name, param.clone().cpu().data.numpy(), simSlot)
                for name, param in env.GCN_model.conv2.named_parameters():
                            writer.add_histogram("GNN conv2" + name, param.clone().cpu().data.numpy(), simSlot)
                for name, param in env.GCN_model.conv3.named_parameters():
                            writer.add_histogram("GNN conv3" + name, param.clone().cpu().data.numpy(), simSlot)
                # for name, param in env.GCN_model.conv1.parameters():
                # writer.add_histogram("GNN conv1", env.GCN_model.conv1.parameters().clone().cpu().data.numpy(), simSlot)

                writer.add_scalar('AOI', np.mean(AOE_T), simSlot)

                writer.add_scalars('Q value', {'q1_E': q1_T[-1],
                                                'q2_E': q2_T[-1],
                                                "q_target_E": q_target_T[-1],
                                                "train_r_E": (train_r_T[-1]),
                                                "target_q_values_E": target_q_values_T[-1]}, simSlot)

                writer.add_scalars('Q loss', {'q1_loss': losses_qf1_T[-1],
                                                      'q2_loss': losses_qf2_T[-1]}, simSlot)
                writer.add_scalar('policy loss', losses_policy_T[-1], simSlot)
                writer.add_scalar('gnn loss', gnnloss_T[-1], simSlot)
                writer.add_scalar('gnn critic loss', gnnCriticloss_T[-1], simSlot)
            #
            if simSlot % save_T == 0:
                env.DRL_model.save(DRL_dir + "SAC_train_" + "_")
                env.GCN_model.save(GCN_dir + "GCN_train_" + "_")

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
        # AOI_E.append(np.mean(AOE_T))
        # meanLoss, q1, q2,q_target,train_r,target_q_values = env.getMeanLoss()
        # losses_policy_E.append(meanLoss["policy_loss"])
        # losses_qf1_E.append(meanLoss["qf1_loss"])
        # losses_qf2_E.append(meanLoss["qf2_loss"])
        #
        # q1_E.append(q1)
        # q2_E.append(q2)
        # q_target_E.append(q_target)
        # train_r_E.append(train_r * 0.001)
        # target_q_values_E.append(target_q_values)

        # weights_policy = env.DRL_model.policy.state_dict()
        # weights_q1 = env.DRL_model.policy.state_dict()
        # weights_q2 = env.DRL_model.policy.state_dict()
        # # 将权重追加写入.txt文件
        # with open('weights.txt', 'a') as f:
        #     f.write("episode :" +str(episode) + '\n')
        #     f.write('\n')  # 添加空行分隔不同层的权重
        #     f.write("policy:"  + '\n')
        #     f.write('\n')  # 添加空行分隔不同层的权重
        #     for key, value in weights_policy.items():
        #         f.write(key + '\n')
        #         f.write(str(value) + '\n')
        #         f.write('\n')  # 添加空行分隔不同层的权重
        #     f.write("q1:"  + '\n')
        #     f.write('\n')  # 添加空行分隔不同层的权重
        #     for key, value in weights_q1.items():
        #         f.write(key + '\n')
        #         f.write(str(value) + '\n')
        #         f.write('\n')  # 添加空行分隔不同层的权重
        #     f.write("q2:"  + '\n')
        #     f.write('\n')  # 添加空行分隔不同层的权重
        #     for key, value in weights_q2.items():
        #         f.write(key + '\n')
        #         f.write(str(value) + '\n')
        #         f.write('\n')  # 添加空行分隔不同层的权重
        #     f.write("AOI :" + str(AOI_E[-1]) + '\n')
        #     f.write("-----------------------------------------------")
        #     f.write('\n')  # 添加空行分隔不同层的权重
        #
        #     if n_veh !=0:
        #         R_vehmean_T.append((AOI+penalty)/n_veh)
        #         AOI_vehmean_T.append(AOI/n_veh)
        #     else:
        #         R_vehmean_T.append(AOI+penalty)
        #         AOI_vehmean_T.append(AOI)
        #
        #
        #     veh_n_T.append(n_veh)
        #     GCNLoss_T.append(GCNLoss)
        #
        #
        #     if n_veh !=0 and n_veh not in numberVeh_AOI_T.keys():
        #         numberVeh_AOI_T[n_veh] = []
        #     if n_veh != 0:
        #         numberVeh_AOI_T[n_veh].append(AOI)
        #
        #     for net,loss_value in losses.items():
        #         if net not in losses_T.keys():
        #             losses_T[net]= []
        #         if net not in losses_vehmean_T:
        #             losses_vehmean_T[net] = []
        #
        #         if n_veh != 0:
        #             losses_vehmean_T[net].append(loss_value/n_veh)
        #         else:
        #             losses_vehmean_T[net].append(loss_value)
        #         losses_T[net].append(loss_value)
        #
        #
        #     print("Episode : ", episode, " slot : ", simSlot , " reward : ", penalty, "AOI : ", AOI)
        #     # print("model T0 : ", modelStart)
        #
        # R_E.append(np.mean(R_T))

        # R_vehmean_E.append(np.mean(R_vehmean_T))
        # AOI_vehmean_E.append(np.mean(AOI_vehmean_T))
        #
        # GCNLoss_E.append(np.mean(GCNLoss_T))
        # destory_AOI_E.append(np.mean(destroyAOI))
        # veh_n_E.append(np.mean(veh_n_T))
        #
        # for n_veh , T_AOI in numberVeh_AOI_T.items():
        #      if n_veh not in numberVeh_AOI_E.keys():
        #          numberVeh_AOI_E[n_veh] = []
        #      numberVeh_AOI_E[n_veh].append(np.mean(numberVeh_AOI_T[n_veh]))
        #
        # for net,loss_t_value in losses_T.items():
        #     if net not in losses_E.keys():
        #         losses_E[net]=[]
        #     losses_E[net].append(np.mean(losses_T[net]))
        #
        # for net,loss_t_value in losses_vehmean_T.items():
        #     if net not in losses_vehmean_E:
        #         losses_vehmean_E[net] = []
        #     losses_vehmean_E[net].append(np.mean(losses_vehmean_T[net]))
        #
        # Train_param = "gridW_"+str(gridW)+"_range"+str(TransmissionRange)+"_vehRate_"+ ''.join(str(i) for i in vehGenRate)\
        #               +"_TaskRandS_"+str(TaskGenRate)+str(TaskSize)+"_"+str(episode)
        #
        # #
        # # "------------------------------------------ plot ---------------------------------------------"
        # plt.plot(R_E)
        # plt.title("AOI penalty no mean veh")
        # plt.savefig(fig_dir + Train_param + "AOI penalty no mean.jpg")
        # plt.savefig(fig_dir + Train_param + "AOI penalty no mean.eps")
        # plt.show()
        #

        # for name, param in env.DRL_model.policy.named_parameters():
        #     writer.add_histogram("actor" + name, param.clone().cpu().data.numpy(), episode)
        # for name, param in env.DRL_model.qf1.named_parameters():
        #     writer.add_histogram("Q1" + name, param.clone().cpu().data.numpy(), episode)
        # for name, param in env.DRL_model.qf2.named_parameters():
        #     writer.add_histogram("Q2" + name, param.clone().cpu().data.numpy(), episode)
        # for name, param in env.DRL_model.target_qf1.named_parameters():
        #     writer.add_histogram("target_qf1" + name, param.clone().cpu().data.numpy(), episode)
        # for name, param in env.DRL_model.target_qf2.named_parameters():
        #     writer.add_histogram("target_qf1" + name, param.clone().cpu().data.numpy(), episode)
        #
        #
        # writer.add_scalar('AOI', AOI_E[-1], episode)
        # writer.add_scalars('Q value', {'q1_E': q1_E[-1],
        #                                'q2_E': q2_E[-1],
        #                                "q_target_E":q_target_E[-1],
        #                                "train_r_E":(train_r_E[-1] ),
        #                                "target_q_values_E":target_q_values_E[-1] }, episode)
        #
        # writer.add_scalars('Q loss', {'q1_loss': losses_qf1_E[-1],
        #                                'q2_loss': losses_qf2_E[-1] }, episode)
        # writer.add_scalar('policy loss',losses_policy_E[-1], episode)

        # if episode % 20 == 0:
        #     print(AOI_E)
        #
        #     plt.plot(AOI_E)
        #     plt.title("mean AOI no mean veh action fix")
        #     # plt.savefig(fig_dir +Train_param+ "mean AOI no mean veh.jpg")
        #     # plt.savefig(fig_dir + Train_param + "mean AOI no mean veh.eps")
        #
        #     plt.savefig(fig_dir + "mean AOI no mean veh action fix.jpg")
        #     plt.show()
        #
        #     plt.plot(q1_E)
        #     plt.plot(q2_E)
        #     plt.plot(q_target_E)
        #     plt.plot(train_r_E)
        #     plt.plot(target_q_values_E)
        #     plt.legend(["q1", "q2", "qtarget", "train_r_E", "target_q_values_E"])
        #     plt.savefig(fig_dir + "q1_q2_qtarget_train_r_E_target_q_values_E.jpg")
        #     plt.show()
        #
        #     plt.plot(losses_policy_E)
        #     plt.title("losses_policy")
        #     plt.savefig(fig_dir + "losses_policy.jpg")
        #     plt.show()
        #     plt.plot(losses_qf1_E)
        #     plt.title("losses_qf1")
        #     plt.savefig(fig_dir + "losses_qf1.jpg")
        #     plt.show()
        #     plt.plot(losses_qf2_E)
        #     plt.title("losses_qf2")
        #     plt.savefig(fig_dir + "losses_qf2.jpg")
        #     plt.show()
        #
        # plt.plot(GCNLoss_E)
        # plt.title("GCN loss")
        # plt.yscale('log')
        # plt.savefig(fig_dir + Train_param + "GCN loss.jpg")
        # plt.savefig(fig_dir + Train_param + "GCN loss.eps")
        # plt.show()
        #
        # plt.plot(destory_AOI_E)
        # plt.title("detory AOI no mean veh")
        # plt.savefig(fig_dir + Train_param + "detory AOI no mean veh.jpg")
        # plt.savefig(fig_dir + Train_param + "detory AOI no mean veh.eps")
        # plt.show()
        #
        # #
        #
        # plt.plot(R_vehmean_E)
        # plt.title("AOI penalty mean veh")
        # plt.savefig(fig_dir + Train_param + "AOI penalty mean veh.jpg")
        # plt.savefig(fig_dir + Train_param + "AOI penalty mean veh.eps")
        # plt.show()
        #
        # plt.plot(AOI_vehmean_E)
        # plt.title("mean AOI mean veh")
        # plt.savefig(fig_dir + Train_param + "mean AOI mean veh.jpg")
        # plt.savefig(fig_dir + Train_param + "mean AOI mean veh.eps")
        # plt.show()
        #
        # n_veh_list = []
        # for n_veh , E_mean_AOI in numberVeh_AOI_E.items():
        #      plt.plot(E_mean_AOI)
        #      n_veh_list.append(n_veh)
        # plt.legend(n_veh_list)
        # plt.savefig(fig_dir + Train_param + "each number of veh of AOI.jpg")
        # plt.savefig(fig_dir + Train_param + "each number of veh of AOI.eps")
        # plt.show()
        #
        # for net,loss_e_value in losses_E.items():
        #     plt.plot(loss_e_value)
        #     plt.title(net + " loss")
        #     plt.savefig(fig_dir + Train_param + net + "mean loss no mean veh.jpg")
        #     plt.savefig(fig_dir + Train_param + net + "mean loss no mean veh.eps")
        #     plt.show()
        #
        # for net,loss_e_value in losses_vehmean_E.items():
        #     plt.plot(loss_e_value)
        #     plt.title(net + " loss")
        #     plt.savefig(fig_dir + Train_param + net + "mean loss mean veh.jpg")
        #     plt.savefig(fig_dir + Train_param + net + "mean loss mean veh.eps")
        #     plt.show()
        # "------------------------------------save----------------------------------------"
        #
        #
        #
        # if episode % 10 == 0:
        #     env.DRL_model.save(DRL_dir + "SAC_train_"+ Train_param+"_")
        #     env.GCN_model.save(GCN_dir + "GCN_train_" + Train_param+"_")
        #
        #
        #     # save data
        #     np.save(data_dir + " reward_penalty_epi_"+Train_param+".npy",R_E)
        #     np.save(data_dir + " AOI_epi_" + Train_param + ".npy", AOI_E)
        #
        #     np.save(data_dir + " mean veh reward_penalty_epi_"+Train_param+".npy",R_vehmean_E)
        #     np.save(data_dir + " mean veh AOI_epi_" + Train_param + ".npy", AOI_vehmean_E)
        #
        #     np.save(data_dir + " GCN_loss_epi_"+Train_param+".npy",GCNLoss_E)
        #     np.save(data_dir + " vehnum_epi_"+Train_param+".npy",numberVeh_AOI_E)
        #     np.save(data_dir + " destory_aoi" + Train_param + ".npy", destory_AOI_E)
        #     np.save(data_dir + " mean vehicle number" + Train_param + ".npy", veh_n_E)
        #
        #     for net,loss_e_value in losses_E.items():
        #         np.save(data_dir + net + "mean_loss no mean veh"+Train_param+".npy", loss_e_value)
        #     for net, loss_e_value in losses_vehmean_E.items():
        #         np.save(data_dir + net + "mean_loss mean veh" + Train_param + ".npy", loss_e_value)




if __name__ == "__main__":
    # 到时候也写成 arg
    main()
