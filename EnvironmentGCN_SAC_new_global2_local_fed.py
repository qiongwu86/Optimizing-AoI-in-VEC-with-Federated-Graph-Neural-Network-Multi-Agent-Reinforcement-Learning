import numpy as np
import torch
import pdb
import copy
from torch.nn import Linear
from nn_function import pytorch_util as ptu
import uuid
import matplotlib.pyplot as plt
from scipy import special as sp
from scipy.constants import pi
from scipy import special
import networkx as nx
import time
import project_backend as pb
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from collections import deque
# from memory_profiler import profile
autograd.set_detect_anomaly(True)

# from thop import profile
# from thop import clever_format
memory_file = open('./memory_profile.txt', 'a+')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ref_loss = 0.001
ref_loss = 10
shadowing_dev = 2.2
rayleigh_var = 1.0
n0_dB = -114.0 - 30
noise_var = np.power(10.0, n0_dB / 10)
f_c = 28 * 10**9
v_c = 3 * 10**8
dcor = 10 #  the correlation length of the environment.

Hight_RSU = 10
# Data size scales
BYTE = 8    #8位
KB = 1024*BYTE
MB = 1024*KB
GB = 1024*MB
TB = 1024*GB
PB = 1024*TB

KHZ = 1e3
MHZ = KHZ*1e3
GHZ = MHZ*1e3
W = 200 * MHZ # 200 MHZ

Training_number = [5, 10, 20, 40, 50]
# Training_slot   = [2, 4, 10, 16, 20]
# Training_slot   = [8, 16, 40, 64, 80]
Training_slot   = [10, 20, 40, 80, 100]
Training_slot   = [25, 50, 100, 200, 250]
#Training_slot   = [50, 100, 200, 400, 500]
class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest
def build_graph(edge_len,connect_edge):
    # TEST
    connect_edge = torch.tensor([[10,10,20,20,20,30,21,26,26,27,27,27,28,28],[11,20,30,11,21,25,25,27,29,29,28,33,31,32]])
    edge_len = {10:{11:5,20:6},11:{20:3},20:{30:7,21:2}, 30 :{25:4},21:{25:10},
                26:{27:10,29:15},27:{29:3,28:8,33:11},28:{31:7,32:9}}

    graph = {}
    c_e = connect_edge.detach().numpy()
    # collect connect node
    node_list = []
    for n, m in zip(c_e[0], c_e[1]):
        node_list.append(n)
        node_list.append(m)
    node_list = sorted(set(node_list))
    Renumber = {}
    for i in range(len(node_list)):
        Renumber[i] = node_list[i]


    for node1_key,node1_value in Renumber.items():
        for node2_key,node2_value in Renumber.items():
            if node1_key not in graph.keys():
                graph[node1_key] = {}
            if node1_key == node2_key:
                continue
            if node1_value in edge_len.keys() and node2_value in edge_len[node1_value].keys():
               graph[node1_key][node2_key] =edge_len[node1_value][node2_value]
            elif node2_value in edge_len.keys() and node1_value in edge_len[node2_value].keys():
                    graph[node1_key][node2_key] = edge_len[node2_value][node1_value]
            else:
                graph[node1_key][node2_key] = 10**10
    return graph,Renumber

def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)

    return tree
def find_odd_vertexes(MST):
    # node degree
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)
    return vertexes
def minimum_weight_matching(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)
    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u
        MST.append((v, closest, length))
        # if length == 10**10:
        #     continue
        odd_vert.remove(closest)

def remove_edge_from_matchedMST(MatchedMST, v1, v2):

    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]

    return MatchedMST
def find_eulerian_tour(MatchedMSTree, G):
    # find neigbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # print("Neighbours: ", neighbours)

    # finds the hamiltonian circuit
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w
    return EP
def Chrisrofide(edge_len,connect_edge):
    tsp_G,Renumber= build_graph(edge_len,connect_edge)
    MSTree = minimum_spanning_tree(tsp_G)
    odd_vertexes = find_odd_vertexes(MSTree)
    minimum_weight_matching(MSTree, tsp_G, odd_vertexes)
    eulerian_tour = find_eulerian_tour(MSTree, tsp_G)
    current = eulerian_tour[0]
    path = [current]
    visited = [False] * len(eulerian_tour)
    visited[eulerian_tour[0]] = True
    length = 0
    for v in eulerian_tour:
        if not visited[v]:
            path.append(v)
            visited[v] = True
            length += tsp_G[current][v]
            current = v
    length += tsp_G[current][eulerian_tour[0]]
    path.append(eulerian_tour[0])
    overlay_edge =[[],[]]
    for path_i , path_j in zip(path[0:-1],path[1:]):
        if tsp_G[path_i][path_j] == 10**10:
            continue
        overlay_edge[0].append(Renumber[path_i])
        overlay_edge[1].append(Renumber[path_j])
    # print(overlay_edge)
    return overlay_edge

    # return length, path

def sigmoid(x):
    exp_input = -x
    result = 1.0/(1+np.exp(exp_input))
        
    
    # 只在结果为NaN或Inf时打印
    if np.isnan(result).any() or np.isinf(result).any():
        print(f"⚠️⚠️⚠️[DEBUG] sigmoid: exp_input = {exp_input}, result = {result}")
        import pdb
        pdb.set_trace()

    return result
def complexGaussian(row=1, col=1, amp=1.0):
    real = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    img = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    return amp*(real + 1j*img)
def get_random_rayleigh_variable(rayleigh_var):
    # 确保rayleigh_var为正数
    safe_var = max(1e-10, rayleigh_var)
    
    # 生成随机数
    real_part = rayleigh_var * np.random.randn()
    img_part = rayleigh_var * np.random.randn()
    
    # 检查是否为NaN或无穷大
    if np.isnan(real_part) or np.isinf(real_part) or np.isnan(img_part) or np.isinf(img_part):
        real_part = 0
        img_part = 0
        import pdb
        pdb.set_trace()
    
    result = np.sqrt(2.0/np.pi) * (real_part + 1j * img_part)
    
    # 检查结果
    if np.isnan(result) or np.isinf(result):
        return 1e-10 + 1j * 1e-10
        import pdb
        pdb.set_trace()
    return result
def get_markov_rayleigh_variable(state, correlation, rayleigh_var):
    # 确保correlation在有效范围内
    correlation = np.clip(correlation, -1, 1)
    
    # 确保rayleigh_var为正数
    safe_var = max(1e-10, rayleigh_var)
    
    # 检查state是否为NaN或无穷大
    if np.isnan(state) or np.isinf(state):
        state = 0
    
    # 计算sqrt(1-correlation^2)，确保非负数
    sqrt_term = 1 - np.square(correlation)
    if sqrt_term < 0:
        sqrt_term = 0
    sqrt_term = np.sqrt(sqrt_term)
    
    # 生成随机数
    real_part = rayleigh_var * np.random.randn()
    img_part = rayleigh_var * np.random.randn()
    
    # 检查随机数
    if np.isnan(real_part) or np.isinf(real_part) or np.isnan(img_part) or np.isinf(img_part):
        real_part = 0
        img_part = 0
    
    # 计算结果
    result = correlation * state + sqrt_term * np.sqrt(2.0/np.pi) * (real_part + 1j * img_part)
    
    # 检查结果
    if np.isnan(result) or np.isinf(result):
        return 1e-10 + 1j * 1e-10
        import pdb
        pdb.set_trace() 

    return result

# GCN相关类已删除


class New_task:
    def __init__(self,TaskSize):
        super().__init__()
        self.size = TaskSize
        self.AOI = 0
    def __del__(self):
        del self
# @profile(precision=4, stream=memory_file)
class VEC_veh:
    def __init__(self,lane,VehSpeed,slotT,rsuX,rsuY,dis,taskGenRate,TaskSize,TransmissionRange,node,model,record_veh_num,reward,seed=123):
        super().__init__()
        self.idx = uuid.uuid4()
        self.record_id = record_veh_num
        self.lane = lane
        self.slotT = slotT
        self.slot  = 0
        self.taskGenRate = taskGenRate
        self.x = 0

        self.y = (3.5/2) + self.lane * .5
        self.v = VehSpeed
        self.Loc = np.array([self.x,self.y,0])
        self.rsuLoc = np.array([rsuX,rsuY,Hight_RSU])
        self.lamda = f_c/v_c # wavelength                     #
        # self.neiborVeh = getneiborVeh(self.x, )            # 邻居车辆
        self.dis = dis
        self.sigma2 = 1e-9
        self.node = node
        self.TransmissionRange = TransmissionRange
        self.maxTaskSize = TaskSize
        self.TaskSize = np.random.uniform(0.1, self.maxTaskSize) * MB
        self.Transimission_tasksize = 0
        self.current_shadowing = np.random.randn()
        self.large_currentH = self.calLargeV2IH()
        self.small_currentH = get_random_rayleigh_variable(rayleigh_var)
        self.channel = self.calChannelGain()
        # self.channel = np.abs(np.random.randn()) / 1e6
        self.sinr = 0
        self.lastSINR = 0
        self.rho1 = 0
        # task
        self.GenTaskNextSlot = np.random.poisson(lam=self.taskGenRate)
        self.queue = []
        self.GenerateTask()
        self.collectAOI()
        self.Translate = False
        "model"
        self.model = model
        self.n_GlobalDRL = 0
        self.n_LocalDRL = 0
        self.trainNumber = 0
        TaskSize = 0
        if len(self.queue) >0:
            TaskSize = self.queue[0].size
        # self.s0      = np.concatenate(([0], [0], [0],[TaskSize],[0],[self.appro_action]))
        self.s0 = np.concatenate(([0], [0], [0], [0], [TaskSize], [0]))

        self.reward = reward
        self.ration = 0
        # self.s1      = np.concatenate(([0], [0], [0],[TaskSize],[0],[self.appro_action]))
        self.s1 = np.concatenate(([0], [0], [0], [0], [TaskSize], [0]))
        s0,self.action,_ = self.model.select_action(self.s0,self.record_id)
        self.action_pre = 0
        self.appro_action = self.model.select_action(self.s0,self.record_id)
        self.last_action = self.action
        self.done = False
        self.update_counter = 256
        self.Training_number = np.random.choice(Training_number)
        self.trainingTimeslot = Training_slot[Training_number.index(self.Training_number)]
        self.need_local_aggregate = False
        self.global_aggregate = False
        self.actor_loss = 0
        self.critic_loss = 0
        self.env_AOI = 0
        self.Transimission_AOI = 0
        self.aggre_veh = []
        self.aggre_weight = []
        self.generateNoise()

    def generateNoise(self):

        s0, act, _ = self.model.select_action(self.s0, self.record_id)
        # self.actorNoise = torch.normal(mean=0, std=self.noise_var, size=act.shape)

        q1 = self.model.qf1(s0, act)
        q2 = self.model.qf1(s0, act)


    def collectAOI(self):
        # print AOI information
        self.AOI_list = []
        for task in self.queue:
            self.AOI_list.append(task.AOI)
        # self.AOI_list.append(self.Transimission_AOI)
        
        # 修复：如果AOI_list为空，设置默认值避免nan
        if not self.AOI_list:
            self.AOI_list = [0.0]  # 设置默认AOI值

    def GenerateTask(self):
        self.Transimission_AOI = 0
        if self.GenTaskNextSlot <= 0:
            self.GenTaskNextSlot = np.random.poisson(lam=self.taskGenRate)
            self.TaskSize = np.random.uniform(0.1, self.maxTaskSize) * MB
            self.queue.append(New_task(self.TaskSize))
        self.collectAOI()

    def __del__(self):
        del self.queue
        # del self.model
        del self
    def reset(self,model):

        self.TaskSize = np.random.uniform(0.1, self.maxTaskSize) * MB
        self.Transimission_tasksize = 0
        self.current_shadowing = np.random.randn()
        self.large_currentH = self.calLargeV2IH()
        self.small_currentH = get_random_rayleigh_variable(rayleigh_var)
        self.channel = self.calChannelGain()
        # self.channel = np.abs(np.random.randn()) / 1e6
        self.sinr = 0
        self.lastSINR = 0
        self.rho1 = 0
        # task
        self.GenTaskNextSlot = np.random.poisson(lam=self.taskGenRate)
        self.queue = []
        self.GenerateTask()
        self.collectAOI()
        self.Translate = False
        "model"
        self.n_GlobalDRL = 0
        self.n_LocalDRL = 0
        self.trainNumber = 0

        TaskSize = 0
        if len(self.queue) >0:
            TaskSize = self.queue[0].size
        # self.s0      = np.concatenate(([0], [0], [0],[TaskSize],[0],[self.appro_action]))
        self.s0 = np.concatenate(([0], [0], [0], [0], [TaskSize], [0]))

        self.reward = 0
        self.ration = 0
        # self.s1      = np.concatenate(([0], [0], [0],[TaskSize],[0],[self.appro_action]))
        self.s1 = np.concatenate(([0], [0], [0], [0], [TaskSize], [0]))
        s0,self.action,_ = self.model.select_action(self.s0,self.record_id)
        self.action_pre = 0
        self.appro_action = self.model.select_action(self.s0,self.record_id)
        self.last_action = self.action
        self.done = False
        self.need_local_aggregate = False
        self.global_aggregate = False
        self.actor_loss = 0
        self.critic_loss = 0
        self.env_AOI = 0
        self.Transimission_AOI = 0
        self.aggre_veh = []
        self.aggre_weight = []
        del self.model
        self.model  = model

    def updateChannel(self):
        self.updateV2I()
        self.updateV2V()
    def calSmallV2IH(self):
        return get_markov_rayleigh_variable(self.small_lastH,self.rho2,rayleigh_var)
    def calLargeV2IH(self):
        # g_dB = - (128.1 + 37.6* np.log10(0.001*self.dis)) + shadowing_dev * self.current_shadowing
        
        # 确保距离和频率为正数，防止log10计算NaN
        safe_dis = max(0.001, self.dis)  # 确保距离至少为1mm
        safe_fc = max(1e6, f_c)  # 确保频率至少为1MHz
        
        g_dB = - (32.4 + 20 * np.log10(0.001 * safe_dis) + 20 * np.log10(safe_fc)) + shadowing_dev * self.current_shadowing
        
        # 限制g_dB范围，防止数值溢出
        g_dB = np.clip(g_dB, -200, 200)
        
        # 检查g_dB是否为NaN或无穷大
        if np.isnan(g_dB) or np.isinf(g_dB):
            print(f"⚠️ ⚠️ Warning: Invalid g_dB detected for vehicle {self.record_id}, setting to minimum value")
            import pdb
            pdb.set_trace()
            g_dB = -200
        
        self.g = np.power(10.0, g_dB/10.0)
        
        # 限制增益范围，防止数值溢出
        self.g = np.clip(self.g, 1e-20, 1e20)
        
        # 检查增益是否为NaN或无穷大
        if np.isnan(self.g) or np.isinf(self.g):
            print(f"⚠️ ⚠️ Warning: Invalid channel gain detected for vehicle {self.record_id}, setting to minimum value")    
            import pdb
            pdb.set_trace()
            self.g = 1e-20
            
        return self.g
    def calChannelGain(self):
        # print("测试信道：",self.dis,self.current_shadowing,self.g,self.large_currentH,self.small_currentH)
        # if np.isnan(self.current_shadowing):
        #     print("0")
        
        # 确保large_currentH为非负数，防止sqrt计算NaN
        safe_large_H = max(0, self.large_currentH)
        
        # 检查输入是否为NaN或无穷大
        if np.isnan(safe_large_H) or np.isinf(safe_large_H):
            print(f"⚠️ ⚠️ Warning: Invalid large_currentH detected for vehicle {self.record_id}, setting to minimum value")  
            import pdb
            pdb.set_trace()
            safe_large_H = 1e-20
            
        if np.isnan(self.small_currentH) or np.isinf(self.small_currentH):
            print(f"⚠️ ⚠️ Warning: Invalid small_currentH detected for vehicle {self.record_id}, setting to minimum value")   
            import pdb
            pdb.set_trace()
            small_currentH = 1e-10
        else:
            small_currentH = self.small_currentH
        
        channel_gain = np.sqrt(safe_large_H) * abs(small_currentH)
        
        # 限制信道增益范围
        channel_gain = np.clip(channel_gain, 1e-20, 1e10)
        
        # 检查结果是否为NaN或无穷大
        if np.isnan(channel_gain) or np.isinf(channel_gain):
            print(f"⚠️ ⚠️ Warning: Invalid channel gain calculated for vehicle {self.record_id}, setting to minimum value")
            import pdb
            pdb.set_trace()
            channel_gain = 1e-20
            
        return channel_gain
    def updateV2I(self):
        self.last_shadowing = self.current_shadowing
        self.compute_V2IRho1()
        self.current_shadowing = self.rho1 * self.last_shadowing + np.random.randn() * np.sqrt(1-np.square(self.rho1))

        self.large_currentH = self.calLargeV2IH()

        self.small_lastH = self.small_currentH
        self.compute_V2IRho2()
        self.small_currentH = self.calSmallV2IH()
        self.channel = self.calChannelGain()
        # self.channel = np.abs(np.random.randn()) / 1e6
        # self.channel = np.abs(np.random.randn()) / 1e6
    def updateV2V(self):
        pass
    def compute_V2IRho1(self):
        x_0 = np.array([1, 0, 0])
        # self.rho1 = np.exp( - self.v * self.slotT * np.dot(x_0, (self.rsuLoc - self.Loc)) / (np.linalg.norm(self.rsuLoc - self.Loc))/dcor)

        self.rho1 = sp.j0(2 * pi * self.slotT * self.v * np.dot(x_0, (self.rsuLoc - self.Loc))
                          / (np.linalg.norm((self.rsuLoc - self.Loc)) * dcor))

        exp_input = - self.dis/dcor
        self.rho1 = np.exp(exp_input)
        
        # 只在结果为NaN或Inf时打印
        if np.isnan(self.rho1) or np.isinf(self.rho1):
            print(f"⚠️⚠️⚠️[DEBUG] compute_V2IRho1: dis = {self.dis}, dcor = {dcor}, exp_input = {exp_input}, rho1 = {self.rho1}")
            import pdb
            pdb.set_trace()

    def compute_V2IRho2(self):
        x_0 = np.array([1, 0, 0])
        f_d =  self.v * self.slotT * np.dot(x_0, (self.rsuLoc - self.Loc)) / \
               (np.linalg.norm(self.rsuLoc - self.Loc)) *f_c/(self.slotT*v_c)

        f_d = self.dis * f_c / (self.slotT*v_c)

        self.rho2 =  special.j0(2.0 * np.pi * f_d * self.slotT)

    def compute_sinr(self,allPowerGain,uploadmodelPower):
        self.lastSINR = self.sinr
        # SINR updating needs to be after last SINR
        
        # 添加数值稳定性检查，防止除零和负数
        denominator = allPowerGain - self.action * self.channel + noise_var + uploadmodelPower
        
        # 确保分母不为零或负数
        if denominator <= 1e-10:
            self.sinr = 1e-10  # 设置最小值
        else:
            self.sinr = self.action * self.channel / denominator
        
        # 限制SINR范围，防止数值溢出
        self.sinr = np.clip(self.sinr, 1e-10, 1e10)
        
        # 检查SINR是否为NaN或无穷大
        if np.isnan(self.sinr) or np.isinf(self.sinr):
            print(f"⚠️ ⚠️ Warning: Invalid SINR detected for vehicle {self.record_id}, setting to minimum value")
            import pdb
            pdb.set_trace()
            self.sinr = 1e-10
        
        # 安全计算v2iRate，确保log2的参数为正数
        if self.sinr <= -1:
            self.v2iRate = 0
        else:
            # 使用数值稳定的log2计算
            self.v2iRate = W * np.log2(1 + self.sinr)
            
        # 检查v2iRate是否为NaN或无穷大
        if np.isnan(self.v2iRate) or np.isinf(self.v2iRate):
            print(f"⚠️ ⚠️ Warning: Invalid v2iRate detected for vehicle {self.record_id}, setting to 0")
            import pdb
            pdb.set_trace()
            self.v2iRate = 0
            
        # print("测试通信:",self.record_id,self.rho1,self.sinr,self.v2iRate)

    def process_state(self,x):
        x = torch.tensor(x)
        x_max = x.max(dim=0,keepdim=True).values
        # 添加数值稳定性处理，防止除零和log(0)
        x_max = torch.clamp(x_max, min=1e-8)
        x_normalized_input = torch.clamp(x / x_max + 1, min=1e-8)
        x_log = torch.log10(x_normalized_input)
        x_min = x_log.min(dim=0, keepdim=True).values
        # 防止除零
        denominator = torch.clamp(1 - x_min, min=1e-8)
        x_normalized = (x_log - x_min) / denominator
        x_standardized = F.normalize(x_normalized, dim=0)
        # return x_standardized.numpy()
        return x.numpy()

    def updatestate(self,n_v):
        self.channel = self.calChannelGain()
        # self.channel = np.abs(np.random.randn()) /1e6
        self.meanAoI = 0
        self.maxAOI = 0
        for task in self.queue:
            self.meanAoI += task.AOI
        # print(self.queue)

        if self.queue:
            self.meanAoI = self.meanAoI/len(self.queue)
            self.maxAOI = self.queue[0].AOI

        TaskSize = 0
        if len(self.queue) >0:
            TaskSize = self.queue[0].size
        # print("测试状态 ：", self.record_id,self.channel,self.meanAoI,self.lastSINR,TaskSize)

        # return self.process_state(np.concatenate(([self.channel],[self.meanAoI],[self.lastSINR],[TaskSize/MB],[n_v])))
        if np.isnan(self.channel) or np.isnan(self.lastSINR):
                print(self.channel, self.maxAOI, self.env_AOI, self.lastSINR, self.dis, TaskSize / MB, n_v)
                import pdb
                pdb.set_trace()
        return self.process_state(np.concatenate(([self.channel * 1e12 ],
                                                  [self.maxAOI],
                                                  [self.env_AOI],
                                                  [self.dis],
                                                  [TaskSize/MB],
                                                  [n_v])))
    # def updatestate(self,n_v,appro_action):
    #     self.channel = self.calChannelGain()
    #     self.meanAoI = 0
    #     for task in self.queue:
    #         self.meanAoI += task.AOI
    #     # print(self.queue)
    #     if self.queue:
    #         self.meanAoI = self.meanAoI/len(self.queue)
    #
    #     TaskSize = 0
    #     if len(self.queue) >0:
    #         TaskSize = self.queue[0].size
    #     # print("测试状态 ：", self.record_id,self.channel,self.meanAoI,self.lastSINR,TaskSize)
    #     return self.process_state(np.concatenate(([self.channel],[self.meanAoI],[self.lastSINR],[TaskSize/MB],[n_v],[appro_action])))

# @profile(precision=4, stream=memory_file)
class VEC_env:

    def __init__(self,lane,vehGenRate,slotT,VehSpeed,rsuW,TaskGenRate,TaskSize,TransmissionRange,gridW,node_feature_size,out_feature
                 ,p,DRL_model,maxP,slideW,w,k,penalty_factor,powerFactor,param_noise_var,critic_noise_var , rateDecay,reward_scale,penalty_multiplier=1.0,plot=False):
        super().__init__()
        self.episode = 0
        self.slotT = slotT
        self.lane = lane
        self.gridW = gridW
        self.plot = plot
        self.maxP = maxP
        self.penalty_factor = penalty_factor
        self.penalty_multiplier = penalty_multiplier  # 存储penalty倍数参数
        self.powerFactor = powerFactor
        self.slot = 0
        self.record_veh_num = 0
        self.VehGenRate = np.array(vehGenRate)
        self.VehSpeed = VehSpeed
        self.GenNextSlot = np.random.poisson(lam=self.VehGenRate/self.slotT, size=self.lane)
        self.Vehicle = []
        self.maxRoadLen = 2 * rsuW
        self.rsuX = rsuW

        # self.rsuY = -10
        self.rsuY = 0

        self.meanAOI = 0.0
        self.DRL_model = DRL_model
        self.rateDecay = rateDecay
        self.param_noise_var =  param_noise_var
        self.critic_noise_var = critic_noise_var
        self.uploadveh = 0
        self.globalaggreNumber = 0
        self.v2vlink = []
        self.TaskGenRate = TaskGenRate / slotT
        self.TransmissionRange = TransmissionRange      #
        self.TaskSize = TaskSize
        # GBB -----------------------------------------------------------------------------------------
        self.p = p
        self.node_feature_size = node_feature_size
        self.out_feature_size = out_feature
        self.node = torch.tensor(np.zeros((4 * (int(self.maxRoadLen/self.gridW)),self.node_feature_size)), dtype=torch.float)
        self.last_node = self.node

        self.edge = torch.tensor([[], []],dtype = torch.long)
        self.train_mask =  np.zeros(4 * (int(self.maxRoadLen/self.gridW)))
        self.lastreward = 0
        self.reward = 0
        self.node_loss = self.reward
        self.slideReward = 0
        self.penalty = 0
        self.w = w
        self.k = k
        self.slideW = slideW
        # self.slideR = self.slidereward()
        self.generateVeh()

        V2Vlink_indx = self.V2Vlink()
        self.updateGrapth(V2Vlink_indx)
        self.getModelsize()
        self.uploadveh_channel = []
        self.updateSINR()
        # self.destroyveh = deque(maxlen=20)
        self.destroyAOI = []
        self.alpha = torch.tensor([0], dtype=torch.float32)


        # self.calculateAOI(0,0)


        # init()
    def reset(self,episode):
        for veh in self.Vehicle:
            idx = self.Vehicle.index(veh)
            del self.Vehicle[idx]
            del veh.model.memory
            del veh.model
            del veh

        self.lastreward = 0
        self.reward = 0
        self.node_loss = self.reward
        self.slideReward = 0
        self.penalty = 0
        self.GCN_loss = 0.0
        self.meanAOI = 0.0
        self.total_GCN_loss = 0.0
        self.slot = 0
        self.record_veh_num = 0
        self.uploadveh = 0
        # self.globalaggreNumber = 0
        self.v2vlink = []
        # self.destroyveh = []
        # self.destroyAOI = []
        self.Vehicle = []
        self.uploadveh_channel = []
        # self.calculateAOI(0,0)
        self.GenNextSlot = np.random.poisson(lam=self.VehGenRate/self.slotT, size=self.lane)
        self.node = torch.tensor(np.zeros((4 * (int(self.maxRoadLen/self.gridW)),self.node_feature_size)), dtype=torch.float)
        self.last_node = self.node
        self.edge = torch.tensor([[], []],dtype = torch.long)
        self.train_mask =  np.zeros(4 * (int(self.maxRoadLen/self.gridW)))
        # self.slideR = self.slidereward()
        self.generateVeh()
        # self.resetVeh(episode)
        V2Vlink_indx = self.V2Vlink()

        self.updateGrapth(V2Vlink_indx)

        self.updateSINR()




    def slidereward(self):

        self.slidecount = 0
        return np.zeros((self.slideW,1))


    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def calculate_size_of_parameters(self,model):
        num_parameters = self.count_parameters(model) * 4 # Assuming 4 bytes per parameter (float32)
        return num_parameters * 8  # bits

    def getModelsize(self):

        self.model_total_size = self.calculate_size_of_parameters(self.DRL_model.policy) + \
                                self.calculate_size_of_parameters(self.DRL_model.qf1) + \
                                self.calculate_size_of_parameters(self.DRL_model.qf2) + \
                                self.calculate_size_of_parameters(self.DRL_model.target_qf1) + \
                                self.calculate_size_of_parameters(self.DRL_model.target_qf2)

        print("The DRL model parameter bits size is {} MB.".format(self.model_total_size/MB))


    def resetVeh(self,episode):
        x =  np.random.choice(np.arange(0.0, 200.0, 0.01), size=self.vehNum, replace=False)
        lane_idx = np.random.randint(low=0, high=4, size=self.vehNum)

        # for i in range(self.vehNum):
        veh_i = 0
        alpha = self.alpha
        for veh in self.Vehicle:
            # self.record_veh_num += 1
            Speed = self.VehSpeed[lane_idx[veh_i]] * 1000 / 3600
            node = int(lane_idx[veh_i] * int(self.maxRoadLen / self.gridW) + x[veh_i] // self.gridW)
            dis = np.sqrt((self.rsuX - x[veh_i]) ** 2 + (self.rsuY - 3.5 / 2 - lane_idx[veh_i] * 3.5) ** 2 + Hight_RSU ** 2)
            veh.x = x[veh_i]
            veh.lane = lane_idx[veh_i]
            veh.VehSpeed = Speed
            veh.dis = dis
            veh.node = node

            veh.reset(copy.deepcopy(self.DRL_model))



            veh.model.update_alpha(alpha.detach().numpy(), episode)

            # if episode == 0 :
            #     veh.model.log_alpha = ptu.zeros(1, requires_grad=True)
            #     veh.model.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr, )
            # else:
            #     veh.model.log_alpha = torch.log(veh.model.alpha)

            veh_i+=1
            #
            # veh = VEC_veh(x[i],lane_idx[i], Speed, self.slotT, self.rsuX, self.rsuY, dis, self.TaskGenRate, self.TaskSize,
            #                        self.TransmissionRange,node,copy.deepcopy(self.DRL_model),
            #                        self.record_veh_num,reward=self.reward,maxp=self.maxP,
            #                        param_noise_var = self.param_noise_var,critic_noise_var = self.critic_noise_var )

            # self.Vehicle.append(veh)


    def generateVeh(self):

        for lane_idx in range(self.lane):
            if self.GenNextSlot[lane_idx] <= 0:
                self.record_veh_num += 1
                Next_slots = np.random.poisson(lam=self.VehGenRate[lane_idx], size=1) / self.slotT
                Speed = self.VehSpeed[lane_idx] * 1000 / 3600
                self.GenNextSlot[lane_idx] = Next_slots[0]
                dis = np.sqrt((self.rsuX - 0) ** 2 + (self.rsuY - 3.5 / 2 - lane_idx * 3.5) ** 2 + Hight_RSU ** 2)
                node = int(lane_idx * int(self.maxRoadLen / self.gridW) + 0 // self.gridW)
                # generate new vehicle with the latest global model
                veh = VEC_veh(lane_idx, Speed, self.slotT, self.rsuX, self.rsuY, dis, self.TaskGenRate, self.TaskSize,
                              self.TransmissionRange, node, copy.deepcopy(self.DRL_model), self.record_veh_num,
                              reward=self.reward)

                veh.model.update_alpha(self.alpha.detach().numpy(), episode=self.episode)

                self.Vehicle.append(veh)


        # for lane_idx in range(self.lane):
        #     if self.GenNextSlot[lane_idx] <= 0:
        #         self.record_veh_num += 1
        #         Next_slots = np.random.poisson(lam=self.VehGenRate[lane_idx], size=1)/self.slotT
        #         Speed = self.VehSpeed[lane_idx] * 1000/3600
        #         self.GenNextSlot[lane_idx] = Next_slots[0]
        #         dis = np.sqrt((self.rsuX - 0) ** 2 + (self.rsuY - 3.5/2 - lane_idx * 3.5) ** 2 + Hight_RSU ** 2)
        #         node = int(lane_idx * int(self.maxRoadLen / self.gridW) + 0 // self.gridW)
        #         # generate new vehicle with the latest global model
        #         veh = VEC_veh(lane_idx,Speed,self.slotT,self.rsuX,self.rsuY,dis,self.TaskGenRate,self.TaskSize,
        #                       self.TransmissionRange,node,copy.deepcopy(self.DRL_model),
        #                       self.record_veh_num,reward=self.reward,maxp=self.maxP,
        #                       param_noise_var = self.param_noise_var,critic_noise_var = self.critic_noise_var )
        #         self.Vehicle.append(veh)


    def calculateAOI(self,Transimission_AOI,Transimission_veh):
        self.meanAOI = 0
        self.slideAOI = 0
        num_task = Transimission_veh
        for veh in self.Vehicle:
            num_task += len(veh.queue)
            if veh.queue:
                if veh.Transimission_AOI is not None:
                    # 修复：确保AOI_list不为空，避免nan
                    aoi_mean = np.mean(veh.AOI_list) if veh.AOI_list else 0.0
                    self.meanAOI += aoi_mean + veh.Transimission_AOI
                else:
                    # 修复：确保AOI_list不为空，避免nan
                    aoi_mean = np.mean(veh.AOI_list) if veh.AOI_list else 0.0
                    self.meanAOI += aoi_mean
            # for task in veh.queue:
            #     self.meanAOI += task.AOI

            # self.meanAOI += np.sum(veh.AOI_list)

        if len(self.Vehicle)>0:
            self.meanAOI = self.meanAOI/len(self.Vehicle)
        else:
            self.meanAOI = 0 # 当没有车辆时，设置默认AOI值
        
        # 限制AOI范围，防止极大值
        # self.meanAOI = np.clip(self.meanAOI, 0.1, 1000)
        
        #     # print("测试AOI：",self.meanAOI,Transimission_AOI,len(self.Vehicle),Transimission_veh)
        #     self.meanAOI = self.meanAOI+Transimission_AOI
        #     # self.meanAOI /= len(self.Vehicle)


        # idx = self.slidecount % self.slideW
        # self.slideR[idx] = self.meanAOI


        # if self.slidecount / self.slideW < 1:
        #     self.slideAOI = np.mean(self.slideR[:idx+1])
        # else:
        #     self.slideAOI = np.mean(self.slideR)
            # print("slide reward : ", self.slideAOI)
        # self.slidecount += 1

        self.slideAOI = 0

        return self.meanAOI,self.slideAOI


    def updateVehLoc(self):
        for veh in self.Vehicle:
            veh.x += veh.v * self.slotT
            veh.dis = np.sqrt((self.rsuX - veh.x) ** 2 + (self.rsuY - veh.y) ** 2 + Hight_RSU**2)
            veh.Loc = np.array([veh.x, veh.y, 0])
            veh.updateChannel()
            veh.GenerateTask()
            veh.node = int(veh.lane * int(self.maxRoadLen/self.gridW) + veh.x//self.gridW)
            veh.GenTaskNextSlot -= 1


    def destroyVeh(self):
        penalty = 0
        for veh in self.Vehicle.copy():
            if veh.x >= self.maxRoadLen:
                # self.destroyveh.append(veh.record_id)
                # self.destroyAOI.append(len(veh.queue))
                if veh.queue:
                    # penalty += np.sum(veh.AOI_list)
                    # 修复：确保AOI_list不为空，避免nan
                    aoi_mean = np.mean(veh.AOI_list) if veh.AOI_list else 0.0
                    penalty += aoi_mean
                self.Vehicle.remove(veh)
        if self.Vehicle:
            self.penalty += penalty/len(self.Vehicle) # 消失的对还存在的影响


    def getuploadPower(self,allPowerGain,uploadveh):
        if uploadveh == 0:
            return [0]
        Model_size = self.model_total_size
        exp_input = Model_size/W*np.log(2)
        a = np.exp(exp_input)-1
        
        # 只在结果为NaN或Inf时打印
        if np.isnan(a) or np.isinf(a):
            print(f"⚠️⚠️⚠️[DEBUG] compute_upload_rate: Model_size = {Model_size}, W = {W}, exp_input = {exp_input}, a = {a}")
            import pdb
            pdb.set_trace()
        G = a * np.array(self.uploadveh_channel) * np.ones((uploadveh,uploadveh)) - a * np.array(self.uploadveh_channel) * np.eye(uploadveh) \
                - np.array(self.uploadveh_channel) * np.eye(uploadveh)
        Y = (noise_var + allPowerGain) * a * np.ones((uploadveh,1))
        p = np.matmul(np.linalg.inv(G),-Y)
        return sum(p)

    # def updateSINR(self):
    #     allPowerGain = 0
    #     for veh in self.Vehicle:
    #         # update current vehicular state
    #         # s = veh.updatestate()
    #         # veh.s0 = s
    #         veh.Translate = False
    #         # _, action = veh.model.select_action(veh.s0,self.maxP,veh.record_id)
    #         _, action, appro_action = veh.model.select_action(veh.s0,self.maxP, veh.record_id,len(self.Vehicle))
    #
    #         veh.action = action[0]
    #         veh.appro_action = appro_action[0]
    #
    #         allPowerGain += veh.action * veh.channel
    #     uploadpower = self.getuploadPower(allPowerGain,self.uploadveh)
    #
    #     for veh in self.Vehicle:
    #         veh.compute_sinr(allPowerGain,uploadpower[0])
    #         # print("ID : ", veh.record_id, " , sinr : ",veh.sinr," , last sinr : ", veh.lastSINR ,"rate : ", veh.v2iRate, " action : ",veh.action)
    #
    #     # Transmission Task based on the v2i rate
    #     self.reward = self.Translate_task()
    #     # update each vehicular memory
    #     self.updateMemory(appro_action)
    #     # self.updateMemory()

    def updateSINR(self):
        allPowerGain = 0
        Channel_list = np.zeros((len(self.Vehicle)))

        for veh in self.Vehicle:
            # update current vehicular state
            # s = veh.updatestate()
            # veh.s0 = s
            veh.Translate = False

            _, action_pre, action = veh.model.select_action(veh.s0,veh.record_id)
            # _, action, appro_action = veh.model.select_action(veh.s0,self.maxP, veh.record_id,len(self.Vehicle))

            if np.isnan(action):
                print(veh.record_id, action)
                import pdb
                pdb.set_trace()

            # veh.appro_action = appro_action[0]
            veh.action_pre = action_pre.item()
            veh.action = action.item()
            allPowerGain += veh.action * veh.channel
        uploadpower = self.getuploadPower(allPowerGain,self.uploadveh)

        veh_i = 0
        for veh in self.Vehicle:
            veh.compute_sinr(allPowerGain,uploadpower[0])
            Channel_list[veh_i] = veh.channel
            veh_i += 1
            # print("ID : ", veh.record_id, " , sinr : ",veh.sinr," , last sinr : ", veh.lastSINR ,"rate : ", veh.v2iRate, " action : ",veh.action)

        # Transmission Task based on the v2i rate
        self.lastreward = self.reward
        self.reward ,self.slideReward = self.Translate_task()
        # update each vehicular memory
        # self.updateMemory(appro_action)
        for veh in self.Vehicle:
            veh.env_AOI = self.reward + self.penalty
        self.updateMemory(Channel_list)


    def Translate_task(self):
        Transimission_AOI = 0
        Transimission_veh = 0
        for veh in self.Vehicle:
            veh.Transimission_AOI = None
            # print(veh.record_id, veh.AOI_list)
            # Translate first task
            veh.Transimission_tasksize = None
            if len(veh.queue)>0:
                First_task = veh.queue[0]
                # print(veh.record_id, veh.v2iRate,First_task.size)
                if veh.v2iRate >= First_task.size and veh.v2iRate > 1e-10:  # 防止除零
                    veh.Translate = True
                    transmission_aoi = First_task.size / veh.v2iRate
                    # 限制传输AOI范围，防止极大值
                    transmission_aoi = np.clip(transmission_aoi, 0.1, 1000)
                    Transimission_AOI += transmission_aoi
                    Transimission_veh += 1
                    veh.Transimission_AOI = transmission_aoi
                    veh.Transimission_tasksize = First_task.size
                    del First_task
                    del veh.queue[0]
                    # OK_task = veh.queue.pop(0)
                for task in veh.queue:
                    task.AOI +=1

            veh.collectAOI()
            if  veh.Transimission_AOI is not None :
                veh.AOI_list.append(veh.Transimission_AOI)
            # print(veh.record_id, veh.AOI_list)
        # print("Transmission : ",Transimission_AOI)
        mean_AOI ,  slideAOI = self.calculateAOI(Transimission_AOI,Transimission_veh)

        return mean_AOI, slideAOI


    def calculatelink(self,Vehicle,Cal_Training_time):
        vehX_RowRepeat = []
        vehY_RowRepeat = []
        vehX = []
        vehY = []
        vehSpeed = []
        for veh in Vehicle:
            if Cal_Training_time == True:
                Training_time = veh.trainingTimeslot
            else:
                Training_time = 0
            vehX.append(veh.x + veh.v * Training_time)
            vehY.append(veh.y)
            vehX_RowRepeat.append([veh.x + veh.v * Training_time] * len(Vehicle))
            vehY_RowRepeat.append([veh.y] * len(Vehicle))
            vehSpeed.append(veh.v)
        vehX_ColRepeat = [vehX] * len(Vehicle)
        vehY_ColRepeat = [vehY] * len(Vehicle)
        v2vMatrix = np.sqrt((np.array(vehX_RowRepeat) - np.array(vehX_ColRepeat)) ** 2 + \
                                 (np.array(vehY_RowRepeat) - np.array(vehY_ColRepeat)) ** 2)
        return v2vMatrix,vehSpeed


    def V2Vlinkmatrix(self):
        if len(self.Vehicle)>=2:
            self.v2vcurrentMatrix,self.vehSpeedList= self.calculatelink(self.Vehicle,Cal_Training_time = False)
            self.v2vcurentlink    = self.v2vcurrentMatrix < self.TransmissionRange
            self.v2vfutrueMatrix,_= self.calculatelink(self.Vehicle, Cal_Training_time=True)
            self.v2vfuturelink   = self.v2vfutrueMatrix < self.TransmissionRange
            # self.v2vlink = self.v2vcurentlink  * self.v2vfuturelink
            self.v2vlink = self.v2vcurentlink


    def V2Vlink(self):
        self.V2Vlinkmatrix()
        V2Vlink_indx = [[],[]]
        if len(self.v2vlink)>1:
            v2vlink_tril = np.triu(self.v2vlink,1)
            V2Vlink_indx = np.where(v2vlink_tril > 0)
        return V2Vlink_indx


    def getVehdis(self,veh1,veh2):
        return np.sqrt((veh1.x-veh2.x)**2+(veh1.y-veh2.y)**2)


    def process_edge(self,edge):
        if len(edge[0]) == 0:
            return [[],[]]
        new_edge = [[],[]]
        for i,j in zip(edge[0],edge[1]):
            if i<j:
                new_edge[0].append(i)
                new_edge[1].append(j)
            else:
                new_edge[0].append(j)
                new_edge[1].append(i)
        return new_edge


    def updateGrapth(self,V2Vlink_indx):
        self.node = torch.tensor(np.zeros((4 * (int(self.maxRoadLen / self.gridW)), self.node_feature_size)),dtype=torch.float)
        self.node_veh = {}
        edge = [[],[]]
        mask = np.zeros(4 * (int(self.maxRoadLen/self.gridW)))
        for veh in self.Vehicle:
            # self.node : 0 - int(self.maxRoadLen/gridW) : lane 0
            # self.node : 0 * int(self.maxRoadLen/gridW) + 1 - 2 * int(self.maxRoadLen/gridW) : lane 1
            # self.node : 2 * int(self.maxRoadLen/gridW) + 1 - 3 * int(self.maxRoadLen/gridW) : lane 2
            # self.node : 3 * int(self.maxRoadLen/gridW) + 1 - 4 * int(self.maxRoadLen/gridW) : lane 3
            # VehSpeed = [60, 90, 100, 120] -- [lane 0, lane 1 , lane 2, lane 3]
            node_index = int(veh.lane * int(self.maxRoadLen/self.gridW) + veh.x//self.gridW)
            self.node[node_index][0] += 1                # first  feature: vehicular number
            self.node[node_index][1] += veh.n_GlobalDRL  # third  feature: current mean global aggregation number
            #self.node[node_index][3] += veh.n_LocalDRL   # firth  feature: current mean global aggregation number
            self.node[node_index][2] += veh.model.loss.policy_loss.detach().numpy()
            self.node[node_index][3] += veh.model.loss.qf1_loss.detach().numpy()
            self.node[node_index][4] += veh.model.loss.qf2_loss.detach().numpy()
            mask[node_index] = 1
            if node_index not in self.node_veh.keys():
                self.node_veh[node_index]=[]
            self.node_veh[node_index].append(veh)
        for node in self.node:
            if node[0]!=0:
                node[1] = node[1] / node[0]
                node[2] = node[2] / node[0]
                node[3] = node[3] / node[0]
                node[4] = node[4] / node[0]
        # vehicular number >= 2 : update edge
        link_temp = []
        link_temp_len = {}
        if len(self.Vehicle)>=2:
            for veh1, veh2 in  zip(V2Vlink_indx[0], V2Vlink_indx[1]):
                node1_index = int(self.Vehicle[veh1].lane * int(self.maxRoadLen / self.gridW) + self.Vehicle[veh1].x // self.gridW)
                node2_index = int(self.Vehicle[veh2].lane * int(self.maxRoadLen / self.gridW) + self.Vehicle[veh2].x // self.gridW)
                if (node1_index != node2_index) and ( [node1_index,node2_index] not in link_temp) and ( [node2_index,node1_index] not in link_temp):
                    link_temp.append([node1_index,node2_index])
                    edge[0].append(node1_index)
                    edge[1].append(node2_index)
                if node1_index != node2_index:
                    if node2_index in link_temp_len.keys() and node1_index in link_temp_len[node2_index].keys():
                        temp_max = self.getVehdis(self.Vehicle[veh1],self.Vehicle[veh2])
                        link_temp_len[node2_index][node1_index] = max(link_temp_len[node2_index][node1_index],temp_max)
                        continue
                    if node1_index not in link_temp_len.keys():
                        link_temp_len[node1_index] = {}
                        if node2_index not in link_temp_len[node1_index].keys():
                            link_temp_len[node1_index][node2_index] = 0
                    else:
                        if node2_index not in link_temp_len[node1_index].keys():
                            link_temp_len[node1_index][node2_index] = 0
                    link_temp_len[node1_index][node2_index] = max(link_temp_len[node1_index][node2_index],
                                                                     self.getVehdis(self.Vehicle[veh1], self.Vehicle[veh2]))

        new_edge = self.process_edge(edge)
        self.edge = torch.tensor(new_edge,dtype=torch.long)
        self.train_mask = torch.tensor(mask)


        self.edge_len = link_temp_len




    def prim(self):
        # self.edge_len, self.connect_edge
        # print("Edge : ", self.edge)
        # print("Connect_edge", self.connect_edge)
        # print("Overlay_edge", self.overlay_edge)
        edges = []
        nodelist = []
        c_e = self.connect_edge.detach().numpy()
        for node1,node2 in zip(c_e[0],c_e[1]):
            nodelist.append(node1)
            nodelist.append(node2)
            if node1 in self.edge_len.keys() and node2 in self.edge_len[node1].keys():
                edges.append( (node1, node2, self.edge_len[node1][node2]))
                edges.append( (node2, node1, self.edge_len[node1][node2]))
            elif node2 in self.edge_len.keys() and node1 in self.edge_len[node2].keys():
                edges.append((node1, node2, self.edge_len[node2][node1]))
                edges.append((node2, node1, self.edge_len[node2][node1]))
            else:
                print("----------------error-----------------------")
        nodelist = list(set(nodelist))
        seen = [nodelist[0]]
        choice = []
        seen_edge = []
        # while len(seen) <= len(graph.keys()):
        while len(seen) < len(nodelist):
            for i in edges:
                if i[0] == seen[-1]:  # 把和最新添加的与i有关的边都拿出来
                    seen_edge.append(i)
            seen_edge.sort(key=lambda x: x[-1], reverse=True)  # 升序
            while 1:
                if len(seen_edge) == 0:
                    if len(seen) == len(nodelist):
                        break
                    res_nodelist = list(set(nodelist).difference(set(seen)))
                    seen.append(res_nodelist[0])
                    break
                if seen_edge[-1][1] not in seen:
                    seen.append(seen_edge[-1][1])
                    choice.append(seen_edge.pop())
                    break
                else:
                    seen_edge.pop()
        # print('-----------------------------------------------------------')
        overlay_edge = [[],[]]
        for path in choice:
            overlay_edge[0].append(path[0])
            overlay_edge[1].append(path[1])
        return overlay_edge


    def getOverlayG(self):
        """
        不使用GNN，直接基于原始边创建overlay图
        """
        # 直接使用所有边作为connect_edge（不使用GNN特征过滤）
        connect_edge = [[],[]]
        for n, m in zip(self.edge[0], self.edge[1]):
            # 直接使用所有边，或者可以根据其他标准过滤
            connect_edge[0].append(n)
            connect_edge[1].append(m)

        self.connect_edge = torch.tensor(connect_edge,dtype=torch.long)
        
        # 创建简单的Data对象用于networkx（不依赖GCN模型）
        if len(self.connect_edge[0]) > 0:
            edge_index = self.connect_edge
            x = self.node  # 使用原始节点特征
            connect_data = Data(x=x, edge_index=edge_index)
            self.connect_G = to_networkx(connect_data,to_undirected=True)
        else:
            # 如果没有边，创建一个空的图
            import networkx as nx
            self.connect_G = nx.Graph()
        
        if len(self.connect_edge[0]) == 0 :
            overlay_edge = [[],[]]
        else:
            overlay_edge = self.prim()
        new_overlay_edge = self.process_edge(overlay_edge)

        self.overlay_edge = self.connect_edge
        if len(self.connect_edge[0]) > 0:
            overlay_data = Data(x=self.node, edge_index=self.overlay_edge)
            self.overlay_data = overlay_data
            self.overlay_G = to_networkx(overlay_data, to_undirected=True)
        else:
            self.overlay_data = None
            import networkx as nx
            self.overlay_G = nx.Graph()


    def processState(self,s):
        if all(not x for x in s):
            return [0,0,0]
        mean = np.mean(s, axis=0)
        std = np.std(s, axis=0)
        # 进行均值方差归一化处理
        s_norm = (s - mean) / std
        return s_norm

    # def updateMemory(self):
    #     for veh in self.Vehicle:
    #         veh.reward = self.reward
    #         s1 = veh.updatestate(len(self.Vehicle),veh.appro_action)
    #         # s1 = self.processState(s1)
    #         if (veh.x + self.slotT * veh.v) > self.maxRoadLen:
    #             veh.done = True
    #         veh.model.memory.Addremember(np.concatenate((veh.s0, [veh.action], [-veh.reward], [-(veh.reward-veh.meanAoI)],s1,[veh.done])))
    #         print(np.concatenate((veh.s0, [veh.action], [-veh.reward], s1, [veh.done])))
    #         veh.s0 = s1

    def updateMemory(self,Channel_list):

        def f(a, b, powerFactor):
            # 防止除零
            if abs(b) < 1e-10:
                return 0
                
            x = a / b
            
            # 检查输入是否为NaN或无穷大
            if np.isnan(x) or np.isinf(x):
                return 0
                
            # 限制x的范围，防止指数运算溢出
            x = np.clip(x, -50, 50)
            
            # 防止powerFactor为0
            if abs(powerFactor) < 1e-10:
                return 0
                
            # 计算指数项，防止溢出
            exp_term = powerFactor * x
            exp_term = np.clip(exp_term, -50, 50)  # 限制指数范围
            
            result = np.exp(exp_term) / powerFactor
            
            # 只在结果为NaN或Inf时打印
            if np.isnan(result) or np.isinf(result):
                print(f"⚠️⚠️⚠️[DEBUG] f function: exp_term = {exp_term}, powerFactor = {powerFactor}, result = {result}")
                import pdb
                pdb.set_trace() 
            
            # 检查结果是否为NaN或无穷大
            if np.isnan(result) or np.isinf(result):
                import pdb
                pdb.set_trace() 
                return 0
                
            return result

        for veh in self.Vehicle:
            # veh.reward = self.reward + self.penalty
            # if veh.queue:
            #     # veh.reward = self.slideReward + np.sum(Channel_list)/veh.channel * veh.queue[0].AOI/self.w
            #     # x = np.abs(np.max((self.reward - np.mean(veh.AOI_list),np.mean(veh.AOI_list))))/(self.reward+1e-3)
            #     x =  (self.reward - np.mean(veh.AOI_list))
            #     y = f(x, (self.reward + 1e-3),self.powerFactor)
            #     # y = x * np.exp(-self.reward - np.mean(veh.AOI_list)10 * x)
            #     # y =  x ** 2 / (1 + self.k *x )
            #     # y = np.log(1+x)
            #
            #     veh.ration = y
            #     # veh.reward = self.reward + np.mean(veh.AOI_list) * len(veh.queue)-\
            #     #              (self.reward - self.lastreward) * 0.99 +  \
            #     #              y * veh.action * self.meanReward * len(self.Vehicle)
            #
            #     # veh.reward = self.reward + np.mean(veh.AOI_list)  +  self.penalty + \
            #     veh.reward =   veh.action *  (self.reward + self.penalty) / np.mean(veh.AOI_list) + self.reward
            #     # veh.reward = (self.reward + self.penalty  + np.sum(veh.AOI_list))/len(self.Vehicle)
            #     # veh.reward = self.reward  + self.penalty
            #     # print( self.slideReward , np.abs(self.reward - np.mean(veh.AOI_list))/(self.reward+1e-3) * veh.action * self.meanReward * len(self.Vehicle))
            #     # print(np.sum(Channel_list) / veh.channel *self.slideReward / self.w,
            #     #       np.abs(self.reward - np.mean(veh.AOI_list)) * veh.action )
            #     # print(self.slideReward, np.sum(Channel_list) / veh.channel * (self.slideReward - np.mean(veh.AOI_list)) * veh.action / self.w)
            #     #veh.reward = np.sum(Channel_list) / veh.channel * self.slideReward
            #     # print("ooooooooooooooooooooooooooooooooooooooooooo")
            #     # print(self.slideReward,Channel_list, np.sum(Channel_list), veh.channel, veh.queue[0].AOI,

            #     #       np.sum(Channel_list) / veh.channel * veh.queue[0].AOI)

            # veh.reward = self.reward
            # if np.mean(veh.AOI_list) ==0 :
            #     print(0)
            # a = np.mean(veh.AOI_list)
            if veh.queue and (np.mean(veh.AOI_list) > 1e-6):  # 防止除零
                # 限制奖励和AOI的范围，防止除法产生极大值
                # safe_reward = np.clip(self.reward, -100, 100)
                safe_aoi = max(np.mean(veh.AOI_list), 1e-6)  # 确保AOI不为零
                
                x = self.reward / safe_aoi
                # 限制x的范围，防止指数运算溢出
                x = np.clip(x, -10, 10)
                
                veh.reward = self.reward + self.penalty * self.penalty_multiplier + veh.action * (1 + x)
            else:
                # 当没有队列或AOI为0时，使用简化的奖励计算
                # safe_reward = np.clip(self.reward, -100, 100)
                veh.reward = self.reward + self.penalty * self.penalty_multiplier + veh.action * (1 + self.reward)
            
            # 最终裁剪奖励值，防止极大值
            veh.reward = np.clip(veh.reward, 0, 666)

            # if math.isnan(veh.reward):
            #
            #     print(0)
            #
            # if math.isinf(veh.reward):
            #     print(0)
            # if (veh.reward == np.NAN) or veh.reward == np.inf:
            #     print("0")
            #     # veh.reward = self.slideReward
            #     # veh.reward = 0
            #     # x = np.abs(self.reward - 0)/(self.reward+1e-3)
            #     # y = x * np.exp(-10 * x)
            #     # y =  x ** 2 / (1 + self.k * x )
            #     # y = np.log(1 + x)
            #     y = f(np.abs(self.reward - 0),
            #           (self.reward + 1e-3),self.powerFactor)
            #     # veh.reward = y * veh.action * self.meanReward * len(self.Vehicle)
            #     veh.ration = y
            #     # veh.reward = ( self.reward -\
            #     #              (self.reward - self.lastreward) * 0.99 +  \
            #     #              y  * self.meanReward  )* veh.action * len(self.Vehicle)
            #      veh.reward = veh.action * (self.reward  +  self.penalty)
            #     # veh.reward = (self.reward + self.penalty)/len(self.Vehicle)
            #     # veh.reward = self.reward + self.penalty
            #     # print(np.sum(Channel_list) / veh.channel *self.slideReward / self.w,
            #     #       np.abs(self.reward - 0) * veh.action )

            s1 = veh.updatestate(len(self.Vehicle))
            # s1 = self.processState(s1)
            if (veh.x + self.slotT * veh.v) >= self.maxRoadLen:
                veh.done = True

            # print(veh.record_id,veh.s0, [veh.action], [-veh.reward], s1, [veh.done])
            # 准备数据并检查是否有NaN值
            train_data = np.concatenate((veh.s0, [veh.action_pre], [-veh.reward], s1, [veh.done]))
            if not (np.isnan(train_data).any() or np.isinf(train_data).any()):
                veh.model.memory.Addremember(train_data)
            else:
                print("⚠️ ⚠️ Warning: NaN or Inf detected in SAC training data, skipping storage")    
                import pdb
                pdb.set_trace()


            veh.s0 = s1
        self.penalty = self.penalty * self.penalty_factor


    def Trainlocalmodel(self):
        # birth : 146ni5 812 857 715
        # len(memory) > self.connter_size and Training time
        for veh in self.Vehicle:
            veh.need_local_aggregate = False
            if (veh.model.memory.memory_counter > veh.update_counter) and ( veh.slot % veh.trainingTimeslot == 0):
                print("slot : ",self.slot,"ID ： ", veh.record_id, " ------------------------------Training----------------------------------------")
                # print("slot : ",self.slot,"ID ： ", veh.record_id, veh.model.actor.state_dict()['l1.weight'][:3][0])
                # veh.model.update(veh.Training_number,veh.q1_aggreNoise,veh.q2_aggreNoise)
                "返回3个梯度"
                veh.model.update(veh.Training_number)
                #print(veh.model.actor.state_dict()['l1.weight'][:3][0])
                #print("ID ： ", veh.record_id," Training---------------------------------------------------------------------------")
                veh.need_local_aggregate = True
                veh.global_aggregate = True
                veh.trainNumber += 1


    def getNodelinkMatrix(self):
        node_link = {}
        for edge in self.overlay_G.edges:
            node1 = edge[0]
            node2 = edge[1]
            if node1 not in node_link.keys():
                node_link[node1]= []
            if node2 not in node_link.keys():
                node_link[node2]= []
            node_link[node1].append(node1)
            node_link[node1].append(node2)
            node_link[node2].append(node1)
            node_link[node2].append(node2)
            node_link[node1] = list(set(node_link[node1]))
            node_link[node2] = list(set(node_link[node2]))
        return node_link


    def model_convert(self,model_grad):

        return torch.flatten(model_grad).tolist()


    def get_losses(self,veh):

        return veh.model.getGradient(veh.Training_number)


    def get_gradient(self,veh):
        losses = veh.model.getGradient(veh.Training_number)
        #
        # gradient = []

        # veh.local_gradients_policy = [param.grad for param in veh.model.policy.parameters()]
        # veh.local_gradients_q1 = [param.grad for param in veh.model.qf1.parameters()]
        # veh.local_gradients_q2 = [param.grad for param in veh.model.qf2.parameters()]
        return losses
        # return [param.grad for param in model.policy.parameters()],\
        #        [param.grad for param in model.qf1.parameters()],\
        #        [param.grad for param in model.qf2.parameters()],\
        #        model.alpha_loss

        # for param in veh.model.policy.parameters():
        #     gradient = gradient + self.model_convert(param.grad)
        #     #print(self.model_convert(param.grad))
        # # print(max(gradient))
        # # print("critic")
        # # gradient = []
        # for param in veh.model.qf1.parameters():
        #
        #     gradient = gradient + self.model_convert(param.grad)
        #
        #     for param in veh.model.qf2.parameters():
        #         gradient = gradient + self.model_convert(param.grad)
        #     #print(self.model_convert(param.grad))
        # # print(max(gradient))
        # # for param in veh.model.actor_target.parameters():
        # #     gradient = gradient + self.model_convert(param.grad)
        # # for param in veh.model.critic_target.parameters():
        # #     gradient = gradient + self.model_convert(param.grad)
        # return gradient,0


    def dot_sum(self,K, L):
        # print(K)
        # print(L)
        # a0 = [i[0] * i[1] for i in zip(K, L)]
        # # print(a0)
        # a0 = np.array(a0)
        # indx = np.argmax(a0)
        # cc = K[indx]
        # dd = L[indx]
        # s_a0 = sum(a0)
        # a = sum(i[0] * i[1] for i in zip(K, L))
        # b = round(sum(i[0] * i[1] for i in zip(K, L)), 2)
        # if math.isinf(b):
        #     print(0)
        return round(sum(i[0] * i[1] for i in zip(K, L)), 2)

    def get_relation(self, pre_grad, update_grad):
        relation = self.dot_sum(pre_grad, update_grad)

        # pre_grad = torch.tensor(pre_grad)
        # update_grad = torch.tensor(update_grad)
        # relation2 = torch.dot(pre_grad.view(-1), update_grad.view(-1))

        return relation


    def updateModel(self, model, gradient, rate):
        for p, grad in zip(model.parameters(), gradient):
            p.data -= rate * grad


    def localAsyFederated(self):
        """
        本地异步联邦聚合 - 使用简单的平均聚合，不依赖GNN
        """
        node_link = self.getNodelinkMatrix()

        self.fedLoss = 0

        for veh in self.Vehicle:
            if veh.need_local_aggregate == False:
                continue

            # 获取需要聚合的车辆列表（基于图连接的邻居节点）
            aggre_veh = [veh]  # 包含自身
            
            # 根据节点链接关系获取邻居车辆
            if node_link and veh.node in node_link.keys():
                aggregate_node_list = node_link[veh.node]
                for aggregate_node in aggregate_node_list:
                    if aggregate_node in self.node_veh:
                        aggregate_veh_list = self.node_veh[aggregate_node]
                        for veh2 in aggregate_veh_list:
                            if veh2 == veh:
                                continue
                            if veh2.trainNumber == 0:
                                continue
                            if veh2 not in aggre_veh:
                                aggre_veh.append(veh2)
                                veh.n_LocalDRL += 1

            # 使用均匀权重进行平均聚合
            num_vehicles = len(aggre_veh)
            if num_vehicles == 0:
                continue
                
            uniform_weight = 1.0 / num_vehicles
            mean_weight = [uniform_weight] * num_vehicles

            veh.aggre_weight = mean_weight
            veh.aggre_veh = aggre_veh

            with torch.no_grad():
                # 获取模型的权重字典的键
                policy_keys = None
                q1_keys = None
                q2_keys = None
                
                # 初始化聚合权重
                avg_weights_policy = None
                avg_weights_q1 = None
                avg_weights_q2 = None

                # 对所有车辆进行均匀加权平均
                for idx, veh_agg in enumerate(aggre_veh):
                    policy_weights_veh = veh_agg.model.policy.state_dict()
                    q1_weights_veh = veh_agg.model.qf1.state_dict()
                    q2_weights_veh = veh_agg.model.qf2.state_dict()
                    
                    # 获取键（只需要第一次）
                    if policy_keys is None:
                        policy_keys = policy_weights_veh.keys()
                        q1_keys = q1_weights_veh.keys()
                        q2_keys = q2_weights_veh.keys()
                        # 初始化
                        avg_weights_policy = {key: torch.zeros_like(policy_weights_veh[key]) for key in policy_keys}
                        avg_weights_q1 = {key: torch.zeros_like(q1_weights_veh[key]) for key in q1_keys}
                        avg_weights_q2 = {key: torch.zeros_like(q2_weights_veh[key]) for key in q2_keys}
                    
                    # 累加权重
                    weight = mean_weight[idx]
                    avg_weights_policy = {key: avg_weights_policy[key] + policy_weights_veh[key] * weight 
                                        for key in policy_keys}
                    avg_weights_q1 = {key: avg_weights_q1[key] + q1_weights_veh[key] * weight 
                                     for key in q1_keys}
                    avg_weights_q2 = {key: avg_weights_q2[key] + q2_weights_veh[key] * weight 
                                     for key in q2_keys}
                
                # 更新车辆模型（只更新critic，不更新policy）
                # veh.model.policy.load_state_dict(avg_weights_policy)
                veh.model.qf1.load_state_dict(avg_weights_q1)
                veh.model.qf2.load_state_dict(avg_weights_q2)

            veh.aggre_veh = []

            veh.global_aggregate = True
            veh.n_GlobalDRL += 1


    def globalAsyFederated(self):
        weight = []
        # generate upload model power
        self.uploadveh = 0
        self.uploadveh_channel = []

        aggre_veh = []
        for veh in self.Vehicle:
            if (veh.x + self.slotT * veh.v) >= self.maxRoadLen:
                aggre_veh.append(veh)
                weight.append(1)  #
                self.uploadveh += 1
                self.uploadveh_channel.append(veh.channel)

        if len(aggre_veh) == 0:
            return 0

        with torch.no_grad():
            policy_weights = [veh.model.policy.state_dict() for veh in aggre_veh]
            q1_weights = [veh.model.qf1.state_dict() for veh in aggre_veh]
            q2_weights = [veh.model.qf2.state_dict() for veh in aggre_veh]
            # 获取模型的权重字典的键
            policy_keys = policy_weights[0].keys()
            q1_keys = q1_weights[0].keys()
            q2_keys = q2_weights[0].keys()
            # 对所有模型的相同权重进行平均
            avg_weights_policy = {key: sum(w[key] for w in policy_weights) / len(policy_weights) for key in policy_keys}
            avg_weights_q1 = {key: sum(w[key] for w in q1_weights) / len(q1_weights) for key in q1_keys}
            avg_weights_q2 = {key: sum(w[key] for w in q2_weights) / len(q2_weights) for key in q2_keys}
            # 创建一个新的模型用于存储平均权重

            # 将平均权重设置给新模型
            self.DRL_model.policy.load_state_dict(avg_weights_policy)
            self.DRL_model.qf1.load_state_dict(avg_weights_q1)
            self.DRL_model.qf2.load_state_dict(avg_weights_q2)
            self.DRL_model.target_qf1.load_state_dict(avg_weights_q1)
            self.DRL_model.target_qf2.load_state_dict(avg_weights_q2)

        aggre_alpha = torch.tensor([0], dtype=torch.float32)
        w_alpha = 1 / (sum(weight))
        
        for veh in aggre_veh:
            aggre_alpha += torch.tensor(w_alpha, dtype=torch.float32) * veh.model.alpha

        self.alpha = aggre_alpha
        self.globalaggreNumber += 1




    def checkgradient(self):
        for veh in self.Vehicle:
            print("------------------------------------------------------------------------------------------------")
            print(veh.record_id)
            a = []
            for name, param in veh.model.policy.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    a.append(grad_norm)

                    print("policy : ", f'{name} grad norm: {param.grad.norm()}', grad_norm)
                else:
                    print("policy : ", f'{name} grad norm:',"no")

            for name, param in veh.model.qf1.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    a.append(grad_norm)
                    print("qf1 : ", f'{name} grad norm: {param.grad.norm()}', grad_norm)
                else:
                    print("qf1 : ", f'{name} grad norm: ', "no")
            for name, param in veh.model.qf2.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    a.append(grad_norm)
                    print("qf2 : ", f'{name} grad norm: {param.grad.norm()}', grad_norm)
                else:
                    print("qf2 : ", f'{name} grad norm: ', "no")
            for name, param in veh.model.target_qf1.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    a.append(grad_norm)
                    print("target_qf1 : ", f'{name} grad norm: {param.grad.norm()}', grad_norm)
                else:
                    print("target_qf1 : ", f'{name} grad norm: ', "no")
            for name, param in veh.model.target_qf2.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    a.append(grad_norm)
                    print("target_qf2 : ", f'{name} grad norm: ', grad_norm)
                else:
                    print("target_qf2 : ", f'{name} grad norm: ', "no")
            for grad in a:
                if grad > 1000:
                    print("vvvvvvvvvvvvvvvvv")

    def generateNoise(self):
        for veh in self.Vehicle:
            if veh.need_local_aggregate == False:
                continue
            s0,act,_ = veh.model.select_action(veh.s0,veh.record_id)
            # veh.actorNoise = torch.normal(mean=0, std=self.noise_var, size= act.shape)

            q1 = veh.model.qf1(s0, act)
            q2 = veh.model.qf1(s0, act)

            veh.q1Noise = torch.normal(mean=0, std=self.critic_noise_var, size=(256,*q1.shape))
            veh.q2Noise = torch.normal(mean=0, std=self.critic_noise_var, size=(256,*q2.shape))
            param_shapes = [param.shape for param in veh.model.policy.parameters()]
            veh.actorparamNoise = [torch.normal(mean=0, std=self.param_noise_var, size=shape) for shape in param_shapes]
            param_shapes = [param.shape for param in veh.model.qf1.parameters()]
            veh.q1paramNoise = [torch.normal(mean=0, std=self.param_noise_var, size=shape) for shape in param_shapes]
            param_shapes = [param.shape for param in veh.model.qf2.parameters()]
            veh.q2paramNoise = [torch.normal(mean=0, std=self.param_noise_var, size=shape) for shape in param_shapes]

    def aggregateNoise(self):

        node_link = self.getNodelinkMatrix()

        for veh in self.Vehicle:
            if veh.need_local_aggregate == False:
                continue
            weight, aggre_veh = self.getWeight(veh, node_link)

            if all(elem == 0 for elem in weight):
                weight = [1]
                aggre_veh = []

            # 修复：防止除零错误，确保权重归一化安全
            if np.sum(weight) != 0:
                veh.aggre_weight = weight/np.sum(weight)
            else:
                veh.aggre_weight = np.ones_like(weight) / len(weight)  # 默认均匀权重
            veh.aggre_veh = aggre_veh

            # veh.actor_aggreNoise = veh.aggre_weight[0] * veh.actorNoise
            veh.q1_aggreNoise = veh.aggre_weight[0] * veh.q1Noise
            veh.q2_aggreNoise = veh.aggre_weight[0] * veh.q2Noise
            veh.actor_aggreParamNoise = [veh.aggre_weight[0] * param for param in veh.actorparamNoise]
            veh.q1_aggreParamNoise = [ veh.aggre_weight[0] * param for param in veh.q1paramNoise ]
            veh.q2_aggreParamNoise = [veh.aggre_weight[0] * param for param in veh.q2paramNoise]

            i = 0
            for veh2 in aggre_veh:
                i += 1
                # veh.actor_aggreNoise += veh.aggre_weight[i] * veh2.actorNoise
                veh.q1_aggreNoise += veh.aggre_weight[i] * veh2.q1Noise
                veh.q2_aggreNoise += veh.aggre_weight[i] * veh2.q2Noise
                veh.actor_aggreParamNoise += [veh.aggre_weight[i] * param for param in veh2.actorparamNoise]
                veh.q1_aggreParamNoise += [veh.aggre_weight[i] * param for param in veh2.q1paramNoise]
                veh.q2_aggreParamNoise += [veh.aggre_weight[i] * param for param in veh2.q2paramNoise]

    def flattenParam(self,param):
        return torch.cat([p.flatten() for p in param])

    def addparamNoise(self):
        self.resistLoss = 0
        loss = []
        theta_global_model_policy = self.flattenParam(self.DRL_model.policy.parameters())
        theta_global_model_q1 = self.flattenParam(self.DRL_model.qf1.parameters())
        theta_global_model_q2 = self.flattenParam(self.DRL_model.qf2.parameters())
        aggre_num_veh = 0
        for veh in self.Vehicle:
            if veh.need_local_aggregate:
                aggre_num_veh+=1
                theta_veh_policy = self.flattenParam(veh.model.policy.parameters())
                theta_veh_q1  = self.flattenParam(veh.model.qf1.parameters())
                theta_veh_q2 = self.flattenParam(veh.model.qf2.parameters())
                cos_sim = torch.nn.functional.cosine_similarity(theta_veh_policy,
                                                                theta_global_model_policy, dim=0) + \
                          torch.nn.functional.cosine_similarity(theta_veh_q1,
                                                                theta_global_model_q1, dim=0) + \
                          torch.nn.functional.cosine_similarity(theta_veh_q2,
                                                                theta_global_model_q2, dim=0)

                for param, noise in zip( veh.model.policy.parameters(), veh.actor_aggreParamNoise):

                    param.data += noise

                for param, noise in zip(veh.model.qf1.parameters(), veh.q1_aggreParamNoise):
                    param.data += noise

                for param, noise in zip(veh.model.qf1.parameters(), veh.q2_aggreParamNoise):
                    param.data += noise

                theta_veh_policy_plus_noise = self.flattenParam(veh.model.policy.parameters())
                theta_veh_q1_plus_noise  = self.flattenParam(veh.model.qf1.parameters())
                theta_veh_q2_plus_noise  = self.flattenParam(veh.model.qf2.parameters())

                cos_sim_plus_noise = torch.nn.functional.cosine_similarity(theta_veh_policy_plus_noise,
                                                                           theta_global_model_policy, dim=0) + \
                                     torch.nn.functional.cosine_similarity(theta_veh_q1_plus_noise,
                                                                           theta_global_model_q1, dim=0) + \
                                     torch.nn.functional.cosine_similarity(theta_veh_q2_plus_noise,
                                                                           theta_global_model_q2, dim=0)

                "计算对抗损失"
                loss.append( 1/2*(cos_sim - cos_sim_plus_noise)**2)

        if aggre_num_veh>0:
            self.resistLoss +=  torch.mean(torch.stack(loss), dim=0)




    def step(self):
        # 判断下一个slot是否会超出范围, 会,上传最新的模型
        # 改变车辆位置
        self.updateVehLoc()
        # 车辆超出范围会消失
        self.destroyVeh()
        # generate veh
        self.generateVeh()

        V2Vlink_indx = self.V2Vlink()
        self.updateGrapth(V2Vlink_indx)
        # "加入一个特征"
        self.getOverlayG()
        if self.slot >= 0:
            self.Trainlocalmodel()
        # self.generateNoise()
        # # Train local model
        # self.aggregateNoise()
        # self.addparamNoise()
        # self.checkgradient()
        # local aggregation based on the overlay G
        "联邦聚合"
        "联合损失"
        self.localAsyFederated()  # 改
        # # global model update, and will interfere with the aoi upload channel
        _ = self.globalAsyFederated() # 改
        self.updateSINR()

        self.slot+=1
        self.GenNextSlot -= 1
        for veh in self.Vehicle:
            veh.slot += 1

        mean_veh_reward = []

        for veh in self.Vehicle:
            mean_veh_reward.append(veh.reward)

        # 修复：确保mean_veh_reward不为空，避免nan
        if mean_veh_reward:
            mean_veh_reward_value = np.mean(mean_veh_reward)
            # 检查是否有nan值
            if np.isnan(mean_veh_reward_value):
                mean_veh_reward_value = 0.0  # 如果出现nan，使用默认值
        else:
            mean_veh_reward_value = 0.0  # 如果没有车辆，使用默认值

        # meanLoss = self.getMeanLoss()
        meanLoss = 0
        self.destroyAOI = 0
        return self.reward, mean_veh_reward_value, self.penalty, len(self.Vehicle), self.destroyAOI


    def getMeanLoss(self):
        mean_actor_loss = []
        mean_critic1_loss = []
        mean_critic2_loss = []
        mean_alpha_loss = []
        trainVeh_n = 0
        q1 = []
        q2 = []
        q_target = []
        train_r = []
        target_q_values = []

        for veh in self.Vehicle:
            if veh.trainNumber > 0:
                trainVeh_n += 1
                loss_value = self.get_gradient(veh)
                # loss_value = dict(update_losses._asdict().items())
                mean_actor_loss.append(loss_value["policy_loss"])
                mean_critic1_loss.append(loss_value["qf1_loss"])
                mean_critic2_loss.append(loss_value["qf2_loss"])
                mean_alpha_loss.append(loss_value["alpha_loss"])
                # mean_log_pi.append(veh.model.log_pi.detach().numpy())
                # mean_q_new_actions.append(veh.model.q_new_actions.detach().numpy())
                q1.append(veh.model.q1.detach().numpy())
                q2.append(veh.model.q2.detach().numpy())
                q_target.append(veh.model.q_target.detach().numpy())
                train_r.append(veh.model.train_r.detach().numpy())
                target_q_values.append(veh.model.target_q_values.detach().numpy())

        if trainVeh_n == 0:
            return dict(policy_loss=0, qf1_loss=0, qf2_loss=0, alpha_loss=0), 0, 0, 0, 0, 0
        else:
            # 修复：添加安全检查，防止空列表导致的nan值
            policy_loss = np.mean(mean_actor_loss) if mean_actor_loss else 0
            qf1_loss = np.mean(mean_critic1_loss) if mean_critic1_loss else 0
            qf2_loss = np.mean(mean_critic2_loss) if mean_critic2_loss else 0
            alpha_loss = np.mean(mean_alpha_loss) if mean_alpha_loss else 0
            
            q1_val = np.mean(q1) if q1 else 0
            q2_val = np.mean(q2) if q2 else 0
            q_target_val = np.mean(q_target) if q_target else 0
            train_r_val = np.mean(train_r) if train_r else 0
            target_q_values_val = np.mean(target_q_values) if target_q_values else 0
            
            return dict(policy_loss=policy_loss, qf1_loss=qf1_loss, qf2_loss=qf2_loss, alpha_loss=alpha_loss), \
                   q1_val, q2_val, q_target_val, train_r_val, target_q_values_val

    def plotEnv(self):
        VehX = []
        VehY = []
        ax1 = plt.gca()
        for veh in self.Vehicle:
            VehX.append(veh.x)
            VehY.append(veh.y)
            ax1.plot([veh.x,self.rsuX],[veh.y,self.rsuY],color = "k")
            ax1.text(x=(veh.x+self.rsuX)/2, y = (veh.y+self.rsuY)/2,  size=10,s='{:.2f}'.format(veh.v2iRate))
        ax1.scatter(VehX,VehY)
        ax1.set_xlim(0, 500)
        ax1.set_ylim(0, 3.5 * 4)
        ax1.grid(which="major", axis="both")
        plt.pause(0.0001)  # 暂停时间
        plt.cla()  # 将当前figure中绘图区的内容清除

    def printVeh(self,cumloss,loss):
        # print("detroy veh : ", self.destroyveh[-5:], " destroy AOI ", self.destroyAOI[-5:])
        for veh in self.Vehicle:
           # print("slot : ", self.slot, "ID : ",veh.record_id, " , AOI : %.2f"%np.mean(veh.AOI_list), " , power : ",veh.action,
           #       " Training number : " , veh.trainNumber, "local aggregation : " ,veh.n_LocalDRL , "glabal aggregation : ",veh.n_GlobalDRL,
           #       "actor loss :", veh.actor_loss ,"critic loss : ", veh.critic_loss)
           first_tasksize = 0

           a = veh.model.qf1.state_dict()
           if len(veh.queue)>0:
               first_tasksize = veh.queue[0].size
           # print("slot : ", self.slot, "ID : ",veh.record_id, "Translate : ",veh.Translate
           #       " , AOI : %.2f"% np.mean(veh.AOI_list), " Training number : " , veh.trainNumber, veh.model.actor.state_dict()['l1.weight'][:3][0],
           #       " , power : %.2f"%veh.action, "glabal aggregation : ",veh.n_GlobalDRL,"actor loss :", veh.actor_loss ,"critic loss : ", veh.critic_loss)
           # a = veh.model.policy.state_dict()
           print(

                 # "slot:",self.slot,
                 " ID:",veh.record_id,
                 # " lane:",veh.lane,
                 # " channel:%.2f"%veh.channel,
                 # " x :" ,veh.x,
                 # "FirstSize:%.2f"%(first_tasksize/MB),
                 # "Transsize:%.2f"%(veh.Transimission_tasksize/MB),
                 # "Transrate:%.2f"%(veh.v2iRate/MB),
                 # "lane ：",veh.lane,
                 # " Channel:",veh.channel,
                 # "sinr:",veh.sinr,
                 # " Translate:",veh.Translate,
                 # " Task number:",len(veh.queue),
                 # " AOI:%.2f" %np.sum(veh.AOI_list),
                 # " reward:%.2f"%veh.reward,
                 # " ration :%.2f"% self.reward/np.mean(veh.AOI_list) if veh.queue else 10,
                 # " power_pre:%.2f"%veh.action_pre,
                 # " power:%.2f" % veh.action,
                 # " Channel:", veh.channel,
                 # " gain : ", veh.g,
                 # " small:",abs(veh.small_currentH),
                 # " dis : ",veh.dis
                 # " Training number:", veh.trainNumber,
                 # " alpha : ", veh.model.alpha if veh.model.alpha else 0,
                 # " rate learning :",veh.model.policy_lr
                 # "alpha loss :", veh.model.alpha_loss  if veh.model.alpha_loss else 0,
                 # "aggre veh:",veh.aggre_veh,
                 # "aggre weight :", veh.aggre_weight,
                 # " actor loss:", veh.model.loss.policy_loss,
                 # " q1 loss:", veh.model.loss.qf1_loss,
                 # " q2 loss:", veh.model.loss.qf2_loss,
                 # " s ", veh.s0,
                 "actor para: " , veh.model.policy.state_dict()['hidden1.weight'][:2][0],
                 "qf1 para: ", veh.model.qf1.state_dict()['fc0.weight'][:2][0],
                 "qf2 para: ", veh.model.qf2.state_dict()['fc0.weight'][:2][0]
                 # "train data number : ",veh.Training_number,
                 # "para: ", veh.model.critic.state_dict()['l1.weight'][:3][0]
                )
           # print(veh.actor_loss,veh.critic_loss)
           # print(veh.model.actor.state_dict()['l1.weight'][:3][0],
           #       veh.model.critic.state_dict()['l1.weight'][:3][0])
           # print()
           # print("slot : ", self.slot,
           #       "ID : ", veh.record_id,
           #       "reward : %.2f"%veh.reward,
           #       " Training number : ", veh.trainNumber,
           #       " param: ", veh.model.actor.state_dict()['l1.weight'][:4][0],
           #       " actor loss :" , veh.actor_loss,
           #       " critic loss : ", veh.critic_loss
           #       )
           # print("slot: ", self.slot, " ID : ", veh.record_id, " lane : ",veh.lane," x:%.2f "%veh.x," y:%.2f "%veh.y,
           #       " dis:%.2f "%veh.dis," large:%.2f "%veh.large_currentH," small:%.2f "%veh.small_currentH," channel : %.2f "%veh.channel,
           #       " power:%.2f " % veh.action," sinr:%.2f "%veh.sinr," v2iRate:%.2f "%(veh.v2iRate/MB),)
        # print("Edge : ", self.edge," , connect edge : ",self.connect_edge , "overlay_edge",self.overlay_edge)
        # print("mask : " ,self.train_mask)
        # print("Edge : ", self.edge)
        # print("Connect_edge", self.connect_edge)
        # print("Overlay_edge", self.overlay_edge)

        # lane0_birth_max = 0
        # lane1_birth_max = 0
        # lane2_birth_max = 0
        # lane3_birth_max = 0
        # for veh in self.Vehicle:
        #     print(" ID :" , veh.idx, " ,birth : ",veh.slot," ,AOI:", veh.AOI_list, "reward :" , veh.reward)
        # for veh in self.Vehicle:
        #     if veh.lane == 0 :
        #
        #         lane0_birth_max = max(lane0_birth_max,veh.slot)
        #     if veh.lane == 1 :
        #         lane1_birth_max = max(lane1_birth_max,veh.slot)
        #     if veh.lane == 2 :
        #         lane2_birth_max = max(lane2_birth_max,veh.slot)
        #     if veh.lane == 3 :
        #         lane3_birth_max = max(lane3_birth_max,veh.slot)
        # print(lane0_birth_max,lane1_birth_max,lane2_birth_max,lane3_birth_max)

        print("slot : ", self.slot,
              "penaly : ", self.penalty,
              # "R+P : ", self.penalty + self.reward,
              " aggregation number : ", self.globalaggreNumber,
              ' AOI : $%.2f ' % self.reward,
              # " slideAOI :%.2f "%self.slideReward
              "rate : ", self.DRL_model.policy_lr,
              "alpha : ", self.alpha,
              "num : ", len(self.Vehicle)
              )
        # print(
        #      #"modeless:", self.modeless,
        #       # "num veh:",len(self.Vehicle),
        #       "GCN Training :", self.GCN_Train,
        #       # "GCN reward : ",self.node_loss,
        #       "GCN cum loss : ", cumloss,
        #       "GCN loss : ", loss, )
        # a = self.GCN_model.state_dict()
        # print(self.G_conv)
        print(self.GCN_model.state_dict()["conv1.lin.weight"][:3])
        # print(self.GCN_model.GCN_critic.state_dict()['fc0.weight'][:2][0])
        # print("actor para: " , self.DRL_model.policy.state_dict()['hidden1.weight'][:2][0],
        #          "qf1 para: ", self.DRL_model.qf1.state_dict()['fc0.weight'][:2][0],
        #          "qf1 para: ", self.DRL_model.qf2.state_dict()['fc0.weight'][:2][0])
        # print(self.DRL_model.actor.state_dict()['l1.weight'][:4][0])
        # print("slot : ",self.slot,
        #       "num veh:",len(self.Vehicle),
        #       " aggregation number : ", self.globalaggreNumber ,
        #       ' AOI : $%.2f '%self.reward,
        #       " param : ",self.DRL_model.actor.state_dict()['l1.weight'][:4][0])


        print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")