import numpy as np
import torch
import pdb
import copy
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from nn_function import pytorch_util as ptu
import uuid
import matplotlib.pyplot as plt
from scipy import special as sp
from scipy.constants import pi
from scipy import special
import networkx as nx
import time
import project_backend as pb
from torch_geometric.utils import  to_networkx
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from collections import deque
from memory_profiler import profile
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
    return 1.0/(1+np.exp(-x))
def complexGaussian(row=1, col=1, amp=1.0):
    real = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    img = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    return amp*(real + 1j*img)
def get_random_rayleigh_variable(rayleigh_var):

    return np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn() +
                                                1j * rayleigh_var * np.random.randn())
def get_markov_rayleigh_variable(state,correlation,rayleigh_var):

    return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn() +
                                                1j * rayleigh_var * np.random.randn())

class GCNMemory:
    def __init__(self,memory_size,input_size ,outputsize):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory_gcn = torch.zeros((self.memory_size, (input_size + outputsize))) # 1 : reward
        #self.memory_r = np.zeros((self.memory_size,  1))
        self.memory_gcn_ = torch.zeros((self.memory_size, (input_size + outputsize)))  # 1 : reward
        #self.G_data = [[] for _ in range(self.memory_size)]
        #self.G_data_ = [[] for _ in range(self.memory_size)]
        self.G_data = []
        self.G_data_ = []
        self.memory_r =[]


    def Addremember(self, TraingcnData,r,TraingcnData_,G_data,G_data_):
        idx = self.memory_counter % self.memory_size
        #self.memory_gcn[idx, :] = TraingcnData
        #self.memory_r[idx, :] = r
        #self.memory_gcn_[idx, :] = TraingcnData_
        self.memory_counter += 1
        #self.G_data[idx] = G_data
        #self.G_data_[idx] = G_data_
        self.memory_r.append(r)
        self.G_data.append(G_data)
        self.G_data_.append(G_data_)

        #self.memory_train_mask.append(train_mask)

        # 如果记忆超过设定的大小，删除最早的记忆
        if len(self.G_data) > self.memory_size:
            self.G_data.pop(0)
            self.G_data_.pop(0)
            self.memory_r.pop(0)
            #self.memory_train_mask.pop(0)

class My_GCN(torch.nn.Module):
    def __init__(self,Input_feauture_size,output_featrue_size):
        super().__init__()
        self.train_total_epoch = 400
        torch.manual_seed(1234)
        self.conv1 = GCNConv(Input_feauture_size,128)
        self.conv2 = GCNConv(128,64)
        self.conv3 = GCNConv(64,output_featrue_size)


         # both output feature size
    def forward(self,x,edge_index):
        h = self.conv1(x,edge_index) # node imformation
        h = h.tanh()
        h = self.conv2(h,edge_index) # node imformation
        h = h.tanh()
        h = self.conv3(h,edge_index) # node imformation
        out = h.tanh()
        return out
    def GCN_train_from_torch(self,Traindata_node,r,Traindata_node_):

        gcn_loss = -self.GCN_critic(Traindata_node.clone()).mean()

        self.gcn_optimizer.zero_grad()

        gcn_loss.backward()
        self.gcn_optimizer.step()

        Traindata_node_new = Traindata_node.clone().detach()
        q = self.GCN_critic(Traindata_node_new)
        target_critic = self.GCN_target_critic(Traindata_node_)
        target_y = self.reward_scale * r.to(torch.float32) + self.discount * target_critic

        critic_loss = nn.MSELoss()(q, target_y.detach())
        self.gcn_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.gcn_critic_optimizer.step()

        ptu.soft_update_from_to(
            self.GCN_critic, self.GCN_target_critic, self.soft_target_tau
        )

        return gcn_loss,critic_loss

    def train(self,num_nodes,GCN_critic,node_feature_size,out_feature_size,reward_scale):
        self.num_nodes = num_nodes
        self.num_sampled = 5
        self.node_feature_size = node_feature_size
        self.out_feature_size = out_feature_size
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(self.num_nodes + 1)  # 权重归一化
                    for k in range(self.num_nodes)
                ]
            ),
            dim=0,
        )

        self.GCN_critic = GCN_critic(input_size= self.num_nodes * (self.node_feature_size + self.out_feature_size),
                                     output_size=1, hidden_sizes=[256, 256], )

        self.GCN_target_critic = GCN_critic(
            input_size=self.num_nodes * (self.node_feature_size + self.out_feature_size), output_size=1,
            hidden_sizes=[256, 256], )

        self.GCN_memory = GCNMemory(5000, self.num_nodes * self.node_feature_size,
                                    self.num_nodes * self.out_feature_size)
        self.GCNupadate_counter = 256
        self.GCN_Training_number = 5
        self.reward_scale = reward_scale
        self.discount = 0.99
        self.soft_target_tau = 0.005
        # b = [{"params": self.GCN_critic.parameters()}]
        self.gcn_critic_optimizer = torch.optim.Adam(
            [{"params": self.GCN_critic.parameters()}], lr=1e-3
        )
        # a = [{"params": self.parameters()},{"params": self.conv2.parameters()},{"params": self.conv3.parameters()}]
        self.gcn_optimizer = torch.optim.Adam(
            [{"params": self.conv1.parameters()},
             {"params": self.conv2.parameters()},
             {"params": self.conv3.parameters()}], lr=1e-3
        )

    def formulate(self,x,edge_indx,train_mask):
        # x = torch.tensor([[1,2,3],[2,3,4],[4,5,6],[7,8,9]],dtpye = torch.float)
        # edge_index = torch.tensor([[0,2,1],                       起始点 node
        #                            [1,3,3]],dtpye = torch.long)   结束点 node
        # y = torch.tensor([0,0,1,1],dtpye = torch.float)
        return Data(x=x,edge_index=edge_indx,train_mask = train_mask)
    def visualize_graph(self,G,connect_G,overlay_G,color):
        ax = plt.gca()
        # plt.subplot(211)
        plt.xticks([])
        plt.yticks([])
        # nx.draw_networkx (G,pos = nx.spring_layout(G, seed=42),nodelist=G.nodes, node_size=50, node_color='k', node_shape='o')

        nx.draw_networkx_nodes(G,  pos=nx.spring_layout(G, seed=42),nodelist=G.nodes, node_size=30, node_color='k', node_shape='o', cmap=None,
                            vmin=None, vmax=None, ax=None, linewidths=None, label=None)

        # nx.draw_networkx_edges(G, pos=nx.spring_layout(G, seed=42),edgelist=G.edges, width=1, edge_color='k', style='solid', alpha=None,
        #                     edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None,
        #                     arrows=False, label=None)

        nx.draw_networkx_edges(G, pos=nx.spring_layout(G, seed=42),edgelist=connect_G.edges, width=1, edge_color='k', style='solid', alpha=None,
                            edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None,
                            arrows=False, label=None)
        nx.draw_networkx_edges(G, pos=nx.spring_layout(G, seed=42),edgelist=overlay_G.edges, width=1, edge_color='r', style='dashdot', alpha=None,
                               edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None,
                               arrows=False, label=None)

        # nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), node_size = 25,font_size=10,
        #                  with_labels=False, node_color=color, cmap="Set2")
        # # nx.draw_networkx_edges(G, pos=nx.spring_layout(G, seed=42), edgelist=connect_G.edges, width=1, edge_color='r')
        #
        #
        # # plt.subplot(212)
        # nx.draw_networkx(overlay_G, pos=nx.spring_layout(overlay_G, seed=42), node_size = 25,font_size=10,
        #                  with_labels=False, node_color="k", cmap="Set2",edge_color='r')
        # nx.draw_networkx_edges(overlay_G, pos=nx.spring_layout(overlay_G, seed=42), edgelist=overlay_G.edges, width=1, edge_color='r')

        plt.pause(0.0001)  # 暂停时间
        plt.cla()  # 将当前figure中绘图区的内容清除
    def visualize_embedding(self,h,color,epoch=None,loss=None):
        plt.subplot(313)
        h = h.detach().cpu().numpy()
        plt.scatter(h[:,0],h[:,1],s=140,c=color,cmap="Set2")
        if epoch is not None and loss is not  None:
            plt.xlabel("f'Epoch:{epoch},Loss:{loss.item():.4f}",fontsize = 16)
    def NSLoss(self,data,out_feature):


        n_batch = data.shape[0]
        embs_node = data[:,0:out_feature]
        embs_NeiborNode = data[:,out_feature:2*out_feature]
        OtherNode = data[:,2*out_feature:].reshape(n_batch,self.num_nodes,-1)

        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs_node, embs_NeiborNode), 1))  # self.weights[label] 生成另一个节点的信息
        )  #

        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n_batch, replacement=True  #
        ).view(n_batch,self.num_sampled)
        embs_OtherNode = []

        for batch in range(n_batch):
            batch_i_embs = []
            for sample_i in range(self.num_sampled):
                batch_i_embs.append(OtherNode[sample_i][negs[batch][sample_i]].detach().numpy())
            embs_OtherNode.append(batch_i_embs)

        embs_OtherNode = torch.tensor(embs_OtherNode)
        noise = torch.neg(embs_OtherNode)  # torch.neg 取负数
        sum_log_sampled = torch.sum(
             torch.log(torch.sigmoid(torch.bmm(noise, embs_node.unsqueeze(2)))), 1
         ).squeeze()
        #
        loss = log_target + sum_log_sampled
        return -loss.sum() / n_batch

    def save(self,directory ):
        torch.save(self.state_dict(), directory + 'GCN.pth')
        print("... ... ...GCN Model has been saved ... ... ...")
        print("========================================================")

    def load(self,directory ):
        self.load_state_dict(torch.load(directory + 'GCN.pth'))
        print("... ... ...GCN Model has been loaded... ... ...")
        print("========================================================")


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
        g_dB = - (32.4 + 20 * np.log10(0.001 * self.dis) + 20 * np.log10(f_c)) + shadowing_dev * self.current_shadowing
        # print(self.current_shadowing,g_dB,np.power(10.0,g_dB/10.0))
        # print(self.record_id, self.dis, - (128.1 + 37.6* np.log10(0.001*self.dis)), g_dB,np.power(10.0,g_dB/10.0))
        self.g = np.power(10.0,g_dB/10.0)
        return np.power(10.0,g_dB/10.0)
    def calChannelGain(self):
        # print("测试信道：",self.dis,self.current_shadowing,self.g,self.large_currentH,self.small_currentH)
        # if np.isnan(self.current_shadowing):
        #     print("0")
        return np.sqrt(self.large_currentH)*abs(self.small_currentH)
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

        self.rho1 = np.exp(- self.dis/dcor)

    def compute_V2IRho2(self):
        x_0 = np.array([1, 0, 0])
        f_d =  self.v * self.slotT * np.dot(x_0, (self.rsuLoc - self.Loc)) / \
               (np.linalg.norm(self.rsuLoc - self.Loc)) *f_c/(self.slotT*v_c)

        f_d = self.dis * f_c / (self.slotT*v_c)

        self.rho2 =  special.j0(2.0 * np.pi * f_d * self.slotT)

    def compute_sinr(self,allPowerGain,uploadmodelPower):
        self.lastSINR = self.sinr
        # SINR updating needs to be after last SINR
        self.sinr = self.action * self.channel/(allPowerGain -self.action * self.channel + noise_var + uploadmodelPower)
        # if np.isnan(self.sinr):
        #     print(self.action,self.channel,allPowerGain,uploadmodelPower,noise_var)
        self.v2iRate = W * np.log2(1+self.sinr)
        # print("测试通信:",self.record_id,self.rho1,self.sinr,self.v2iRate)

    def process_state(self,x):
        x = torch.tensor(x)
        x_max = x.max(dim=0,keepdim=True).values
        x_log = torch.log10(x / x_max + 1)
        x_min = x_log.min(dim=0, keepdim=True).values
        x_normalized = (x_log - x_min) / (1 - x_min)
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
                 ,p,DRL_model,GCN_batchsize,maxP,slideW,w,k,penalty_factor,powerFactor,param_noise_var,GCN_factor, critic_noise_var , rateDecay,GCN_critic,reward_scale,plot=False):
        super().__init__()
        self.episode = 0
        self.slotT = slotT
        self.lane = lane
        self.gridW = gridW
        self.plot = plot
        self.maxP = maxP
        self.penalty_factor = penalty_factor
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
        self.GCN_model = My_GCN(self.node_feature_size,self.out_feature_size) # output feature size = input feature size
        self.GCN_model.train(4 * (int(self.maxRoadLen/self.gridW)),GCN_critic,self.node_feature_size,self.out_feature_size,reward_scale)
        self.GCN_model.to(device)
        self.GCN_data = self.GCN_model.formulate(self.node,self.edge,train_mask = self.train_mask)
        self.G = to_networkx(self.GCN_data,to_undirected=True)
        self.GCN_factor = GCN_factor
        self.GCN_loss = 0.0
        self.total_GCN_loss = 0.0
        self.GCN_batchsize = GCN_batchsize
        self.optimizer = torch.optim.Adam(
            [{"params": self.GCN_model.parameters()}], lr=1e-4
        )
        self.lastreward = 0
        self.reward = 0
        self.node_loss = self.reward
        self.slideReward = 0
        self.penalty = 0
        self.w = w
        self.k = k
        self.slideW = slideW
        # self.slideR = self.slidereward()
        self.GCN_Train = False
        #
        self.generateVeh()

        V2Vlink_indx = self.V2Vlink()
        self.updateGrapth(V2Vlink_indx)
        self.last_G_conv = self.GCN_model.forward(self.GCN_data.x.detach(), self.GCN_data.edge_index.detach())
        self.last_GCN_data = self.GCN_data
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
        # self.GCN_data = self.GCN_model.formulate(self.node,self.edge,train_mask = self.train_mask)
        # self.G = to_networkx(self.GCN_data,to_undirected=True)
        self.GCN_loss = 0.0
        self.total_GCN_loss = 0.0
        # self.slideR = self.slidereward()
        self.generateVeh()
        # self.resetVeh(episode)
        V2Vlink_indx = self.V2Vlink()

        self.updateGrapth(V2Vlink_indx)
        self.last_GCN_data = self.GCN_data
        self.last_G_conv = self.GCN_model.forward(self.GCN_data.x.detach(), self.GCN_data.edge_index.detach())

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
                    self.meanAOI += np.mean(veh.AOI_list) + veh.Transimission_AOI
                else:
                    self.meanAOI += np.mean(veh.AOI_list)
            # for task in veh.queue:
            #     self.meanAOI += task.AOI

            # self.meanAOI += np.sum(veh.AOI_list)

        if len(self.Vehicle)>0:
            self.meanAOI = self.meanAOI/len(self.Vehicle)
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
                    penalty += np.mean(veh.AOI_list)
                self.Vehicle.remove(veh)
        if self.Vehicle:
            self.penalty += penalty/len(self.Vehicle) # 消失的对还存在的影响


    def getuploadPower(self,allPowerGain,uploadveh):
        if uploadveh == 0:
            return [0]
        Model_size = self.model_total_size
        a = np.exp(Model_size/W*np.log(2))-1
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
                if veh.v2iRate >= First_task.size:
                    veh.Translate = True
                    Transimission_AOI += First_task.size/veh.v2iRate
                    Transimission_veh += 1
                    veh.Transimission_AOI = First_task.size/veh.v2iRate
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
        self.GCN_data = self.GCN_model.formulate(self.node,self.edge,train_mask = self.train_mask)
        self.G = to_networkx(self.GCN_data,to_undirected=True)

        self.G_conv = self.GCN_model.forward(self.GCN_data.x.detach(), self.GCN_data.edge_index.detach())


        self.edge_len = link_temp_len


    def Hadamard(self,G_conv,n,m):
        "-------------------------------------------edge process------------------------------------------------------"
        return torch.mean(G_conv[n]*G_conv[m])


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
        self.G_conv = self.GCN_model.forward(self.GCN_data.x.detach(),self.GCN_data.edge_index.detach())
        # Because I store the node information output from GCN into memory as training data,
        # the gradient grad_fn version will change after each network update,
        # putting the gradient node information with different versions into memory for training will result in an error,
        # so the detach operation is needed
        # self.G_conv = self.G_conv.detach()
        connect_edge = [[],[]]
        total_node_feature = self.G_conv
        total_node_feature = total_node_feature.reshape(1,-1)
        for n, m in zip(self.edge[0], self.edge[1]):
            # edge_feature = self.Hadamard(self.G_conv,n,m)
            edge_feature = torch.sum(torch.mul(self.G_conv[n],self.G_conv[m]))
            # add memory
            # self.GCN_model.memory.add(torch.cat([self.G_conv[n],self.G_conv[m],total_node_feature[0]]))
            P = sigmoid(edge_feature.detach().numpy())
            if P >=self.p:
                connect_edge[0].append(n)
                connect_edge[1].append(m)

        self.connect_edge = torch.tensor(connect_edge,dtype=torch.long)
        # self.overlay_G = Chrisrofide()
        self.connect_data = self.GCN_model.formulate(self.node,self.connect_edge,train_mask = self.train_mask)
        self.connect_G = to_networkx(self.connect_data,to_undirected=True)
        if len(self.connect_edge[0]) == 0 :
            overlay_edge = [[],[]]
        else:
            overlay_edge = self.prim()
        new_overlay_edge = self.process_edge(overlay_edge)

        # self.overlay_edge = torch.tensor(new_overlay_edge,dtype=torch.long)
        # self.overlay_data = self.GCN_model.formulate(self.node,self.overlay_edge,train_mask = self.train_mask)
        # self.overlay_G = to_networkx(self.overlay_data, to_undirected=True)

        self.overlay_edge = self.connect_edge
        self.overlay_data = self.connect_data
        self.overlay_G = self.connect_G


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

        def f(a, b,powerFactor):
            x = a/b
            # if a == 0:
            #     return 0
            # y = a * np.log(1 + b / a) / (1 + np.exp(-a * np.log(1 + b / a)))
            # return y/100
            return np.exp( powerFactor * x) / powerFactor

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

            if veh.queue and (np.mean(veh.AOI_list) > 0):
                x = (self.reward) / np.mean(veh.AOI_list)
                # veh.reward = self.reward + self.penalty * 1 + veh.action *  np.exp((self.reward) / np.mean(veh.AOI_list))
                # veh.reward = self.reward + self.penalty * 1 + veh.action * np.exp(x)/(1+np.exp(x))
                veh.reward = self.reward + self.penalty * 1 + veh.action *  (1 + self.reward / np.mean(veh.AOI_list))

                 #self.penalty + veh.action *  (self.reward) / np.mean(veh.AOI_list)
            else:
                x = self.reward
                veh.reward = self.reward + self.penalty * 1 + veh.action * (1 + self.reward)

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
            veh.model.memory.Addremember(np.concatenate((veh.s0, [veh.action_pre], [-veh.reward], s1, [veh.done])))


            veh.s0 = s1
            self.penalty = self.penalty * self.penalty_factor


    def Trainlocalmodel(self):
        # birth : 1465 812 857 715
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

    def getWeight(self,veh,node_link):
        # The weight is equal to the node's feature
        aggre_veh = []
        node_feature = self.G_conv[veh.node].detach().numpy()
        weight = [np.exp(np.mean(np.abs(node_feature), axis=0))]
        if not node_link or veh.node not in node_link.keys():
            aggregate_node_list = [veh.node]
        else:
            aggregate_node_list = node_link[veh.node]

        aggre_veh.append(veh)

        for aggregate_node in aggregate_node_list:
            aggregate_veh_list = self.node_veh[aggregate_node]
            for veh2 in aggregate_veh_list:
                if veh == veh2:
                    continue
                if veh2.trainNumber == 0:
                    continue
                node_feature = self.G_conv[veh2.node].detach().numpy()
                veh.n_LocalDRL += 1

                weight.append(np.exp(np.mean(np.abs(node_feature), axis=0)))
                aggre_veh.append(veh2)


        return weight ,aggre_veh

    def updateModel(self, model, gradient, rate):
        for p, grad in zip(model.parameters(), gradient):
            p.data -= rate * grad


    def localAsyFederated(self):
        self.G_total_feature = 0
        modeless_n = 0

        for node_feture in self.G_conv:
            self.G_total_feature += torch.mean(torch.abs(node_feture))

        node_link = self.getNodelinkMatrix()

        # aggre_num_veh = 0
        node_loss = []
        self.fedLoss = 0

        for veh in self.Vehicle:
            model_loss = 0
            if veh.need_local_aggregate == False:
                continue

            weight, aggre_veh = self.getWeight(veh, node_link)


            if all(elem == 0 for elem in weight):
                mean_weight = weight
            else:
                #mean_weight = (weight / np.sum(weight)).tolist()
                mean_weight = weight
                # weight = [1]
                # aggre_veh = [veh]

            mean_weight = (weight / np.sum(mean_weight)).tolist()

            veh.aggre_weight = mean_weight
            veh.aggre_veh = aggre_veh

            with torch.no_grad():
                # 获取模型的权重字典的键
                avg_weights_policy = veh.model.policy.state_dict()
                avg_weights_q1 = veh.model.qf1.state_dict()
                avg_weights_q2 = veh.model.qf2.state_dict()

                policy_keys = avg_weights_policy.keys()
                q1_keys = avg_weights_q1.keys()
                q2_keys = avg_weights_q2.keys()

                avg_weights_policy = {key: avg_weights_policy[key]  * veh.aggre_weight[0] for key in policy_keys}
                avg_weights_q1 = {key: avg_weights_q1[key] * veh.aggre_weight[0] for key in q1_keys}
                avg_weights_q2 = {key: avg_weights_q2[key] * veh.aggre_weight[0] for key in q2_keys}

                if len(veh.aggre_veh)>1:
                    for weight2, veh2 in zip(veh.aggre_weight,veh.aggre_veh):
                        if  veh == veh2:
                            continue
                        else:
                            policy_weights_veh2 = veh2.model.policy.state_dict()
                            q1_weights_veh2 = veh2.model.qf1.state_dict()
                            q2_weights_veh2 = veh2.model.qf2.state_dict()
                            avg_weights_policy = {key: avg_weights_policy[key] + policy_weights_veh2[key] * weight2 for
                                                  key in policy_keys}
                            avg_weights_q1 = {key: avg_weights_q1[key] + q1_weights_veh2[key] * weight2 for key in
                                              q1_keys}
                            avg_weights_q2 = {key: avg_weights_q2[key] + q2_weights_veh2[key] * weight2 for key in
                                              q2_keys}
                else:
                    pass
                # veh.model.policy.load_state_dict(avg_weights_policy)
                veh.model.qf1.load_state_dict(avg_weights_q1)
                veh.model.qf2.load_state_dict(avg_weights_q2)


            veh.aggre_veh = []

            veh.global_aggregate = True
            veh.n_GlobalDRL+=1


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
        G_total_feature = 0

        for node_feture in self.G_conv:
            G_total_feature += np.exp(np.mean(np.abs(node_feture.detach().numpy())))

        for veh in aggre_veh:
            aggre_alpha += torch.tensor(w_alpha, dtype=torch.float32) * veh.model.alpha

        self.alpha = aggre_alpha
        self.globalaggreNumber += 1


    def updateGCN(self):
        gcn_loss = 0
        critic_loss = 0
        minibatch = 128
        print("GNN update ********************************************************")
        if len(self.Vehicle) > 0 :
            if self.GCN_model.GCN_memory.memory_counter > self.GCN_model.GCNupadate_counter :

                for it in range(self.GCN_model.GCN_Training_number):
                    if self.GCN_model.GCN_memory.memory_counter > self.GCN_model.GCN_memory.memory_size:
                        sample_index = np.random.choice(self.GCN_model.GCN_memory.memory_size, size=minibatch)
                    else:
                        sample_index = np.random.choice(self.GCN_model.GCN_memory.memory_counter, size=minibatch)

                    Traindata_node = self.GCN_model.GCN_memory.memory_gcn[sample_index, :]
                    r = [torch.tensor(self.GCN_model.GCN_memory.memory_r[i],dtype=torch.float32) for i in sample_index]
                    r = torch.stack(r).reshape(-1,1)
                    Traindata_node_ = self.GCN_model.GCN_memory.memory_gcn_[sample_index, :]

                    G_data = [self.GCN_model.GCN_memory.G_data[i] for i in sample_index]
                    G_data_ = [self.GCN_model.GCN_memory.G_data_[i] for i in sample_index]


                    G_datax =  torch.cat([data.x.detach()  for data in G_data],dim = 0).view(minibatch,-1)
                    G_datax_ = torch.cat([data.x.detach() for data in G_data_], dim=0).view(minibatch, -1)

                    Convdata = torch.cat([self.GCN_model.forward(data.x.detach(),data.edge_index.detach()) for data in G_data],dim = 0).view(minibatch,-1)
                    Convdata_ = torch.cat([self.GCN_model.forward(data.x.detach(),data.edge_index.detach()) for data in G_data_],dim=0).view(minibatch,-1)
                    GCNdata = torch.cat([G_datax, Convdata], dim=1)
                    GCNdata_ = torch.cat([G_datax_, Convdata_], dim=1)
                    # GCNdata_ = Traindata_node_.clone()
                    # node = torch.tensor(Traindata[:, :len(self.node) * (self.node_feature_size)],
                    #                  dtype=torch.float32)
                    #
                    # r = torch.tensor(Traindata[:, len(self.node) * (self.node_feature_size)],
                    #                  dtype=torch.float32)
                    # node_ = torch.tensor(
                    #     Traindata[:, len(self.node) * (self.node_feature_size) + 1:],
                    #     dtype=torch.float32)

                    # r = torch.unsqueeze(Traindata_r, dim=1)
                    gcn_loss,critic_loss = self.GCN_model.GCN_train_from_torch(GCNdata,r,GCNdata_)



        return gcn_loss,critic_loss


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

            veh.aggre_weight = weight/np.sum(weight)
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
                loss.append( 1/2*(cos_sim - cos_sim_plus_noise)**2 * torch.mean(self.G_conv[veh.node]))

        if aggre_num_veh>0:
            self.resistLoss +=  torch.mean(torch.stack(loss), dim=0)


    def updateGCNMemory(self):
        self.GCN_model.GCN_memory.Addremember( torch.cat((self.last_node.view(-1),self.last_G_conv.view(-1))),-self.reward,
                                               torch.cat((self.node.view(-1), self.G_conv.view(-1))),self.last_GCN_data,self.GCN_data)

        # print(np.concatenate((veh.s0, [veh.action], [-veh.reward], s1, [veh.done])))
        self.last_node = self.node
        self.last_G_conv = self.G_conv
        self.last_GCN_data = self.GCN_data


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
        self.updateGCNMemory()
        # caculate sinr --- every agent make action --get v2i rate
        # Train the GCN model
        gnnloss = None
        gnnCriticloss = None
        if self.slot % 1000 == 0:
            gnnloss,gnnCriticloss = self.updateGCN()

        self.slot+=1
        self.GenNextSlot -= 1
        for veh in self.Vehicle:
            veh.slot += 1
        if self.plot:
            # self.plotEnv()
            self.GCN_model.visualize_graph(self.G,self.connect_G,self.overlay_G,color='k')
        # self.printVeh(gnnloss,gnnCriticloss)
        self.GCN_Train = False

        mean_veh_reward = []


        for veh in self.Vehicle:
            mean_veh_reward.append(veh.reward)

        # meanLoss = self.getMeanLoss()
        meanLoss = 0
        self.destroyAOI = 0
        return self.reward, np.mean(mean_veh_reward) , self.penalty, len(self.Vehicle), self.destroyAOI,gnnloss,gnnCriticloss


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
            return dict(policy_loss=np.mean(mean_actor_loss), qf1_loss=np.mean(mean_critic1_loss),
                        qf2_loss=np.mean(mean_critic2_loss), alpha_loss=np.mean(mean_alpha_loss)), \
                   np.mean(q1), np.mean(q2), np.mean(q_target), np.mean(train_r), np.mean(target_q_values)

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