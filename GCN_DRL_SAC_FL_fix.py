import numpy as np

#import tensorflow as tf
from collections import OrderedDict, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from nn_function.pythonplusplus import identity
from nn_function import pytorch_util as ptu
from torch.distributions import MultivariateNormal
from nn_function.distributions import TanhNormal
from nn_function.core import PyTorchModule
from nn_function.normalization import LayerNorm
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss',
)

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,       # [256,256]
            output_size,        # =1
            input_size,         # =action_dim+obs_dim = 13+6
            init_w=3e-3,        # 当成
            hidden_activation=F.relu,   #
            output_activation=identity, # 恒同映射
            hidden_init=ptu.fanin_init, # return tensor.data.uniform_(-bound, bound) 产生(-bound, bound)均匀分布
            b_init_value=0.,
            layer_norm=False,           # 网络层 norm 化
            layer_norm_kwargs=None,
    ):
        super().__init__()
        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        self.input_size = input_size                # 23 因为输出的是 action 和 state
        self.output_size = output_size
        self.hidden_activation = hidden_activation  # 这里没有任何输入参数，说明附值为一个函数 F.relu
        self.output_activation = output_activation  # 这里没有任何输入参数，说明附值为一个函数 identity
        self.layer_norm = layer_norm                # none
        self.fcs = []
        self.layer_norms = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes): # 0 256 ; 1 256
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)              # fc.weight 网络的权重 。 均匀分布权重。 但函数没有接收返回，是怎么改变fc.weight的值的？
            fc.bias.data.fill_(b_init_value)    #  bias  参数初始化 = 0
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            if self.layer_norm:                 # 网络层 norm 化
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class Memory:
    def __init__(self,memory_size,input_size ,outputsize):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = np.zeros(( self.memory_size, input_size + outputsize + 1 + input_size + 1))


    def Addremember(self , TrainData):
        idx = self.memory_counter % self.memory_size
        self.memory[idx,:] = np.array(TrainData)
        self.memory_counter += 1





class ActorCritic(nn.Module):
    def __init__(self,input_size,output_size,hidden_init=ptu.fanin_init,b_init_value=0.,init_w=3e-3):
        super(ActorCritic, self).__init__()

        self.hidden1 = nn.Linear(input_size, 256)  # 输入参数 5 待修改
        hidden_init(self.hidden1.weight)
        self.hidden1.bias.data.fill_(b_init_value)

        self.hidden2= nn.Linear(256, 256)  # 输入参数 5 待修改
        hidden_init(self.hidden2.weight)
        self.hidden2.bias.data.fill_(b_init_value)
        #

        self.last_mean = nn.Linear(256,output_size)
        self.last_mean.weight.data.uniform_(-init_w, init_w)
        self.last_mean.bias.data.fill_(0)

        self.last_fc_log_std = nn.Linear(256,output_size)  # 输出层
        self.last_fc_log_std.weight.data.uniform_(-init_w/3, init_w/3)  # 输出层权重初始化
        self.last_fc_log_std.bias.data.uniform_(-init_w/3, init_w/3)  # 输出层bias初始化


    def forward(self, state, memory):
        raise NotImplementedError

    def act_dist(self,input,max):
        input_1 = input.reshape(len(input),1,-1)
        h_1 = F.relu(self.hidden1(input_1))
        h_2 = F.relu(self.hidden2(h_1))

        mean=self.last_mean(h_2) # choose : [1,4], exercise : [256,6]

        log_std = self.last_fc_log_std(h_2)
        log_std = torch.clamp(log_std, -max,max)        # log_std本来就是在-20到20之间，夹逼也没什么作用
        # print(log_std)
        std = torch.exp(log_std)                                    # choose : [1,4] , exercise :  [256,6]

        mean = mean[:,0,:]  # 中间取个0，可以把第二维去掉了
        std = std[:,0,:]

        # print("log_std : ", log_std, " std : ",std)
        # print(std)
        return TanhNormal(mean, std)

    def act(self,input,max):
        dist = self.act_dist(input,max)  # -->list():  [[]]
        act= dist.sample()

        return act

    def evaluate(self,state, action):
        action_mean = self.act_mean(state.cpu().data.numpy()) # -->np.array: [[],[],[]]
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)  # 相对熵 对数概率
        dist_entropy = dist.entropy()  # 熵
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)   # self.dim = 1
        return super().forward(flat_inputs, **kwargs)
class DRL_SAC:
    def __init__(self,state_dim, action_dim, max_action,policy_rate, critic_rate,alpha_lr,reward_scale):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_lr = policy_rate
        self.qf_lr = critic_rate
        self.alpha_lr = alpha_lr
        self.epsilon_max = 0.9
        self.eps_clip = 0.2      # 0.2
        self.c1 = 0.01
        self.c2 = 1
        self.soft_target_tau = 0.005   # 0.005: 用于target 网络更新参数
        self.policy = ActorCritic( input_size= self.state_dim ,
                                   output_size = self.action_dim,
                                 ).to(device) # 创建 action_std =0.5 作用分布的常数标准差（多元正态

        self.qf1 = ConcatMlp(input_size= self.state_dim  + self.action_dim,output_size=1,hidden_sizes=[256, 256],)
        self.qf2 = ConcatMlp(input_size= self.state_dim  + self.action_dim,output_size=1,hidden_sizes=[256, 256],)
        self.target_qf1 = ConcatMlp(input_size= self.state_dim  + self.action_dim, output_size=1, hidden_sizes=[256, 256],)
        self.target_qf2 = ConcatMlp(input_size= self.state_dim  + self.action_dim, output_size=1, hidden_sizes=[256, 256],)

        self.memory = Memory(500,self.state_dim ,self.action_dim)
        self.use_automatic_entropy_tuning = True  # True
        self.train(policy_lr=self.policy_lr, qf_lr=self.qf_lr, alpha_lr  =self.alpha_lr,reward_scale=reward_scale)

        self.max_action = max_action
        self.alpha_loss = torch.tensor(0)
        self.alpha = torch.tensor([0], dtype=torch.float32)
        self.loss = SACLosses(
            policy_loss=torch.tensor([0], dtype=torch.float32),
            qf1_loss=torch.tensor([0], dtype=torch.float32),
            qf2_loss=torch.tensor([0], dtype=torch.float32),
            alpha_loss=torch.tensor([0], dtype=torch.float32),
        )

    def select_action(self,state,veh_id):

        state = torch.Tensor(state[np.newaxis,:])  # np.newaxis = none 第一维度为0

        act_pre = self.policy.act(state,self.max_action) # 这个是需要存储起来的
        act = act_pre * self.max_action
        act = act.clamp(0,self.max_action)
        # act = np.abs(act)
        return state, act_pre, act


    def train(self,policy_lr,qf_lr,alpha_lr,reward_scale,optimizer_class = optim.Adam):
        self.target_update_period = 1  # target 网络更新参数间隔
        self.target_entropy = None
        if self.use_automatic_entropy_tuning:
            if self.target_entropy is None:
                # Use heuristic value from SAC paper
                # 使用SAC论文中的启发值
                self.target_entropy = -np.prod( self.action_dim).item() # 连乘操作---维度相乘 = 6 : 这个数应该就是论文里面的 H = dim(a)  # 动作为6维, 每个action取值为 -1 和 1 = Box([-1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1.], (6,), float32)
            else:
                self.target_entropy = self.target_entropy

        self.alpha = torch.tensor([0], dtype=torch.float32)
        self.log_pi = torch.tensor([0], dtype=torch.float32)

        self.q_new_actions = torch.tensor([0], dtype=torch.float32)
        self.new_obs_actions = torch.tensor([0], dtype=torch.float32)
        self.q1 = torch.tensor([0], dtype=torch.float32)
        self.q2 = torch.tensor([0], dtype=torch.float32)
        self.q_target = torch.tensor([0], dtype=torch.float32)
        self.train_r = torch.tensor([0], dtype=torch.float32)
        self.target_q_values = torch.tensor([0], dtype=torch.float32)

        self.log_alpha = ptu.zeros(1, requires_grad=True)       # self.log_alpha = tensor([0.],requires_grad = True)
        self.alpha_optimizer = optimizer_class( [self.log_alpha],lr=alpha_lr)

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(self.policy.parameters(),lr=policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(),lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(),lr=qf_lr)
        self.discount = 0.99  # 0.99
        self.reward_scale = reward_scale  # 1
        self._n_train_steps_total = 0 # self._n_train_steps_total % self.target_update_period 更新target

    def update_alpha(self,alpha,episode,optimizer_class = optim.Adam):

        if episode == 0:

            self.alpha = torch.tensor([0], dtype=torch.float32)
            self.log_alpha = ptu.zeros(1, requires_grad=True)  # self.log_alpha = tensor([0.],requires_grad = True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr= self.alpha_lr)
        else:
            log_alpha = np.log(alpha)
            self.alpha = torch.tensor([alpha],dtype=torch.float32)
            # self.log_alpha = log_alpha #
            self.log_alpha = torch.tensor(log_alpha, requires_grad=True)
            # self.alpha_optimizer = optimizer_class([self.log_alpha], lr= self.alpha_lr,weight_decay=1e-4)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=self.alpha_lr)

    def decayrate(self,decayRate):

        self.policy_lr = self.policy_lr * decayRate
        self.qf_lr = self.qf_lr * decayRate
        # self.alpha_lr = self.alpha_lr * decayRate

    def compute_loss( self,s,r,a,s_,d):

        dist = self.policy.act_dist(s,self.max_action)
        new_obs_actions, log_pi = dist.rsample_and_logprob()  # new_obs_actions [256,4] , log_pi= {Size:1} 256
        log_pi = log_pi.unsqueeze(-1)  # log_pi : log(pi(a|s)) 变成 2 维
        if self.use_automatic_entropy_tuning:
            # 为什么这里是 self.log_alpha = 0(初始值) ，论文里是 alpha ？？？？？？？？？？？？？
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean() # self.target_entropy = dim(a) < 0 ?
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        self.alpha = alpha
        self.alpha_loss = alpha_loss.detach()

        s = torch.squeeze(s, dim = 1)
        q_new_actions = torch.min(                   # --> [256,1]
            #self.qf1(s[:,:32], new_obs_actions),     # Q1(st,at+1)
            #self.qf2(s[:,:32], new_obs_actions),     # Q2(st,at+1)
            self.qf1(s, new_obs_actions),     # Q1(st,at+1)
            self.qf2(s, new_obs_actions),     # Q2(st,at+1)
        )

        policy_loss = (alpha*log_pi - q_new_actions).mean() # 注意取均值 。 q_new_actions 为什么是下一个状态 new_obs_action
        """
        QF Loss
        """
        # 用来产生 t+1 时刻的 Q 值，需要用到 t+1 时刻的 pi

        q1_pre = self.qf1(s, a)                 # Q1(st,at) --> [256,1]
        q2_pre = self.qf2(s, a)                 # Q2(st,at) --> [256,1]
        "=============================================================="
        # q1_pred = q1_pre + q1Noise
        # q2_pred = q2_pre + q2Noise
        q1_pred = q1_pre
        q2_pred = q2_pre
        "=============================================================="
        next_dist = self.policy.act_dist(s_,self.max_action)     # s(t+1) 时刻的 pi ----> 产生t+1时刻的at+1
        s_ = torch.squeeze(s_, dim=1)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob() # new_next_actions: [256,4] , new_log_pi = {Size:1} 256
        new_log_pi = new_log_pi.unsqueeze(-1)     # 变成 2 维 : new_log_pi = {Size:1} 256--> new_log_pi = {Size:2} [256,1]

        target_q_values = torch.min(
            self.target_qf1(s_, new_next_actions),  # s(t+1),a(t+1)时刻的Q
            self.target_qf2(s_, new_next_actions),
        ) - alpha * new_log_pi

        #q_target = self.reward_scale * r.unsqueeze(-1).to(torch.float32) + self.discount * target_q_values  # r = {Size:2}[256,1]
        q_target = self.reward_scale * r.to(torch.float32) + self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach()) # qf_criterion = nn.MSELoss()
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        # if qf1_loss>10000 or qf2_loss > 10000:
        #     print("q1")
        #     print(q1_pre)
        #     print("q2")
        #     print(q2_pre)
        #     print("q1 target")
        #     print(self.target_qf1(s_, new_next_actions))
        #     print("q2 target")
        #     print(self.target_qf2(s_, new_next_actions))
        #     print("target = r + min(q1 target,q2 target)")
        #     print(q_target)
        #     print(qf1_loss)
        #     print(qf2_loss)
        #     print("mimus")
        #     print(q1_pred-q_target)
        #
        #
        #     print("stop")

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )

        # print(" A loss : ",policy_loss)
        # print("Q1 loss : ", policy_loss)
        # print("Q2 loss : ", policy_loss)
        return loss,alpha,log_pi,\
               q_new_actions,new_next_actions,\
               q1_pred,q2_pred,q_target,r,target_q_values

    def train_from_torch(self, s,r,a,s_,d):

        losses ,alpha,log_pi,q_new_actions,new_obs_actions,q1_pred,q2_pred,q_target,r,target_q_values = self.compute_loss(s,r,a,s_,d)
        self.new_obs_actions = new_obs_actions
        self.q1 = q1_pred
        self.q2 = q2_pred
        self.q_target = q_target
        self.train_r = r
        self.target_q_values = target_q_values
        # print(losses)
        """
        Update networks
        """
        # print("actor loss : " , losses.policy_loss)
        # print("  qf1_loss : ", losses.qf1_loss)
        # print("  qf2_loss : ", losses.qf2_loss)

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward(retain_graph=True)

            self.alpha_optimizer.step()



        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward(retain_graph=True)
        # max_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward(retain_graph=True)
        # max_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), max_norm)
        self.qf1_optimizer.step()
        #print(self.qf1.state_dict())

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward(retain_graph=True)
        # max_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), max_norm)
        self.qf2_optimizer.step()


        self._n_train_steps_total += 1
        self.try_update_target_networks()


        return  losses,log_pi,q_new_actions,q1_pred,q2_pred,q_target


    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def processReward(self,rewards):
        reward_mean = rewards.mean()
        reward_std = rewards.std()

        # 对reward进行归一化处理
        normalized_rewards = (rewards - reward_mean) / reward_std
        return normalized_rewards


    def getGradient(self,Training_number):

        policy_loss = []
        q1_loss = []
        q2_loss =[]
        alpha_loss = []
        for it in range(Training_number):
            if self.memory.memory_counter > self.memory.memory_size:
                    sample_index = np.random.choice(self.memory.memory_size, size = 128)
            else:
                    sample_index = np.random.choice(self.memory.memory_counter, size= 128)


            Traindata = self.memory.memory[sample_index, :]


            x = torch.tensor(Traindata[:, :self.state_dim], dtype=torch.float32)
            u = torch.tensor(Traindata[:, self.state_dim: self.state_dim + self.action_dim], dtype=torch.float32)
            r = torch.tensor(Traindata[:, self.state_dim + self.action_dim], dtype=torch.float32)
            y = torch.tensor(Traindata[:, self.state_dim + self.action_dim+1:2 * self.state_dim + self.action_dim+1], dtype=torch.float32)
            d  = torch.tensor(Traindata[:, -1], dtype=torch.float32)


            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            reward = torch.unsqueeze(reward , dim=1)
            # reward = self.processReward(reward)
            done = torch.unsqueeze(done,dim = 1)

            losses ,_,_,_,_,_,_,_,_,_ = self.compute_loss(state,reward,action,next_state,done)

            policy_loss.append(losses._asdict()["policy_loss"].detach().numpy())
            q1_loss.append(losses._asdict()["qf1_loss"].detach().numpy())
            q2_loss.append(losses._asdict()["qf2_loss"].detach().numpy())
            alpha_loss.append(losses._asdict()["alpha_loss"].detach().numpy())

            self.policy_optimizer.zero_grad()
            losses.policy_loss.backward(retain_graph=True)


            self.qf1_optimizer.zero_grad()
            losses.qf1_loss.backward(retain_graph=True)


            self.qf2_optimizer.zero_grad()
            losses.qf2_loss.backward(retain_graph=True)


        return dict(policy_loss = np.mean(policy_loss),qf1_loss = np.mean(q1_loss),
                        qf2_loss = np.mean(q2_loss) , alpha_loss = np.mean(alpha_loss))


    def update(self,Training_number):

        for it in range(Training_number):

            if self.memory.memory_counter > self.memory.memory_size:
                    sample_index = np.random.choice(self.memory.memory_size, size = 128)
            else:
                    sample_index = np.random.choice(self.memory.memory_counter, size= 128)

            Traindata = self.memory.memory[sample_index, :]

            x = torch.tensor(Traindata[:, :self.state_dim], dtype=torch.float32)
            u = torch.tensor(Traindata[:, self.state_dim: self.state_dim + self.action_dim], dtype=torch.float32)
            r = torch.tensor(Traindata[:, self.state_dim + self.action_dim], dtype=torch.float32)
            y = torch.tensor(Traindata[:, self.state_dim + self.action_dim+1:2 * self.state_dim + self.action_dim+1], dtype=torch.float32)
            d  = torch.tensor(Traindata[:, -1], dtype=torch.float32)


            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            reward = torch.unsqueeze(reward , dim=1)
            # reward = self.processReward(reward)
            done = torch.unsqueeze(done,dim = 1)

            losses,log_pi,q_new_actions,q1,q2,q_target = self.train_from_torch(state, reward, action, next_state,done)

            self.loss = losses
            self.log_pi = log_pi
            self.q_new_actions = q_new_actions


    def save(self, directory):
        torch.save(self.policy.state_dict(), '{}_policy_network.pth'.format(directory))
        torch.save(self.qf1.state_dict(), '{}_qf1_network.pth'.format(directory))
        torch.save(self.qf2.state_dict(), '{}_qf2_network.pth'.format(directory))
        torch.save(self.target_qf1.state_dict(), '{}_target_qf1_network.pth'.format(directory))
        torch.save(self.target_qf2.state_dict(), '{}_target_qf2_network.pth'.format(directory))
        print("====================================")
        print("SAC Model has been saved...")
        print("====================================")

    def load(self, directory):
        self.policy.load_state_dict(torch.load('{}_policy_network.pth'.format(directory)))
        self.qf1.load_state_dict(torch.load('{}_qf1_network.pth'.format(directory)))
        self.qf2.load_state_dict(torch.load('{}_qf2_network.pth'.format(directory)))
        self.target_qf1.load_state_dict(torch.load('{}_target_qf1_network.pth'.format(directory)))
        self.target_qf1.load_state_dict(torch.load('{}_target_qf2_network.pth'.format(directory)))

        print("====================================")
        print("model has been loaded...")









