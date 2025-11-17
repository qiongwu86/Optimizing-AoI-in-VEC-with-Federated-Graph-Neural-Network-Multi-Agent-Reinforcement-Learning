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
        # 检查TrainData是否包含NaN或无穷大值
        if np.isnan(TrainData).any() or np.isinf(TrainData).any():
            print("Warning: NaN or Inf detected in SAC data, skipping storage")
            import pdb
            pdb.set_trace()
            return  # 不存储坏数据
            
        idx = self.memory_counter % self.memory_size
        self.memory[idx,:] = np.array(TrainData)
        self.memory_counter += 1





class ActorCritic(nn.Module):
    def __init__(self,input_size,output_size,hidden_init=ptu.fanin_init,b_init_value=0.,init_w=3e-3,hidden_size=64):
        super(ActorCritic, self).__init__()

        # 使用可配置的网络结构，默认64，可设置为64, 128, 256
        self.hidden1 = nn.Linear(input_size, hidden_size)
        hidden_init(self.hidden1.weight)
        self.hidden1.bias.data.fill_(b_init_value)

        self.hidden2= nn.Linear(hidden_size, hidden_size)
        hidden_init(self.hidden2.weight)
        self.hidden2.bias.data.fill_(b_init_value)
        #

        self.last_mean = nn.Linear(hidden_size,output_size)
        self.last_mean.weight.data.uniform_(-init_w, init_w)
        self.last_mean.bias.data.fill_(0)

        self.last_fc_log_std = nn.Linear(hidden_size,output_size)  # 输出层
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
        
        # 只在结果为NaN或Inf时打印
        if torch.isnan(std).any() or torch.isinf(std).any():
            print(f"⚠️⚠️⚠️[DEBUG] SAC policy: log_std = {log_std}, std = {std}")
            import pdb
            pdb.set_trace()

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
    def __init__(self,state_dim, action_dim, max_action,policy_rate, critic_rate,alpha_lr,reward_scale,hidden_size=64):
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
                                   hidden_size=hidden_size
                                 ).to(device) # 创建 action_std =0.5 作用分布的常数标准差（多元正态

        # 使用可配置的Q网络结构，默认64，可设置为64, 128, 256
        self.qf1 = ConcatMlp(input_size= self.state_dim  + self.action_dim,output_size=1,hidden_sizes=[hidden_size, hidden_size],)
        self.qf2 = ConcatMlp(input_size= self.state_dim  + self.action_dim,output_size=1,hidden_sizes=[hidden_size, hidden_size],)
        self.target_qf1 = ConcatMlp(input_size= self.state_dim  + self.action_dim, output_size=1, hidden_sizes=[hidden_size, hidden_size],)
        self.target_qf2 = ConcatMlp(input_size= self.state_dim  + self.action_dim, output_size=1, hidden_sizes=[hidden_size, hidden_size],)

        self.memory = Memory(20000,self.state_dim ,self.action_dim)
        self.use_automatic_entropy_tuning = True  # True # 自动调整熵系数
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
        # 检查输入数据是否有NaN或无穷大值
        if torch.isnan(s).any() or torch.isinf(s).any():
            print(" ❗Warning: NaN or Inf detected in state input")
            import pdb
            pdb.set_trace()
            s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(r).any() or torch.isinf(r).any():
            print("❗❗Warning: NaN or Inf detected in reward input")
            import pdb
            pdb.set_trace()
            r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(a).any() or torch.isinf(a).any():
            print("❗❗❗Warning: NaN or Inf detected in action input")
            import pdb
            pdb.set_trace()
            a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(s_).any() or torch.isinf(s_).any():
            print("❗❗❗❗Warning: NaN or Inf detected in next state input")
            import pdb
            pdb.set_trace()
            s_ = torch.nan_to_num(s_, nan=0.0, posinf=1.0, neginf=-1.0)

        dist = self.policy.act_dist(s,self.max_action)
        new_obs_actions, log_pi = dist.rsample_and_logprob()  # new_obs_actions [256,4] , log_pi= {Size:1} 256
        log_pi = log_pi.unsqueeze(-1)  # log_pi : log(pi(a|s)) 变成 2 维
        
        # 检查log_pi是否有异常值
        if torch.isnan(log_pi).any() or torch.isinf(log_pi).any():
            
            print(" ❗ ⚠️ ❗ ⚠️ Warning: NaN or Inf detected in log_pi")
            import pdb
            pdb.set_trace()
            log_pi = torch.nan_to_num(log_pi, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean() # self.target_entropy = dim(a) < 0 
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
        
        # 检查Q值是否有异常值
        if torch.isnan(q_new_actions).any() or torch.isinf(q_new_actions).any():
            
            print(" ❗⚠️ ❗⚠️ Warning: NaN or Inf detected in q_new_actions")
            print(f"  s shape: {s.shape}, s range: [{s.min():.6f}, {s.max():.6f}]")
            print(f"  new_obs_actions shape: {new_obs_actions.shape}, range: [{new_obs_actions.min():.6f}, {new_obs_actions.max():.6f}]")
            print(f"  qf1 output range: [{self.qf1(s, new_obs_actions).min():.6f}, {self.qf1(s, new_obs_actions).max():.6f}]")
            print(f"  qf2 output range: [{self.qf2(s, new_obs_actions).min():.6f}, {self.qf2(s, new_obs_actions).max():.6f}]")
            import pdb
            pdb.set_trace()
            q_new_actions = torch.nan_to_num(q_new_actions, nan=0.0, posinf=1.0, neginf=-1.0)

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

    def check_network_weights(self):
        """检查网络权重是否包含NaN值"""
        for name, param in self.policy.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN/Inf detected in policy.{name}")
                import pdb
                pdb.set_trace()
                return False
        for name, param in self.qf1.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN/Inf detected in qf1.{name}")
                import pdb
                pdb.set_trace()
                return False
        for name, param in self.qf2.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN/Inf detected in qf2.{name}")
                import pdb
                pdb.set_trace()
                return False
        return True

    def _create_dummy_losses(self):
        """创建虚拟的损失对象，用于跳过训练时返回"""
        class DummyLosses:
            def __init__(self):
                self.policy_loss = torch.tensor(0.0)
                self.qf1_loss = torch.tensor(0.0)
                self.qf2_loss = torch.tensor(0.0)
                self.alpha_loss = torch.tensor(0.0)
        
        dummy_losses = DummyLosses() # 创建虚拟的损失对象，用于跳过训练时返回
        dummy_log_pi = torch.tensor(0.0)
        dummy_q_new_actions = torch.tensor(0.0)
        dummy_q1_pred = torch.tensor(0.0)
        dummy_q2_pred = torch.tensor(0.0)
        dummy_q_target = torch.tensor(0.0)
        
        return dummy_losses, dummy_log_pi, dummy_q_new_actions, dummy_q1_pred, dummy_q2_pred, dummy_q_target
    
    def _check_and_fix_batch_data(self, s, r, a, s_, d):
        """检查并修复batch数据中的NaN或Inf值"""
        data_valid = True
        fixed_data = False
        
        # 检查并修复状态数据 s
        if torch.isnan(s).any() or torch.isinf(s).any():
            print(f"❌ Invalid state data detected:")
            print(f"   Shape: {s.shape}")
            print(f"   NaN count: {torch.isnan(s).sum().item()}")
            print(f"   Inf count: {torch.isinf(s).sum().item()}")
            print(f"   Min value: {s.min().item():.6f}")
            print(f"   Max value: {s.max().item():.6f}")
            import pdb
            pdb.set_trace()
            
            # 修复异常值
            s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=-1.0)
            s = torch.clamp(s, -10.0, 10.0)  # 限制范围
            print("   ✅ State data fixed")
            fixed_data = True
            import pdb
            pdb.set_trace()
        
        # 检查并修复奖励数据 r
        if torch.isnan(r).any() or torch.isinf(r).any():
            print(f"❌ Invalid reward data detected:")
            print(f"   Shape: {r.shape}")
            print(f"   NaN count: {torch.isnan(r).sum().item()}")
            print(f"   Inf count: {torch.isinf(r).sum().item()}")
            print(f"   Min value: {r.min().item():.6f}")
            print(f"   Max value: {r.max().item():.6f}")
            
            # 修复异常值
            r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=-1.0)
            r = torch.clamp(r, -100.0, 100.0)  # 限制奖励范围
            print("   ✅ Reward data fixed")
            fixed_data = True
            import pdb
            pdb.set_trace()     
        
        # 检查并修复动作数据 a
        if torch.isnan(a).any() or torch.isinf(a).any():
            print(f"❌ Invalid action data detected:")
            print(f"   Shape: {a.shape}")
            print(f"   NaN count: {torch.isnan(a).sum().item()}")
            print(f"   Inf count: {torch.isinf(a).sum().item()}")
            print(f"   Min value: {a.min().item():.6f}")
            print(f"   Max value: {a.max().item():.6f}")
            
            # 修复异常值
            a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
            a = torch.clamp(a, -1.0, 1.0)  # 限制动作范围
            print("   ✅ Action data fixed")
            fixed_data = True
            import pdb
        
        # 检查并修复下一状态数据 s_
        if torch.isnan(s_).any() or torch.isinf(s_).any():
            print(f"❌ Invalid next state data detected:")
            print(f"   Shape: {s_.shape}")
            print(f"   NaN count: {torch.isnan(s_).sum().item()}")
            print(f"   Inf count: {torch.isinf(s_).sum().item()}")
            print(f"   Min value: {s_.min().item():.6f}")
            print(f"   Max value: {s_.max().item():.6f}")
            
            
            # 修复异常值
            s_ = torch.nan_to_num(s_, nan=0.0, posinf=1.0, neginf=-1.0)
            s_ = torch.clamp(s_, -10.0, 10.0)  # 限制范围
            print("   ✅ Next state data fixed")
            fixed_data = True
            import pdb
        
        # 检查并修复终止标志数据 d
        if torch.isnan(d).any() or torch.isinf(d).any():
            print(f"❌ Invalid done flag data detected:")
            print(f"   Shape: {d.shape}")
            print(f"   NaN count: {torch.isnan(d).sum().item()}")
            print(f"   Inf count: {torch.isinf(d).sum().item()}")
            print(f"   Min value: {d.min().item():.6f}")
            print(f"   Max value: {d.max().item():.6f}")
            
            # 修复异常值
            d = torch.nan_to_num(d, nan=0.0, posinf=1.0, neginf=-1.0)
            d = torch.clamp(d, 0.0, 1.0)  # 限制为0或1
            print("   ✅ Done flag data fixed")
            fixed_data = True
            import pdb
            pdb.set_trace()
        
        # 检查数据范围是否合理
        # if data_valid:
            # 检查状态数据范围
            # if s.abs().max() > 1000:
            #     print(f"⚠️ Warning: State values out of reasonable range: max={s.abs().max().item():.6f}")
            #     s = torch.clamp(s, -10.0, 10.0)
            #     fixed_data = True
            #     import pdb
            
            # # 检查奖励数据范围
            # if r.abs().max() > 1000:
            #     print(f"⚠️ Warning: Reward values out of reasonable range: max={r.abs().max().item():.6f}")
            #     r = torch.clamp(r, -100.0, 100.0)
            #     fixed_data = True
            #     import pdb
            #     pdb.set_trace() 

            
            # # 检查动作数据范围
            # if a.abs().max() > 1000:
            #     print(f"⚠️ Warning: Action values out of reasonable range: max={a.abs().max().item():.6f}")
            #     a = torch.clamp(a, -1.0, 1.0)
            #     fixed_data = True
            #     import pdb
            #     pdb.set_trace()

        # 打印修复后的数据形状
        if fixed_data:
            print(f"✅ Batch data fixed and validated - Shape: s{s.shape}, r{r.shape}, a{a.shape}, s_{s_.shape}, d{d.shape}") # 打印修复后的数据形状
            import pdb
            pdb.set_trace() 
           
        else:
            # print(f"✅ Batch data validation passed - Shape: s{s.shape}, r{r.shape}, a{a.shape}, s_{s_.shape}, d{d.shape}") # 打印数据形状
            pass

        
        return data_valid, s, r, a, s_, d

    def train_from_torch(self, s,r,a,s_,d):
        # 首先检查网络权重  如果参数里有 NaN/Inf，跳过本次训练并返回占位损失 _create_dummy_losses()
        if not self.check_network_weights(): # 检查网络权重是否包含NaN值
            print("⚠️ ⚠️ Warning: Network weights contain NaN/Inf, skipping training")
            import pdb
            pdb.set_trace()
            return self._create_dummy_losses() # 创建虚拟的损失对象，用于跳过训练时返回：跳过本次训练并返回占位损失

        # 详细检查batch数据中的NaN/Inf值
        batch_data_valid, s, r, a, s_, d = self._check_and_fix_batch_data(s, r, a, s_, d)
        if not batch_data_valid:
            print("⚠️ ⚠️ Warning: Invalid batch data detected, skipping this training step")
            import pdb
            pdb.set_trace()
            return self._create_dummy_losses()


        losses ,alpha,log_pi,q_new_actions,new_obs_actions,q1_pred,q2_pred,q_target,r,target_q_values = self.compute_loss(s,r,a,s_,d)
        
        # 检查是否有NaN值
        skip_training = False
        if torch.isnan(losses.policy_loss) or torch.isnan(losses.qf1_loss) or torch.isnan(losses.qf2_loss):
            print("⚠️ ⚠️ Warning: NaN detected in losses, skipping this training step")   
            import pdb
            pdb.set_trace()
            skip_training = True
        
        # 检查输入数据是否有NaN
        if torch.isnan(s).any() or torch.isnan(r).any() or torch.isnan(a).any() or torch.isnan(s_).any():
            print("⚠️ ⚠️ Warning: NaN detected in input data, skipping this training step") 
            import pdb
            pdb.set_trace()
            skip_training = True
            
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

        # 只有在没有检测到NaN值时才进行训练
        if not skip_training:
            # 优化：先计算并保存loss值，然后分别backward，避免retain_graph导致的内存泄漏
            # 先处理qf1和qf2的loss（它们相互独立）
            self.qf1_optimizer.zero_grad()
            losses.qf1_loss.backward()  # 移除 retain_graph=True
            # 添加梯度裁剪防止梯度爆炸
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), max_norm)
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            losses.qf2_loss.backward()  # 移除 retain_graph=True
            # 添加梯度裁剪防止梯度爆炸
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), max_norm)
            self.qf2_optimizer.step()

            # 然后处理policy loss（可能需要重新计算，因为它可能依赖qf1/qf2）
            # 但为了保持代码结构，我们尝试直接backward
            # 如果policy_loss的计算依赖于已释放的计算图，需要重新计算loss
            self.policy_optimizer.zero_grad()
            # 重新计算policy_loss以确保计算图完整（如果需要）
            # 如果compute_loss中policy_loss不依赖qf1/qf2的梯度，可以直接backward
            try:
                losses.policy_loss.backward()  # 移除 retain_graph=True
            except RuntimeError as e:
                # 如果计算图已被释放，需要重新计算loss
                if "has been freed" in str(e) or "is invalid" in str(e):
                    # 重新计算loss（这会增加计算开销，但避免内存泄漏）
                    losses_refresh,_,_,_,_,_,_,_,_,_ = self.compute_loss(s, r, a, s_, d)
                    losses_refresh.policy_loss.backward()
                else:
                    raise
            # 添加梯度裁剪防止梯度爆炸
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm)
            self.policy_optimizer.step()

            # 最后处理alpha loss（如果有自动熵调整）
            if self.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                try:
                    losses.alpha_loss.backward()  # 移除 retain_graph=True
                except RuntimeError as e:
                    # 如果计算图已被释放，需要重新计算loss
                    if "has been freed" in str(e) or "is invalid" in str(e):
                        losses_refresh,_,_,_,_,_,_,_,_,_ = self.compute_loss(s, r, a, s_, d)
                        losses_refresh.alpha_loss.backward()
                    else:
                        raise
                self.alpha_optimizer.step()


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

            # 优化：先计算并保存loss值，然后分别backward，避免retain_graph导致的内存泄漏
            # 注意：getGradient()方法只计算梯度，不更新参数（没有step()），所以顺序不重要
            # 但我们仍然需要确保每个loss的backward都能成功
            
            # 先backward qf1和qf2（它们相互独立）
            self.qf1_optimizer.zero_grad()
            losses.qf1_loss.backward()  # 移除 retain_graph=True

            self.qf2_optimizer.zero_grad()
            losses.qf2_loss.backward()  # 移除 retain_graph=True

            # 然后backward policy loss
            self.policy_optimizer.zero_grad()
            try:
                losses.policy_loss.backward()  # 移除 retain_graph=True
            except RuntimeError as e:
                # 如果计算图已被释放，需要重新计算loss
                if "has been freed" in str(e) or "is invalid" in str(e):
                    losses_refresh,_,_,_,_,_,_,_,_,_ = self.compute_loss(state,reward,action,next_state,done)
                    losses_refresh.policy_loss.backward()
                else:
                    raise


        # 修复：添加安全检查，防止空列表导致的nan值
        return dict(
            policy_loss=np.mean(policy_loss) if policy_loss else 0,
            qf1_loss=np.mean(q1_loss) if q1_loss else 0,
            qf2_loss=np.mean(q2_loss) if q2_loss else 0,
            alpha_loss=np.mean(alpha_loss) if alpha_loss else 0
        )


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

    def parameters(self):
        """返回所有网络参数的迭代器"""
        for net in [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]:
            for param in net.parameters():
                yield param
    
    def state_dict(self):
        """返回所有网络的状态字典"""
        return {
            'policy': self.policy.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'target_qf1': self.target_qf1.state_dict(),
            'target_qf2': self.target_qf2.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.policy.load_state_dict(state_dict['policy'])
        self.qf1.load_state_dict(state_dict['qf1'])
        self.qf2.load_state_dict(state_dict['qf2'])
        self.target_qf1.load_state_dict(state_dict['target_qf1'])
        self.target_qf2.load_state_dict(state_dict['target_qf2'])
    
    def cpu(self):
        """将所有网络移动到CPU"""
        self.policy.cpu()
        self.qf1.cpu()
        self.qf2.cpu()
        self.target_qf1.cpu()
        self.target_qf2.cpu()
        return self
    
    def to(self, device):
        """将所有网络移动到指定设备"""
        self.policy.to(device)
        self.qf1.to(device)
        self.qf2.to(device)
        self.target_qf1.to(device)
        self.target_qf2.to(device)
        return self









