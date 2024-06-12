import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties  # 导入FontProperties

# font = FontProperties(fname="SimHei.ttf", size=14)  # 设置字体

def moving_average(original_curve, weight):
    # 创建一个权重数组，其中包含平均窗口内的权重（全部为1，窗口大小为window_size）
    # last = data[0]  # First value in the plot (first timestep)
    # smoothed = list()

    # 平滑窗口大小（窗口大小越大，平滑效果越明显）
    window_size = 5

    # 使用移动平均滤波器平滑曲线
    smoothed_curve = []
    for i in range(len(original_curve)):
        start = max(0, i - window_size // 2)
        end = min(len(original_curve), i + window_size // 2 + 1)
        window = original_curve[start:end]
        smoothed_value = np.mean(window)
        smoothed_curve.append(smoothed_value)

    # for point in data:
    #     smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
    #     smoothed.append(smoothed_val)  # Save it
    #     last = smoothed_val
    return smoothed_curve

"GNN50"
weight= 0.1
data = np.genfromtxt("../save/MyCode1/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN50 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn50 = moving_average(v,weight)

"GNN100"
# weight= 0.9
data = np.genfromtxt("../save/Mycode_size1/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN100 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn100 = moving_average(v,weight)

"GNN25"
# weight= 0.9
data = np.genfromtxt("../save/Mycode_size2/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN25 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn25 = moving_average(v,weight)

"GNN20"
# weight= 0.9
data = np.genfromtxt("../save/Mycode_size3/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN20 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn20 = moving_average(v,weight)

"GNN125"
# weight= 0.9
data = np.genfromtxt("../save/MyCode_size4/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN125 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn125 = moving_average(v,weight)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # 科学计数法
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))# 科学计数法
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10} # 字体大小，相当于也是leged框的大

# plt.plot(Episode_GNN20 * 0.02,v_gnn20, linewidth=3.0)
plt.plot(Episode_GNN20 * 0.02,v_gnn20, linewidth=3.0)
plt.plot(Episode_GNN25 * 0.02, v_gnn25, linewidth=3.0)
plt.plot(Episode_GNN50 * 0.02,v_gnn50, linewidth=3.0)
plt.plot(Episode_GNN100* 0.02 ,v_gnn100, linewidth=3.0)
# plt.plot(Episode_GNN25,v_gnn25, linewidth=3.0)
# plt.plot(Episode_GNN20,v_gnn20, linewidth=3.0)
# plt.plot(Episode_GNN125 * 0.02,v_gnn125, linewidth=3.0)

# custom_labels = ['A', 'B', 'C', 'D', 'E']

# plt.xticks(Episode_GNN20, custom_labels)
plt.legend(["$L_g = 20$ (m)",
            "$L_g = 25$ (m)",
            "$L_g = 50$ (m)",
            "$L_g = 100$ (m)",
            # "$L_g = 125$ (m)"
            ],
           loc='upper left', bbox_to_anchor=(0.4, 0.8),fancybox=True, shadow=False, prop=font2)

plt.grid(True, linestyle='--')
# plt.yscale("log")
# plt.xscale("log")
# plt.xlabel("Traning time")
# plt.ylabel("Average system AOI (ms) ")
# mpl.rcParams['font.family'] = 'STKAITI' #'STKAITI'——字体

# font = FontProperties(fname="SimHei.ttf", size=14)  # 步骤二
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.xlabel("训练回合")
plt.ylabel("系统平均信息年龄(ms) ")
# plt.savefig("../save/Test/Picture/Training_GNNsize.eps")
# plt.savefig("../save/Test/Picture/Training_GNNsize.jpg")


plt.show()