import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def moving_average(data, weight):
    # 创建一个权重数组，其中包含平均窗口内的权重（全部为1，窗口大小为window_size）
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val
    return smoothed

"GNN8888"
weight= 0.9
data = np.genfromtxt("../save/Mycode1/AOI.csv",delimiter=",",skip_header=1)
# data = np.genfromtxt("F:\研究生资料\课题组资料\自己论文\第二篇\Final\save\MyCode1\AOI.csv",delimiter=",",skip_header=1)
Episode_GNN8888 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn8888 = moving_average(v,weight)

"GNN5555"
# weight= 0.9
data = np.genfromtxt("../save/Mycode2/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN5555 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn5555 = moving_average(v,weight)

"GNN10101010"
# weight= 0.9
data = np.genfromtxt("../save/Mycode3/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN10101010 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn10101010 = moving_average(v,weight)

"GNN78910"
# weight= 0.9
data = np.genfromtxt("../save/Mycode4/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN78910 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn78910 = moving_average(v,weight)

"GNN681012"
# weight= 0.9
data = np.genfromtxt("../save/Mycode5/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN681012 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn681012 = moving_average(v,weight)

MakerColor = ["y","g","b","r","m","saddlebrown"]
MakerShape = ["*","^","s","o"]
LineShape = ["-","-.","--",'-']
plt.plot(Episode_GNN8888,v_gnn8888, linewidth=3.0)
plt.plot(Episode_GNN5555,v_gnn5555, linewidth=3.0)
plt.plot(Episode_GNN10101010,v_gnn10101010, linewidth=3.0)
plt.plot(Episode_GNN78910,v_gnn78910,linewidth=3.0)
plt.plot(Episode_GNN681012,v_gnn681012,linewidth=3.0)

font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12} # 字体大小，相当于也是leged框的大
plt.legend(["$\lambda_1 = \\frac{1}{8}$, $\lambda_2 = \\frac{1}{8}$, $\lambda_3 = \\frac{1}{8}$, $\lambda_4 = \\frac{1}{8}$ (veh/s)",
            "$\lambda_1 = \\frac{1}{5}$, $\lambda_2 = \\frac{1}{5}$, $\lambda_3 = \\frac{1}{5}$, $\lambda_4 = \\frac{1}{5}$ (veh/s)",
            "$\lambda_1 = \\frac{1}{10}$, $\lambda_2 = \\frac{1}{10}$, $\lambda_3 = \\frac{1}{10}$, $\lambda_4 = \\frac{1}{10}$ (veh/s)",
            "$\lambda_1 = \\frac{1}{7}$, $\lambda_2 = \\frac{1}{8}$, $\lambda_3 = \\frac{1}{9}$, $\lambda_4 = \\frac{1}{10}$ (veh/s)",
            "$\lambda_1 = \\frac{1}{6}$, $\lambda_2 = \\frac{1}{8}$, $\lambda_3 = \\frac{1}{10}$, $\lambda_4 = \\frac{1}{12}$ (veh/s)",
            ], loc='upper left', bbox_to_anchor=(0.25, 0.85),fancybox=True, shadow=False, prop=font2)
plt.grid(True, linestyle='--')
# plt.yscale("log")
plt.xlabel("Traning time")
plt.ylabel("Average system AOI (ms) ")
plt.show()
