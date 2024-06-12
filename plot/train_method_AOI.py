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

"GNN 9"
weight= 0.1
data = np.genfromtxt("F:\研究生资料\课题组资料\自己论文\第二篇\Final\save\MyCode1\AOI.csv",delimiter=",",skip_header=1)
Episode_GNN = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn = moving_average(v,weight)

"No local aggragation"
data = np.genfromtxt("F:\研究生资料\课题组资料\自己论文\第二篇\Final\save\Compare1_Mycode2\AOI.csv",delimiter=",",skip_header=1)
Episode_Nolocal = data[:,1]
v = data[:,2]
# v_nolocal = v
v_nolocal = moving_average(v,weight)
"Average local aggregation"
data = np.genfromtxt("F:\研究生资料\课题组资料\自己论文\第二篇\Final\save\Compare2_MyCode\AOI.csv",delimiter=",",skip_header=1)
Episode_Averagelocal = data[:,1]
v = data[:,2]
# v_averagelocal = v
v_averagelocal = moving_average(v,weight)
# sns.lineplot(x=Episode_GNN, y=v_gnn)
# sns.lineplot(x=Episode_Nolocal, y=v_nolocal)
# sns.lineplot(x=Episode_Averagelocal, y=v_averagelocal)
plt.plot(Episode_GNN,v_gnn)
plt.plot(Episode_Nolocal,v_nolocal)
plt.plot(Episode_Averagelocal,v_averagelocal)
plt.legend(["GNN","No local","Average local"])
plt.grid()
# plt.yscale("log")
plt.show()