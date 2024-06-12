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

"GNN30354050"
weight= 0.1
data = np.genfromtxt("../save/MyCode1/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN30354050 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn30354050 = moving_average(v,weight)

"GNN6090100120"
# weight= 0.1
data = np.genfromtxt("../save/MyCode_speed1/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN6090100120 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn6090100120 = moving_average(v,weight)

"GNN50505050"
# weight= 0.9
data = np.genfromtxt("../save/Mycode_speed2/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN50505050 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn50505050 = moving_average(v,weight)

"GNN607080100"
# weight= 0.9
data = np.genfromtxt("../save/Mycode_speed3/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN607080100 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn607080100 = moving_average(v,weight)

"GNN80808080"
# weight= 0.9
data = np.genfromtxt("../save/MyCode_speed4/AOI.csv",delimiter=",",skip_header=1)
Episode_GNN80808080 = data[:,1]
v = data[:,2]
# v_gnn = v
v_gnn80808080 = moving_average(v,weight)
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 7} # 字体大小，相当于也是leged框的大
plt.plot(Episode_GNN30354050,v_gnn30354050, linewidth=3.0)
plt.plot(Episode_GNN6090100120,v_gnn6090100120, linewidth=3.0)
plt.plot(Episode_GNN50505050,v_gnn50505050, linewidth=3.0)
plt.plot(Episode_GNN607080100,v_gnn607080100, linewidth=3.0)
plt.plot(Episode_GNN80808080,v_gnn80808080, linewidth=3.0)

plt.legend(["$v_{\max}^1= 30$, $v_{\max}^2= 35$, $v_{\max}^3= 40$, $v_{\max}^4= 50$ ($km$/h)",
            "$v_{\max}^1= 60$, $v_{\max}^2= 90$, $v_{\max}^3= 100$, $v_{\max}^4= 120$ ($km$/h)",
            "$v_{\max}^1= 50$, $v_{\max}^2= 50$, $v_{\max}^3= 50$, $v_{\max}^4= 50$ ($km$/h)",
            "$v_{\max}^1= 60$, $v_{\max}^2= 70$, $v_{\max}^3= 80$, $v_{\max}^4= 100$ ($km$/h)",
            "$v_{\max}^1= 80$, $v_{\max}^2= 80$, $v_{\max}^3= 80$, $v_{\max}^4= 80$ ($km$/h)"],
           loc='upper left', bbox_to_anchor=(0.4, 1.0),fancybox=True, shadow=False, prop=font2)
plt.grid(True, linestyle='--')
# plt.yscale("log")
plt.xlabel("Traning time")
plt.ylabel("Average system AOI (ms) ")
plt.show()