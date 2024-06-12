import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# vehGenRate = [5, 5, 5, 5] : dir 2
# vehGenRate = [6, 6, 6, 6] : dir 4
# vehGenRate = [7, 7, 7, 7] : dir 5
# vehGenRate = [8, 8, 8, 8] : dir 1
# vehGenRate = [9, 9, 9, 9] : dir 3
# vehGenRate = [10, 10, 10, 10] : dir 6
# 6 3 1 5 4 2
"-----------------------------------------------------------------------"
"gridW = 50"
data_1= []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/MyCode_size1_num/6/power.npy"
data = np.load(data_dir)
data_1.append(data)

data_dir = "../save/Test/MyCode_size1_num/3/power.npy"
data = np.load(data_dir)
data_1.append(data)


data_dir = "../save/Test/MyCode_size1_num/1/power.npy"
data = np.load(data_dir)
data_1.append(data)

data_dir = "../save/Test/MyCode_size1_num/5/power.npy"
data = np.load(data_dir)
data_1.append(data)


data_dir = "../save/Test/MyCode_size1_num/4/power.npy"
data = np.load(data_dir)
data_1.append(data)


data_dir = "../save/Test/MyCode_size1_num/2/power.npy"
data = np.load(data_dir)
data_1.append(data)



"-----------------------------------------------------------------------"
"gridW = 100"
data_2= []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/CompareCode1_num/6/power.npy"
data = np.load(data_dir)
data_2.append(data)

data_dir = "../save/Test/CompareCode1_num/3/power.npy"
data = np.load(data_dir)
data_2.append(data)

data_dir = "../save/Test/CompareCode1_num/1/power.npy"
data = np.load(data_dir)
data_2.append(data)

data_dir = "../save/Test/CompareCode1_num/5/power.npy"
data = np.load(data_dir)
data_2.append(data)



data_dir = "../save/Test/CompareCode1_num/4/power.npy"
data = np.load(data_dir)
data_2.append(data)

data_dir = "../save/Test/CompareCode1_num/2/power.npy"
data = np.load(data_dir)
data_2.append(data)

"-----------------------------------------------------------------------"
"gridW = 25"
data_3 = []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/CompareCode2_num/6/power.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/CompareCode2_num/3/power.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/CompareCode2_num/1/power.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/CompareCode2_num/5/power.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/CompareCode2_num/4/power.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/CompareCode2_num/2/power.npy"
data = np.load(data_dir)
data_3.append(data)

"-----------------------------------------------------------------------"
"gridW = 20"
data_4 = []
# vehGenRate = [10, 10, 10, 10]
# data_dir = "../save/Test/CompareCode3_num/3/power.npy"
# data = np.load(data_dir)
# data_4.append(data)
#
# data_dir = "../save/Test/CompareCode3_num/1/power.npy"
# data = np.load(data_dir)
# data_4.append(data)
#
# data_dir = "../save/Test/CompareCode3_num/4/power.npy"
# data = np.load(data_dir)
# data_4.append(data)
#
# data_dir = "../save/Test/CompareCode3_num/5/power.npy"
# data = np.load(data_dir)
# data_4.append(data)
#
#
# data_dir = "../save/Test/CompareCode3_num/2/power.npy"
# data = np.load(data_dir)
# data_4.append(data)

"-----------------------------------------------------------------------"
"gridW = 125"
data_5 = []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/CompareCode4_num/6/power.npy"
data = np.load(data_dir)
data_5.append(data)

data_dir = "../save/Test/CompareCode4_num/3/power.npy"
data = np.load(data_dir)
data_5.append(data)

data_dir = "../save/Test/CompareCode4_num/1/power.npy"
data = np.load(data_dir)
data_5.append(data)

data_dir = "../save/Test/CompareCode4_num/5/power.npy"
data = np.load(data_dir)
data_5.append(data)


data_dir = "../save/Test/CompareCode4_num/4/power.npy"
data = np.load(data_dir)
data_5.append(data)

data_dir = "../save/Test/CompareCode4_num/2/power.npy"
data = np.load(data_dir)
data_5.append(data)

font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10} # 字体大小，相当于也是leged框的大

# plt.plot(data_1,"b*-", markersize = 8, linewidth = 3)   # 50
# plt.plot(data_2,"k^-.",markersize = 8,linewidth = 3) # 100
# # plt.plot(data_3)                                   # 25
# plt.plot(data_4,"rs--",markersize = 8,linewidth = 3)  # 20
# plt.plot(data_5,"bo-.",markersize = 8, linewidth = 3) # 125

# plt.legend(["$L_g = 50$ (m)",
#             "$L_g = 100$ (m)",
#             "$L_g = 25$ (m)",
#             "$L_g = 20$ (m)",
#             "$L_g = 125$ (m)"],
#            loc='upper left', bbox_to_anchor=(0.3, 1),fancybox=True, shadow=False, prop=font2)

# plt.plot(data_3,"rs--",markersize = 8,linewidth = 3)    # 20
# plt.plot(data_1,"b*-", markersize = 8, linewidth = 3)   # 50
# plt.plot(data_2,"k^-.",markersize = 8,linewidth = 3) # 100
# plt.plot(data_5,"bo-.",markersize = 8, linewidth = 3) # 125

n = len(data_1)
ind = range(n)  # the x locations for the groups
width_bar = 0.2
# Plotting the bar charts
plt.bar(ind, data_3, width= width_bar, color='#205072', label='data_3',zorder=100)
plt.bar([p + width_bar for p in ind], data_1, width=width_bar, color='#329D9C', label='data_1',zorder=100)
plt.bar([p + width_bar * 2 for p in ind], data_2, width=width_bar, color='#56C596', label='data_2',zorder=100)
plt.bar([p + width_bar * 3 for p in ind], data_5, width=width_bar, color='#7BE495', label='data_5',zorder=100)

plt.legend([
            "LFSAC",
            "FGNN-MADRL",
            "GFSAC",
            "GDBR"],
           loc='upper left', bbox_to_anchor=(0.1, 1),fancybox=True, shadow=False, prop=font2)


x_labels = [r"$\frac{1}{10}$",
            r"$\frac{1}{9}$",
            r"$\frac{1}{8}$",
            r"$\frac{1}{7}$",
            r"$\frac{1}{6}$",
            r"$\frac{1}{5}$"]

x_labels_values = [p + width_bar * (1 + 0.5)  for p in ind]
plt.xticks(x_labels_values,x_labels)

plt.xlabel("Vehicle Arrival rate $\lambda$ (veh/s) ")
plt.ylabel("Average System Power (W)")

plt.grid(True, linestyle='--',zorder = 0)


plt.savefig("../save/Test/Picture/gnnsize_compare_density_power.eps")
plt.savefig("../save/Test/Picture/gnnsize_compare_density_power.jpg")

plt.show()