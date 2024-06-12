import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# VehSpeed = [30, 30, 30, 30] : dir = 1
# VehSpeed = [40, 40, 40, 40] : dir = 2
# VehSpeed = [50, 50, 50, 50] : dir = 3
# VehSpeed = [60, 60, 60, 60] : dir = 4
# VehSpeed = [70, 70, 70, 70] : dir = 5
# VehSpeed = [80, 80, 80, 80] : dir = 6
#
"-----------------------------------------------------------------------"
"gridW = 50"
data_1= []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/MyCode_size1_speed/6/Queue.npy"
data = np.load(data_dir)
data_1.append(data)

data_dir = "../save/Test/MyCode_size1_speed/5/Queue.npy"
data = np.load(data_dir)
data_1.append(data)


data_dir = "../save/Test/MyCode_size1_speed/4/Queue.npy"
data = np.load(data_dir)
data_1.append(data)

data_dir = "../save/Test/MyCode_size1_speed/3/Queue.npy"
data = np.load(data_dir)
data_1.append(data)


data_dir = "../save/Test/MyCode_size1_speed/2/Queue.npy"
data = np.load(data_dir)
data_1.append(data)

data_dir = "../save/Test/MyCode_size1_speed/1/Queue.npy"
data = np.load(data_dir)
data_1.append(data)

"-----------------------------------------------------------------------"
"gridW = 100"
data_2= []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/MyCode_size2_speed/6/Queue.npy"
data = np.load(data_dir)
data_2.append(data)

data_dir = "../save/Test/MyCode_size2_speed/5/Queue.npy"
data = np.load(data_dir)
data_2.append(data)

data_dir = "../save/Test/MyCode_size2_speed/4/Queue.npy"
data = np.load(data_dir)
data_2.append(data)

data_dir = "../save/Test/MyCode_size2_speed/3/Queue.npy"
data = np.load(data_dir)
data_2.append(data)



data_dir = "../save/Test/MyCode_size2_speed/2/Queue.npy"
data = np.load(data_dir)
data_2.append(data)


data_dir = "../save/Test/MyCode_size2_speed/1/Queue.npy"
data = np.load(data_dir)
data_2.append(data)
"-----------------------------------------------------------------------"
"gridW = 25"
data_3 = []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/MyCode_size3_speed/6/Queue.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/MyCode_size3_speed/5/Queue.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/MyCode_size3_speed/4/Queue.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/MyCode_size3_speed/3/Queue.npy"
data = np.load(data_dir)
data_3.append(data)



data_dir = "../save/Test/MyCode_size3_speed/2/Queue.npy"
data = np.load(data_dir)
data_3.append(data)

data_dir = "../save/Test/MyCode_size3_speed/1/Queue.npy"
data = np.load(data_dir)
data_3.append(data)
"-----------------------------------------------------------------------"
"gridW = 20"
data_4 = []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/MyCode_size4_speed/6/Queue.npy"
data = np.load(data_dir)
data_4.append(data)

data_dir = "../save/Test/MyCode_size4_speed/5/Queue.npy"
data = np.load(data_dir)
data_4.append(data)

data_dir = "../save/Test/MyCode_size4_speed/4/Queue.npy"
data = np.load(data_dir)
data_4.append(data)

data_dir = "../save/Test/MyCode_size4_speed/3/Queue.npy"
data = np.load(data_dir)
data_4.append(data)


data_dir = "../save/Test/MyCode_size4_speed/2/Queue.npy"
data = np.load(data_dir)
data_4.append(data)


data_dir = "../save/Test/MyCode_size4_speed/1/Queue.npy"
data = np.load(data_dir)
data_4.append(data)
"-----------------------------------------------------------------------"
"gridW = 125"
data_5 = []
# vehGenRate = [10, 10, 10, 10]
data_dir = "../save/Test/MyCode_size5_speed/6/Queue.npy"
data = np.load(data_dir)
data_5.append(data)

data_dir = "../save/Test/MyCode_size5_speed/5/Queue.npy"
data = np.load(data_dir)
data_5.append(data)

data_dir = "../save/Test/MyCode_size5_speed/4/Queue.npy"
data = np.load(data_dir)
data_5.append(data)

data_dir = "../save/Test/MyCode_size5_speed/3/Queue.npy"
data = np.load(data_dir)
data_5.append(data)


data_dir = "../save/Test/MyCode_size5_speed/2/Queue.npy"
data = np.load(data_dir)
data_5.append(data)

data_dir = "../save/Test/MyCode_size5_speed/1/Queue.npy"
data = np.load(data_dir)
data_5.append(data)

font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10} # 字体大小，相当于也是leged框的大

plt.plot(data_4,"rs--",markersize = 8,linewidth = 3)    # 20
plt.plot(data_1,"b*-", markersize = 8, linewidth = 3)   # 50
plt.plot(data_2,"k^-.",markersize = 8,linewidth = 3)    # 100
plt.plot(data_5,"bo-.",markersize = 8, linewidth = 3)   # 125


plt.legend([
            # "$L_g = 25$ (m)",
            "$L_g = 20$ (m)",
            "$L_g = 50$ (m)",
            "$L_g = 100$ (m)",
            "$L_g = 125$ (m)"],
           loc='upper left', bbox_to_anchor=(0.7, 1),fancybox=True, shadow=False, prop=font2)

x_labels = ["80",
            "70",
            "60",
            "50",
            "40",
            "30"]

x_labels_values = [0, 1, 2, 3, 4,5]
plt.xticks(x_labels_values,x_labels)

plt.xlabel("Vehicle Speed $v^{i}_{max}$ (Km/h) ")
plt.ylabel("Average System Queue (bit)")

plt.grid(True, linestyle='--')


plt.savefig("../save/Test/Picture/gnnsize_speed_queue.eps")
plt.savefig("../save/Test/Picture/gnnsize_speed_queue.jpg")

plt.show()