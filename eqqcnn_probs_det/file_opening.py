import numpy as np
import matplotlib.pyplot as plt
import os
'''
import os

folderpath = r"D:\my_data" # make sure to put the 'r' in front
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
all_files = []

for path in filepaths:
    with open(path, 'r') as f:
        file = f.readlines()
        all_files.append(file)
        '''

path = "/Users/francescoaldoventurelli/Desktop/file_MNIST_ref_rot_qcnn/"
#train_acc, test_acc, train_loss = np.zeros([30]), np.zeros([30]), np.zeros([30])
file_name = "mnist_ref_rot_"
last_part=".txt"

#file_f'{ }'.format("i") = open(path+file_name+str(i)+last_part, "r")
#f = open(path+file_name+str(i)+last_part, "r")
'''filepaths = [os.path.join(path, name) for name in os.listdir(path)]
print(os.listdir(path))
all_files = []
for path in filepaths:
    with open(path, 'r') as f:
        file = f.readlines()
        all_files.append(file)

for list in all_files:
    for line in list:
        for l in line:
            print(l)
        #newline = [float(l) for l in line]
        #print(type(newline))'''


'''f1 = open("graph_nn_eqv_0.txt", "r")
f2 = open("graph_nn_eqv_1.txt", "r")
f3 = open("graph_nn_eqv_2.txt", "r")
f4 = open("graph_nn_eqv_3.txt", "r")
f5 = open("graph_nn_eqv_4.txt", "r")
f6 = open("graph_nn_eqv_5.txt", "r")
f7 = open("graph_nn_eqv_6.txt", "r")
f8 = open("graph_nn_eqv_7.txt", "r")
f9 = open("graph_nn_eqv_8.txt", "r")
f10 = open("graph_nn_eqv_9.txt", "r")'''

#f1 = open("fmnist_0.txt", "r")
#f2 = open("fmnist_1.txt", "r")
#f3 = open("fmnist_2.txt", "r")
#f4 = open("fmnist_3.txt", "r")
#f5 = open("fmnist_4.txt", "r")
#f6 = open("fmnist_5.txt", "r")
#f7 = open("fmnist_6.txt", "r")
#f8 = open("fmnist_7.txt", "r")
#f9 = open("fmnist_8.txt", "r")
#f10 = open("fmnist_9.txt", "r")

#print(f1.readlines()[0].split("	")[1].split("  ")[1])


'''accuracy = np.zeros((10, 101))
#print(np.shape(accuracy[1, :]))

#iterations = [x.split('	')[0] for x in f1.readlines()]

accuracy[0, :]= [x.split('	')[1].split("  ")[0] for x in f1.readlines()]
accuracy[1, :]= [x.split('	')[1].split("  ")[0] for x in f2.readlines()]
accuracy[2, :]= [x.split('	')[1].split("  ")[0] for x in f3.readlines()]
accuracy[3, :]= [x.split('	')[1].split("  ")[0] for x in f4.readlines()]
accuracy[4, :]= [x.split('	')[1].split("  ")[0] for x in f5.readlines()]
accuracy[5, :]= [x.split('	')[1].split("  ")[0] for x in f6.readlines()]
accuracy[6, :]= [x.split('	')[1].split("  ")[0] for x in f7.readlines()]
accuracy[7, :]= [x.split('	')[1].split("  ")[0] for x in f8.readlines()]
accuracy[8, :]= [x.split('	')[1].split("  ")[0] for x in f9.readlines()]
accuracy[9, :]= [x.split('	')[1].split("  ")[0] for x in f10.readlines()]

f = open("graph_mean_sd_acc_nn_eqv_10_0.005_pbc.txt", "w")
for i in range(101):
    f.write(str(i*5) + "	" + str(np.mean(accuracy[:, i])) + "	" + str(np.std(accuracy[:, i])))
    f.write("\n")
'''

f1 = open(path+"mnist_ref_rot_1.txt", "r")
f2 = open(path+"mnist_ref_rot_2.txt", "r")
f3 = open(path+"mnist_ref_rot_3.txt", "r")
f4 = open(path+"mnist_ref_rot_4.txt", "r")
f5 = open(path+"mnist_ref_rot_5.txt", "r")
f6 = open(path+"mnist_ref_rot_6.txt", "r")
f7 = open(path+"mnist_ref_rot_7.txt", "r")
f8 = open(path+"mnist_ref_rot_8.txt", "r")
f9 = open(path+"mnist_ref_rot_9.txt", "r")
f10 = open(path+"mnist_ref_rot_10.txt", "r")
f11 = open(path+"mnist_ref_rot_11.txt", "r")
f12 = open(path+"mnist_ref_rot_12.txt", "r")
f13 = open(path+"mnist_ref_rot_13.txt", "r")
f14 = open(path+"mnist_ref_rot_14.txt", "r")
f15 = open(path+"mnist_ref_rot_15.txt", "r")
f16 = open(path+"mnist_ref_rot_16.txt", "r")
f17 = open(path+"mnist_ref_rot_17.txt", "r")
f18 = open(path+"mnist_ref_rot_18.txt", "r")
f19 = open(path+"mnist_ref_rot_19.txt", "r")
f20 = open(path+"mnist_ref_rot_20.txt", "r")


accuracy_train, accuracy_test, loss_train = np.zeros((20, 30)), np.zeros((20, 30)), np.zeros((20, 30))
#[print(x.split("  ")[2]) for x in f1.readlines()]
#print(f1.readlines())



#loss_train[0, :]= [x.split('  ')[1] for x in f1.readlines()]


'''def loss_extraction():
    loss_train[0, :] = [x.split('  ')[1] for x in f1.readlines()]
    loss_train[1, :] = [x.split('  ')[1] for x in f2.readlines()]
    loss_train[2, :] = [x.split('  ')[1] for x in f3.readlines()]
    loss_train[3, :] = [x.split('  ')[1] for x in f4.readlines()]
    loss_train[4, :] = [x.split('  ')[1] for x in f5.readlines()]
    loss_train[5, :] = [x.split('  ')[1] for x in f6.readlines()]
    loss_train[6, :] = [x.split('  ')[1] for x in f7.readlines()]
    loss_train[7, :] = [x.split('  ')[1] for x in f8.readlines()]
    loss_train[8, :] = [x.split('  ')[1] for x in f9.readlines()]
    loss_train[9, :] = [x.split('  ')[1] for x in f10.readlines()]
    loss_train[10, :] = [x.split('  ')[1] for x in f11.readlines()]
    loss_train[11, :] = [x.split('  ')[1] for x in f12.readlines()]
    loss_train[12, :] = [x.split('  ')[1] for x in f13.readlines()]
    loss_train[13, :] = [x.split('  ')[1] for x in f14.readlines()]
    loss_train[14, :] = [x.split('  ')[1] for x in f15.readlines()]
    loss_train[15, :] = [x.split('  ')[1] for x in f16.readlines()]
    loss_train[16, :] = [x.split('  ')[1] for x in f17.readlines()]
    loss_train[17, :] = [x.split('  ')[1] for x in f18.readlines()]
    loss_train[18, :] = [x.split('  ')[1] for x in f19.readlines()]
    loss_train[19, :] = [x.split('  ')[1] for x in f20.readlines()]
    #loss_av, loss_std = np.mean(loss_train, axis=0), np.std(loss_train, axis=0)
    #return loss_av, loss_std
    return [loss_train[i, :] for i in range(20)]
    #return [loss_train[1, :], loss_train[2, :], loss_train[3, :], loss_train[4, :]]
'''

'''def train_acc_extraction():
    accuracy_train[0, :] = [x.split('  ')[2] for x in f1.readlines()]
    accuracy_train[1, :] = [x.split('  ')[2] for x in f2.readlines()]
    accuracy_train[2, :] = [x.split('  ')[2] for x in f3.readlines()]
    accuracy_train[3, :] = [x.split('  ')[2] for x in f4.readlines()]
    accuracy_train[4, :] = [x.split('  ')[2] for x in f5.readlines()]
    accuracy_train[5, :] = [x.split('  ')[2] for x in f6.readlines()]
    accuracy_train[6, :] = [x.split('  ')[2] for x in f7.readlines()]
    accuracy_train[7, :] = [x.split('  ')[2] for x in f8.readlines()]
    accuracy_train[8, :] = [x.split('  ')[2] for x in f9.readlines()]
    accuracy_train[9, :] = [x.split('  ')[2] for x in f10.readlines()]
    accuracy_train[10, :] = [x.split('  ')[2] for x in f11.readlines()]
    accuracy_train[11, :] = [x.split('  ')[2] for x in f12.readlines()]
    accuracy_train[12, :] = [x.split('  ')[2] for x in f13.readlines()]
    accuracy_train[13, :] = [x.split('  ')[2] for x in f14.readlines()]
    accuracy_train[14, :] = [x.split('  ')[2] for x in f15.readlines()]
    accuracy_train[15, :] = [x.split('  ')[2] for x in f16.readlines()]
    accuracy_train[16, :] = [x.split('  ')[2] for x in f17.readlines()]
    accuracy_train[17, :] = [x.split('  ')[2] for x in f18.readlines()]
    accuracy_train[18, :] = [x.split('  ')[2] for x in f19.readlines()]
    accuracy_train[19, :] = [x.split('  ')[2] for x in f20.readlines()]
    return [accuracy_train[i, :] for i in range(20)]'''


def test_acc_extraction():
    accuracy_test[0, :] = [x.split('  ')[3] for x in f1.readlines()]
    accuracy_test[1, :] = [x.split('  ')[3] for x in f2.readlines()]
    accuracy_test[2, :] = [x.split('  ')[3] for x in f3.readlines()]
    accuracy_test[3, :] = [x.split('  ')[3] for x in f4.readlines()]
    accuracy_test[4, :] = [x.split('  ')[3] for x in f5.readlines()]
    accuracy_test[5, :] = [x.split('  ')[3] for x in f6.readlines()]
    accuracy_test[6, :] = [x.split('  ')[3] for x in f7.readlines()]
    accuracy_test[7, :] = [x.split('  ')[3] for x in f8.readlines()]
    accuracy_test[8, :] = [x.split('  ')[3] for x in f9.readlines()]
    accuracy_test[9, :] = [x.split('  ')[3] for x in f10.readlines()]
    accuracy_test[10, :] = [x.split('  ')[3] for x in f11.readlines()]
    accuracy_test[11, :] = [x.split('  ')[3] for x in f12.readlines()]
    accuracy_test[12, :] = [x.split('  ')[3] for x in f13.readlines()]
    accuracy_test[13, :] = [x.split('  ')[3] for x in f14.readlines()]
    accuracy_test[14, :] = [x.split('  ')[3] for x in f15.readlines()]
    accuracy_test[15, :] = [x.split('  ')[3] for x in f16.readlines()]
    accuracy_test[16, :] = [x.split('  ')[3] for x in f17.readlines()]
    accuracy_test[17, :] = [x.split('  ')[3] for x in f18.readlines()]
    accuracy_test[18, :] = [x.split('  ')[3] for x in f19.readlines()]
    accuracy_test[19, :] = [x.split('  ')[3] for x in f20.readlines()]
    return [accuracy_test[i, :] for i in range(20)]


acc_test = list(test_acc_extraction())
u = np.zeros([30, 20])
for i in range(len(u[0])):
    u[:, i] = np.array(acc_test[i])
'''l1 = loss_av[0]
l2 = loss_av[1]
l3 = loss_av[2]
l4 = loss_av[3]
x = np.arange(1,31,1)
l_mean = np.mean([l1,l2,l3,l4], axis=0)
l_std = np.std([l1,l2,l3,l4], axis=0)
print(l_mean)
plt.plot(l1)
plt.plot(l2)
plt.plot(l3)
plt.plot(l4)
plt.errorbar(x, l_mean, yerr=l_std, linestyle='--')
plt.show()'''
x = np.arange(1,31,1)
l_mean = np.mean(u, axis=1)
l_std = np.std(u, axis=1)
fig = plt.figure(figsize=(12,9))
plt.errorbar(x, l_mean, yerr=l_std, linestyle='--')
plt.fill_between(x, np.subtract(l_mean, l_std), np.add(l_mean, l_std), color='lightblue', alpha=0.4)
plt.xlabel("Epoch", fontsize=26)
plt.ylabel("Avg test accuracy", fontsize=26)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0.6,0.75)
#plt.savefig(path+'test_acc.pdf', dpi=500)
plt.show()
