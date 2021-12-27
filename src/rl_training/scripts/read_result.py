import pickle
import matplotlib.pyplot as plt
import sys

train_loss_file = "training_result/TRAIN_LOSS.pkl"
avf_return_file = "training_result/AVG_RETURN.pkl"
step_file = "training_result/STEP.pkl"

open_file = open(train_loss_file, "rb")
TRAIN_LOSS = pickle.load(open_file)
open_file.close()

open_file = open(avf_return_file, "rb")
AVG_RETURN = pickle.load(open_file)
open_file.close()

TRAIN_LOSS_STEP = []
AVG_RETURN_STEP = []

for i in range(len(TRAIN_LOSS)):
    TRAIN_LOSS_STEP.append((i+1)*200)

for i in range(len(AVG_RETURN)):
    AVG_RETURN_STEP.append((i+1)*1000)

if (sys.argv[1]=="l"):
    ax = plt.axes()
    ax.plot(TRAIN_LOSS_STEP, TRAIN_LOSS)
    ax.set(xlim=(0, 1000000), xlabel='step', ylabel='loss', title='TRAIN_LOSS')
    for i in range(len(TRAIN_LOSS_STEP)):
    	print("STEP ", TRAIN_LOSS_STEP[i], ", TRAIN_LOSS ",  TRAIN_LOSS[i])
    plt.show()

elif (sys.argv[1]=="r"):
    bx = plt.axes()
    bx.plot(AVG_RETURN_STEP, AVG_RETURN)
    bx.set(xlim=(0, 1000000), xlabel='step', ylabel='average return', title='AVG_RETURN')
    for i in range(len(AVG_RETURN_STEP)):
    	print("STEP ", AVG_RETURN_STEP[i], ", AVG_RETURN ",  AVG_RETURN[i])
    plt.show()

else:
    print("arg must be r or l")
    
