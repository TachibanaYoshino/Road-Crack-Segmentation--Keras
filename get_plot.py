# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def get_loss_list(model):
    dir = os.path.join('checkpoint', model)
    files = os.listdir(dir)
    sorted(files)

    index =[]
    losses = []
    acc =[]
    for i,x in enumerate(files):
        index.append(i)
        losses.append(float(x.split("-")[2].strip()))
        acc.append(float(x.split("-")[3][:-3].strip()))

    return index, losses,acc



def plot(model1,model2,if_loss=1):

    x1,y1,z1 = get_loss_list(model1)
    x2,y2,z2 = get_loss_list(model2)

    assert  x1 ==x2
    #开始画图

    if if_loss:
        pic_name = 'loss'
        plt.plot(x1, y1, color='green', label=model1,linewidth=1.0)
        plt.plot(x2, y2, color='red', label=model2,linewidth=1.0)

    else:
        pic_name = 'acc'
        plt.plot(x1, z1, color='blueviolet', label=model1,linewidth=1.0)
        plt.plot(x2, z2, color='chocolate', label=model2,linewidth=1.0)

    plt.grid(linestyle='-.')
    plt.legend() # 显示图例

    plt.xlabel('epoch')
    plt.ylabel('val_'+pic_name)
    # plt.show()
    plt.savefig(pic_name+'.png')


def plot1(model1, model2, name):
    x1, y1, z1 = get_loss_list(model1)
    x2, y2, z2 = get_loss_list(model2)

    assert x1 == x2
    # 开始画图
    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(1, 2, 1)

    pic_name = 'loss'
    plt.plot(x1, y1, color='green', label=model1, linewidth=1.0)
    plt.plot(x2, y2, color='red', label=model2, linewidth=1.0)
    plt.grid(linestyle='-.')
    plt.legend()  # 显示图例
    plt.xlabel('epoch')
    plt.ylabel('val_' + pic_name)

    fig.add_subplot(1, 2, 2)
    pic_name = 'acc'
    plt.plot(x1, z1, color='blueviolet', label=model1, linewidth=1.0)
    plt.plot(x2, z2, color='chocolate', label=model2, linewidth=1.0)
    plt.grid(linestyle='-.')
    plt.legend()  # 显示图例
    plt.xlabel('epoch')
    plt.ylabel('val_' + pic_name)
    # plt.show()
    plt.savefig(name + '.png')

if __name__ == '__main__':
    model1 = 'Unet'
    model2 = 'GCUnet'

    # plot(model1,model2,if_loss=0)

    plot1(model1,model2,'result')

