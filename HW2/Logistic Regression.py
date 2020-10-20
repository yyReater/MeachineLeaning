# -*- coding: utf-8 -*-
import numpy as np
import csv
import os
import random

def select_min_max(dataset):
    min_max = []
    for i in range(len(dataset[0])):
        set_tmp = [example[i] for example in dataset] #取出dataset的第i列
        now_min = min(set_tmp)
        now_max = max(set_tmp)
        min_max.append([now_min, now_max])
    return min_max

def std_init(dataset):
    min_max = np.array(select_min_max(dataset))
    dataset = (dataset - min_max[:, 0]) / (min_max[:, 1] - min_max[:, 0])
    return dataset.tolist()

def separate_data(dataset, k):
    clone_dataset = list(dataset)
    total = len(clone_dataset) / k
    s_data = []
    for i in range(k):
        now = []
        while ((len(now) < total) and (len(clone_dataset) > 0)):
            index = random.randint(0, len(clone_dataset)-1)
            now.append(clone_dataset[index])
            clone_dataset.pop(index)
        s_data.append(now)
    return s_data

def predict(data, theta, theta0):#计算预测函数
    data = np.array(data)
    theta = np.array(theta)
    data = data[:, 0:-1]
    y = np.dot(data, theta.T) + theta0
    try: 
        ans = np.exp(-1 * y)
    except OverflowError:
        ans = float('inf')
    return 1.0 / (1.0 + ans)

def gradient_descent(dataset, alpha = 0.005, iter = 1000): #梯度下降 得到theta
    theta = np.random.random((1, len(dataset[0]) - 1))
    theta0 = 0
    for i in range(iter):
        pred = predict(dataset, theta, theta0)
        #print(type(pred), pred.shape)
        pred = np.round(pred)
        pred = list(pred)
        for j in range(len(pred)):
            theta0 -= alpha * (pred[j] - dataset[j][-1])
            theta -= alpha * (pred[j] - dataset[j][-1]) * np.array(dataset[j][:-1])
    return theta, theta0

def cal_accuracy(actual, pred):
    correct = 0
    #print(len(actual), len(pred))
    actual = np.array(actual)
    pred = np.array(pred)
    #print(actual)
    #print(pred)
    #print(actual.shape, pred.shape)
    correct = (actual == pred).sum()
    #print(correct, len(actual))
    return correct / float(len(actual)) * 100 

def get_result(s_data):
    datas = list(s_data)
    #print(len(datas), type(datas), len(datas[0]), type(datas[0]), len(datas[0][0]), type(datas[0][0]))
    accs = []
    for data in datas:
        #print(type(data),type(datas))
        #print('asd')

        train_set = list(datas)
        #print(len(s_data), len(s_data[0]), len(s_data[0][0]),type(s_data[0]))
        #print(len(data),len(data[0]),type(data))
        train_set.remove(data)

        train_set = sum(train_set, []) #!!!!python神奇应用之 list合并 令人直呼卧槽！
        test_set = list(data)

        theta, theta0 = gradient_descent(train_set)
        pred = predict(test_set, theta, theta0)
        pred = np.round(pred)
        pred = pred.tolist()
        
        #print(type(pred), type(pred[0]))

        actual = np.array(data)
        #print(actual.shape)
        actual = (actual[:, -1].reshape(actual.shape[0], 1)).tolist()
        #print(len(actual), 'asd', actual[0])

        #print(len(actual), type(actual))
        #print(len(pred), type(pred))

        acc = cal_accuracy(actual, pred)
        accs.append(acc)

    return accs


#rint(os.getcwd())
filename = './HW2/pima-indians-diabetes.csv'
dataset = []
with open(filename, 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        dataset.append(row)
#print(dataset)

for i in range(len(dataset)):
    for j in range(len(dataset[0])):
        dataset[i][j] = float(dataset[i][j])
#print(dataset[:1]) 输出前1行

dataset = std_init(dataset)  #数据归一化
#print(type(dataset), type(dataset[0]))
#print(dataset[:5 ])

s_data = separate_data(dataset, 4) #分为k组 用来做k-折叠交叉验证
'''for i in range(4):
    print(len(s_data[i]))
print(len(dataset))'''

'''dataset =[[2.7810836,2.550537003,0],[1.465489372,2.362125076,0],[3.396561688,4.400293529,0],[1.38807019,1.850220317,0],[3.06407232,3.005305973,0],[7.627531214,2.759262235,1]]
wht = [0.852573316, -1.104746259]
b = -0.406605464
yhat = predict(dataset, wht, b)
dataset = np.array(dataset)
print("predicted = ",np.around(yhat))'''

accs = get_result(s_data)

print('Accuracies: %s' % accs)
print('Mean Accuracy: %.3f%%' % (sum(accs) / float(len(accs))))