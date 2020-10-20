# -*- coding: utf-8 -*-

def readin():
    f = open('./boston_house_price_dataset.txt', 'r')
    dataset = []
    while 1:
        now = f.readline()
        if not now:
            break
        aline = now.split()
        dataset.append([float(i) for i in aline])    
    f.close()
    return dataset