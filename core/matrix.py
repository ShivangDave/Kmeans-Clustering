# Copyright (C) 2018 Shivang Dave <mail@shivangdave.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, see <http://www.gnu.org/licenses/>.
import numpy as np
from core import centroid as ce
import sklearn
from sklearn import model_selection

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split

def splitting(dataset):
    t_size = float(input('Input measure to split the data in train and test sets (recommended: 0.8): '))
    train_data, test_data = sklearn.model_selection.train_test_split(dataset, train_size=(t_size if t_size>0.0 else 0.8))
    return train_data, test_data

def load(filename,extension):
    if extension == "csv":
        dataset = np.genfromtxt(filename,delimiter=",")
        dataset = dataset[1:dataset.shape[0],1:dataset.shape[1]-1]
        train_data, test_data = splitting(dataset)
        return train_data, test_data
    elif extension == "txt":
        dataset = np.loadtxt(filename)
        train_data, test_data = splitting(dataset)
        return train_data, test_data
    else:
        print('It only supports csv and txt files at the moment.')
        sys.exit()

def nearest_cent(dataset,k,init_cent,cl_b,features):
    for x,y in enumerate(dataset):
        vector = ce.create_gen_mat(k,1)
        for p,q in enumerate(init_cent):
            vector[p] = ce.calcDistance(q,y)
        cl_b[x] = np.argmin(vector)

def predict(dataset,k,init_cent):
    predicted_labels = ce.create_gen_mat(dataset.shape[0],1)
    for x,y in enumerate(dataset):
        vector = ce.create_gen_mat(k,1)
        for p,q in enumerate(init_cent):
            vector[p] = ce.calcDistance(q,y)
        predicted_labels[x] = np.argmin(vector)
    return predicted_labels

def find_new_cent_mean(init_cent,cl_b,dataset,dist_mat_cent):
    for item in range(len(init_cent)):
        close = []
        for i in range(len(cl_b)):
            if cl_b[i] == item:
                close.append(i)
        y = np.mean(dataset[close],axis=0)
        dist_mat_cent[item,:] = y
    return dist_mat_cent

def calcError(dataset,k,centroids,cl_b,error = 0):
    for i in range(k):
        for j in range(len(cl_b)):
            if cl_b[j] == i:
                error = error + ce.calcDistance(dataset[j],centroids[i])
    return error

############################# Non-numpy functions ###########################
# def typemat(filename,extension):
#     matrix = []
#     if extension == "csv" or "data":
#         f = open(filename,'rb')
#         for line in f:
#             l = [i.strip() for i in line.split(',')]
#             matrix.append(l)
#         return [matrix,0]
#     elif extension == "txt":
#         f = open(filename,'r')
#         l = [map(int,line.split(',')) for line in f]
#         return [l,1]
#     else:
#         print('It only supports csv and txt files at the moment.')
#         sys.exit()
#
# def createMatrix(l,type,k):
#     fMatrix = []
#     real_class = []
#     if type != 0:
#         for i in range(1,len(l)):
#             fMatrix.append([l[i][0],l[i][1]])
#     else:
#         for i in range(1,len(l)):
#             fMatrix.append([float(l[i][1]),float(l[i][2]),float(l[i][3]),float(l[i][4]),float(l[i][5]),float(l[i][6]),float(l[i][7]),float(l[i][8]),float(l[i][9]),float(l[i][10]),float(l[i][11]),float(l[i][12]),float(l[i][13]),float(l[i][14])])
#             real_class.append(int(l[i][15]))
#     return [fMatrix,real_class]
#
# def argmin(array):
#     temp = []
#     for i in range(0,len(array)):
#         temp.append(array[i])
#     return min(temp)
#
# def mean(array):
#     mean = 0
#     for i in zip(*array):
#         mean = mean + float(sum(i))/len(i)
#     return [mean]
