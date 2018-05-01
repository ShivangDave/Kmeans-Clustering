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

import matplotlib.pyplot as plt

def plot_train(colors,axis,train_data,cl_b):
    for index in range(train_data.shape[0]):
        close = []
        for i in range(len(cl_b)):
            if cl_b[i] == index:
                close.append(i)
        for item in close:
            axis.plot(train_data[item][0],train_data[item][1],(colors[index] + 'o'))

def plot_pred(colors,axis,test_data,predicted_labels):
    for index in range(test_data.shape[0]):
        close = []
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == index:
                close.append(i)
        for item in close:
            axis.plot(test_data[item][0],test_data[item][1],(colors[index] + 's'))

def plot_cent(colors,axis,prev_cent):
    arr = []
    for index, centroids in enumerate(prev_cent):
        for inner, item in enumerate(centroids):
            if index == 0:
                arr.append(axis.plot(item[0], item[1], colors[inner]+'v', markersize=15)[0])
            else:
                arr[inner].set_data(item[0], item[1])
                #print("iteration {} {}".format(index, item))
                plt.pause(0.4)

def plotdata(train_data,test_data,prev_cent,cl_b,predicted_labels):
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'w']
    axis = plt.subplots()[1]

    axis.plot(train_data,'gx')
    axis.plot(test_data,'ko')
    plt.pause(3)
    axis.cla()

    plot_train(colors,axis,train_data,cl_b)
    plot_pred(colors,axis,test_data,predicted_labels)
    plot_cent(colors,axis,prev_cent)

    plt.pause(10)
