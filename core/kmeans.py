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
from core import matrix as mat

def clustered(k,centroids,cl_b,key):
    output_matrix = np.zeros(shape=(k,1))
    total = []

    for i in range(k):
        summ = 0
        for j in range(len(cl_b)):
            if cl_b[j] == i:
                summ += 1
        output_matrix[i] = summ

    if key == 0:
        print "Training Results: "
        print "# of training set: %d" %(len(cl_b))
        print "Cluster Labels | Total data points"
    else:
        print "Predictions: "
        print "# of prediction set: %d" %(len(cl_b))
        print "Cluster Labels | Total data points"

    for x,y in enumerate(output_matrix):
        print "             %d | %f                              " %(x,y)
    return output_matrix

def train_kmeans(k,dataset):
    prev_cent, cl_b = [],[]

    iteration, rows, features = 0,dataset.shape[0],dataset.shape[1]
    init_cent = ce.init_cent(rows,k,dataset)
    prev_cent.append(init_cent)

    prev_cent_updated = np.zeros(init_cent.shape)
    cl_b = ce.create_gen_mat(rows,1)
    dist = ce.calcDistance(init_cent,prev_cent_updated)

    while dist>0 or iteration>1000:
        iteration += 1
        dist = ce.calcDistance(init_cent,prev_cent_updated) #Error

        prev_cent_updated = init_cent
        mat.nearest_cent(dataset,k,init_cent,cl_b,features)

        dist_mat_cent = ce.create_gen_mat(k,features)
        mean_cent = mat.find_new_cent_mean(init_cent,cl_b,dataset,dist_mat_cent)
        init_cent = mean_cent
        prev_cent.append(init_cent)

    clustered(k,prev_cent,cl_b,0)

    return prev_cent, cl_b
