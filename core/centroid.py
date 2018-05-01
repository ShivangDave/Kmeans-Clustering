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
#import math
import numpy as np

def calcDistance(p,c):
    return np.linalg.norm(p-c)

def init_cent(rows,k,dataset):
    return dataset[np.random.randint(0,rows-1,size=k)]

def create_gen_mat(a,b):
    return np.zeros((a,b))

############################# Non-numpy functions ###########################
# def sampleCentroid(fMatrix,k):
#     cent = []
#     one = ra.randint(1,len(fMatrix))
#     two = ra.randint(1,len(fMatrix))
#     three = ra.randint(1,7)
#     four = ra.randint(1,7)
#     cent.append([fMatrix[one][three],fMatrix[two][four]])
#     cent.append([fMatrix[two][three],fMatrix[one][four]])
#     return cent
#
# def calcDistance(p,c):
#     summed = 0
#     for i in range(0,len(p)):
#         for j in range(0,len(c)):
#             t1 = 0.0
#             t2 = 0.0
#             try:
#                 t1 = p[i][j]
#             except IndexError:
#                 t1 = 0.0
#             try:
#                 t2 = c[i][j]
#             except IndexError:
#                 t2 = 0.0
#             summed = summed + ((t1 - t2)**2)
#     return math.sqrt(summed)
