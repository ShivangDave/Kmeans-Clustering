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
import sys
from core import plotdata as plot
from core import matrix as mat
from core import centroid as cent
from core import kmeans as km

def main(filename,k):
    extension = filename.split(".")[1]
    train_data, test_data = mat.load(filename,extension)

    print('\nTraining in progress....')
    prev_cent, cl_b = km.train_kmeans(k,train_data)
    centroid = prev_cent[len(prev_cent)-1]
    error_in = mat.calcError(train_data,k,centroid,cl_b)

    print('\nEin for K-means clustering: %fe5'%(error_in/100000.0))
    print('Training Complete....')

    print('\nPrediction in progress....')
    predicted_labels = mat.predict(test_data,k,centroid)
    km.clustered(k,centroid,predicted_labels,1)
    print('\nPrediction Complete....')

    error_out = mat.calcError(test_data,k,centroid,predicted_labels)
    #print('\nEout for K-means clustering: %fe5'%(error_out/100000.0))

    if k<7:
        print('\nPlotting centroids....\n')
        #plot.plotpoints(train_data)
        plot.plotdata(train_data,train_data,prev_cent,cl_b,predicted_labels)

if len(sys.argv) != 2:
    main(str(raw_input('Enter filename with its extension: ')),int(input('Enter # of clusters: ')))
else:
    main('small_data.txt',2)
