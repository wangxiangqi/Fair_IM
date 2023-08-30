import numpy as np
import matplotlib.pyplot as plt
import glob

# Get a list of all .npy files in the current directory
import pickle
files = glob.glob('./ploting_npy_wel/*.pkl')

# Create a new figure
plt.figure()

# Load data from each file and plot
for file in files:
    data = pickle.load(open(file, 'rb'), encoding='latin1')
    data=list(data)
    data_1=[data[0]]
    for i in range(len(data)-1):
        data_1.append(data[i+1]-data[i])
    #if np.array(data).shape[0]>1:
    #    data=np.array(data)[:,0]
    plt.plot(data_1, label=file)

# Add a legend and show the plot
plt.legend()
plt.show()