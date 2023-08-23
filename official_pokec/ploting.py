import numpy as np
import matplotlib.pyplot as plt
import glob

# Get a list of all .npy files in the current directory
import pickle
files = glob.glob('./ploting_npy_div/*.pkl')

# Create a new figure
plt.figure()

# Load data from each file and plot
for file in files:
    data = pickle.load(open(file, 'rb'), encoding='latin1')
    data=list(data)
    #if np.array(data).shape[0]>1:
    #    data=np.array(data)[:,0]
    plt.plot(data, label=file)

# Add a legend and show the plot
plt.legend()
plt.show()