#------------------------------------------------------------------------------------------------------------------
#   Mobile sensor data acquisition and processing
#------------------------------------------------------------------------------------------------------------------
import pickle
import numpy as np
from scipy import stats

# Load data
file_name = 'actividad_ximena.obj'
inputFile = open(file_name, 'rb')
experiment_data = pickle.load(inputFile)

# Process each trial and build data matrices
features = []
for tr in experiment_data:
    
    # For each signal (one signal per axis)
    feat = [tr[1]]
    rms = 0
    for s in range(tr[2].shape[1]):
        sig = tr[2][:,s]

        feat.append(np.average(sig))
        feat.append(np.std(sig))
        feat.append(stats.kurtosis(sig))
        feat.append(stats.skew(sig))
        feat.append(np.min(sig))            # Valor mínimo
        feat.append(np.max(sig))            # Valor máximo
        feat.append(np.median(sig))         # Mediana
        feat.append(np.percentile(sig, 25)) # Percentil 25
        feat.append(np.percentile(sig, 75)) # Percentil 75
        feat.append(np.ptp(sig))            # Peak-to-peak (máx - mín)
        vel = np.gradient(sig)
        feat.append(np.mean(vel))
        feat.append(np.std(vel))
        rms += np.sum(sig**2)
        
    rms = np.sqrt(rms)    
    feat.append(rms)
    
    features.append(feat)      

# Build x and y arrays
processed_data =  np.array(features)
x = processed_data[:,1:]
y = processed_data[:,0]

# Save processed data
np.savetxt("actividad_ximena.txt", processed_data)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
