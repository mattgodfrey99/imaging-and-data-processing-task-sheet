import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import ifft
from scipy.signal import butter
from scipy.signal import freqz
from scipy.signal import hilbert

## Initial Signals ###########################################################

signal_brain = np.loadtxt('signal_brain.mat')
signal_brain2 = np.loadtxt('signal_brain2.mat')

signal = signal_brain2.copy() # change between signal_brain and signal_brain2

time_tot = 617.5 # total time span
s_r = 600;  # sampling rate
dt = 1.0/s_r; # sampling interval
t = np.arange(0,time_tot,dt) # time array
task_time = time_tot/95 # time for task
task_t = np.arange(0,task_time,dt) # time array for task

n = len(signal_brain)/95 # task length
k = np.arange(n) # 0 to n
freq = k/task_time # frequency range
freq = freq[:len(freq)//2] # only want one half as repeated

## Functions #################################################################

def butterworth(low_cut, high_cut, s_r, order=5):
    
    nyq = s_r/2
    limits = np.array((low_cut,high_cut))/nyq
    b, a = butter(order, limits, btype='band')
    return b, a

def freq_time_dist_func(task_start,signal_brain,s_r,task_time,n,order,end_freq,filter_size,step_size):
    
    signal_task = signal_brain[int(task_start*s_r):int((task_start+task_time)*s_r)]
    signal_Y = fft(signal_task)/n
    signal_Y = signal_Y[:int(n/2)]
    freq_time_dist = np.zeros((int((end_freq*(1/step_size))-filter_size-1),len(signal_Y)))
    
    for i in np.arange(1,(end_freq*(1/step_size))-filter_size):
        
       low_cut = i*step_size
       high_cut = low_cut+filter_size
        
       b, a = butterworth(low_cut, high_cut, s_r, order)
       w, h = freqz(b, a, len(signal_Y))
        
       filtered = abs(h)*signal_Y
       new_signal = ifft(filtered)
       analytic = abs(hilbert(np.real(new_signal)))
        
       freq_time_dist[int(i-1),:] = analytic
        
    return freq_time_dist

## Average TFD ###############################################################
    
order = 3
end_freq = 80 # can change to max to see full distribution
filter_size = 3
step_size = 1
full_image = np.zeros((int((end_freq*(1/step_size))-filter_size-1),len(freq)))

for i in range(95):
    
    full_image += freq_time_dist_func(i*6.5,signal,s_r,task_time,n,order,end_freq,filter_size,step_size)
    
full_image /= 95 # normalise  

## TFD Plots #################################################################

f, ax = plt.subplots()
plt.imshow(full_image, origin='lower', extent=(0, task_time, 0,end_freq-filter_size-1 ))
ax.axis('tight')
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)');
ax.set_title('Time-Frequency Distribution (Relative)')
cbar = plt.colorbar()
cbar.set_label('Relative Intensity', rotation=270)

n_base = s_r # number of points over 1 second

base_image = freq_time_dist_func((time_tot-1),signal,s_r,1,n_base,order,end_freq,filter_size,step_size)
base_mean = np.mean(base_image,1)
base_mean = np.array([base_mean,]*len(freq)).transpose()

full_image = ((full_image/base_mean)*100)-100

    
f, ax = plt.subplots()
plt.imshow(full_image, origin='lower', extent=(0, task_time, 0,end_freq-filter_size-1 ))
ax.axis('tight')
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_title('Time-Frequency Distribution (Percentage Change)')
cbar = plt.colorbar()
cbar.set_label('Percentage Change from Baseline', rotation=270)
