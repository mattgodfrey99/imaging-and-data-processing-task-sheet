import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import ifft
from scipy.signal import butter
from scipy.signal import freqz
from scipy.signal import hilbert

## Initial Signal ###########################################

signal = np.loadtxt('signal.mat')
time_tot = 40 # total time span
s_r = 1200;  # sampling rate
dt = 1.0/s_r; # sampling interval
t = np.arange(0,time_tot,dt) # time array

plt.figure(0)
plt.plot(t, signal)
plt.title('Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

## Fourier Signal ############################################################

n = len(signal) # signal length
k = np.arange(n) # 0 to n
freq = k/time_tot # frequency range
freq = freq[:len(freq)//2] # only want one half as repeated

signal_Y = fft(signal)/n # power of fourier transform with normalization
signal_Y = signal_Y[:int(n/2)] # only want one half as repeated

plt.figure(1)
plt.plot(freq,abs(signal_Y)) # plotting the spectrum
plt.title('Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

## Filter Tests ##############################################################

def butterworth(low_cut, high_cut, s_r, order=5):
    
    nyq = s_r/2
    limits = np.array((low_cut,high_cut))/nyq
    b, a = butter(order, limits, btype='band')
    return b, a

low_cut = 30
high_cut = 35
order = 3

b, a = butterworth(low_cut, high_cut, s_r, order=order)
w, h = freqz(b, a, len(freq))
plt.figure(2)
plt.plot(freq, abs(h))
# could replace (s_r * 0.5 / np.pi) * w by frq

filtered = abs(h)*signal_Y
plt.plot(freq,np.real(filtered)) # filtered fft of signal
plt.title('Frequency Space Filtered')

new_signal = ifft(filtered)
plt.figure(3)
plt.plot(np.real(new_signal)) # plots new filtered signal

analytic = abs(hilbert(np.real(new_signal)))
plt.plot(analytic) # plots envelope
plt.title('Time Space Filtered w/ Envelope')

## Frequency-Time Distribution ###############################################

def butterworth(low_cut, high_cut, s_r, order=5):
    
    nyq = s_r/2
    limits = np.array((low_cut,high_cut))/nyq
    b, a = butter(order, limits, btype='band')
    return b, a

order = 3
step_size = 0.5
end_freq = 60 # change to max value if needed
end_freq_input = int(end_freq*(1/step_size))
filter_size = 2
freq_time_dist = np.ones((end_freq_input-filter_size-1,len(signal_Y)))

for i in np.arange(1,end_freq_input-filter_size):
    
    low_cut = i*step_size
    high_cut = low_cut+filter_size
    
    b, a = butterworth(low_cut, high_cut, s_r, order)
    w, h = freqz(b, a, len(freq))
    
    filtered = abs(h)*signal_Y
    new_signal = ifft(filtered)
    analytic = abs(hilbert(np.real(new_signal)))
    
    freq_time_dist[i-1,:] = analytic
    
    
f, ax = plt.subplots()
plt.imshow(freq_time_dist, origin='lower', extent=(0, time_tot, 0,end_freq-filter_size-1 ))
ax.axis('tight')
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)');
plt.title('Time-Frequency Distribution')
cbar = plt.colorbar()
cbar.set_label('Relative Amplitude', rotation=270)

