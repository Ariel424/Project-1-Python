#import scipy
from scipy.io import wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import scipy.signal as scisig
import padasip as pa

#Calculate the standard deviation
def stan_dev(y, f, f_avg):
    indx = np.argsort(xf) #Sorting frequencies
    sum1 = 0
    sum2 = 0
    for i in indx:
        sum1 += ((f[i]-f_avg)**2) * y[i]
        sum2 += y[i]
    standard = np.sqrt(sum1 / sum2)
    return standard
#Plot desired funcion
def plot_it(y, x, indx, title, xlabel, ylabel):
    plt.subplot(2, 1, indx)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplots_adjust(hspace=0.4)  #Adjust the spacing between the plots

#Calculate average and basic harmonic
def avg(y, xf):
    idx = np.argsort(xf)  # Sorting frequencies
    sum1 = 0
    sum2 = 0
    y_abs = abs(y)
    for i in idx:
        if (xf[i] >= 0) and (xf[i] <= 20000):
            sum1 += (y_abs[i] ** 2) * xf[i]
            sum2 += y_abs[i] ** 2
    avg_value = sum1 / sum2
    return avg_value

############# MAIN ###############

#Gather sample rate and data from audio file
fs, signal = wavfile.read("BGoode.wav")
print("Sample rate:",fs, "[Hz]")
channels = len(signal.shape)
print("Channels:", channels)
sig = signal
if channels == 2:
    sig = signal.sum(axis=1) / 2
N = sig.shape[0]
t_audio = N / fs
print("Audio time:", t_audio, "[sec]") #Length of audio in seconds
time = np.linspace(0, t_audio, N) #Time axis

#Fourier Transform
f_sig = rfft(sig)
xf = rfftfreq(N, 1 / fs) #Frequency axis
plt.figure(figsize=(12, 7))
#plot_it(sig, time, 1, "Original Signal", "Time[sec]", "Amplitude")
#plot_it(abs(f_sig), xf, 2, "Original Signal in frequency domain", "Frequency[Hz]", "Amplitude")
#plt.show()
#Average frequency and STD
f_avg = avg(f_sig, xf)
print("Before\nAverage frequency:", f_avg)

#Band Pass Filter
f_pass = [85, 155] #Passband frequency [Hz]
f_stop = [85/2, fs/2] #Stopband frequency [Hz]
g_stop = 20 #Stopband attenuation [dB]
g_pass = 1
order, wn = scisig.buttord(f_pass, f_stop, g_pass, g_stop, analog=False, fs=fs) #Butterworth type
b, a = scisig.butter(3, wn, 'band', analog=False, output='ba', fs=fs)
filt_sig = scisig.filtfilt(b, a, sig)#Filter the signal
w, h = scisig.freqz(b, a) #Compute the frequency response of the digital filter
w = w * fs / (2 * np.pi) #Radians to Hz
filt_sig_f = rfft(filt_sig)

#Plot the butterworth filter and the filtered signal
#plot_it(20*np.log(abs(h)), w, 1, "Butterworth Filter between 85-155[Hz]", "Frequency[Hz]", "Amplitude[dB]")
#plot_it(abs(filt_sig_f), xf, 2, "1st Harmony", "Frequency[Hz]", "Amplitude")
#plt.show()

#Normalize the amplitude to the original one
first_harmony = (filt_sig / max(abs(filt_sig))) * max(abs(sig))

#Calculate the average filter after the filter
filt_avg = avg(abs(filt_sig_f), xf)
print("1st")
print("Average frequency:", filt_avg)
#Save filtered file
#wavfile.write('1stHarmony.wav', fs, first_harmony.astype(np.int16))

#Calculate standard deviation
std = stan_dev(abs(filt_sig_f), xf, filt_avg)
print("Standard Deviation:", std)

######### 4 ##########
def filter(signal, fAvg, std, i, name):
    f_pass = [i * fAvg - (2 * std), i * fAvg + (2 * std)]  #Passband frequency [Hz]
    f_stop = [f_pass[0] / 2, fs / 2]  #Stopband frequency [Hz]
    order, wn = scisig.buttord(f_pass, f_stop, g_pass, g_stop, analog=False, fs=fs)  #Butterworth type
    b, a = scisig.butter(3, wn, 'band', analog=False, output='ba', fs=fs)
    signal = scisig.filtfilt(b, a, signal)  #Filter the signal
    w, h = scisig.freqz(b, a)  #Compute the frequency response of the digital filter
    w = w * fs / (2 * np.pi)  #Radians to Hz
    signal_f = rfft(signal)
    #Plot the butterworth filter and the filtered signal
    #plot_it(20 * np.log(abs(h)), w, 1, f"Butterworth Filter around {int(i*fAvg)}[Hz]", "Frequency[Hz]", "Amplitude[dB]")
    #plot_it(abs(signal_f), xf, 2, f"{name} Harmony", "Frequency[Hz]", "Amplitude")
    #plt.show()
    # Normalize the amplitude to the original one
    signal = (signal / max(abs(signal))) * max(abs(sig))
    # Calculate the average filter after the filter
    f_avg = avg(abs(signal_f), xf)
    print(f"{name}:")
    print("Average frequency:", f_avg)
    return signal, f_avg

second_harmony, filt_avg2 = filter(sig, filt_avg, std, 2, "2nd")
#Save filtered file
#wavfile.write('2ndHarmony.wav', fs, second_harmony.astype(np.int16))

########## 5 ##########

third_harmony, filt_avg3 = filter(sig, filt_avg, std, 3, "3rd")
#wavfile.write('3rdHarmony.wav', fs, third_harmony.astype(np.int16))

########### 6 ##########
#sig_mult_f = rfft(sig_mult)
    #cosine_f = rfft(cosine)
    #Plot the signal and the cosine wave in the frequency domain
    #plot_it(cosine_f, f, 1, f"Cosine Wave around {int(fAvg)}[Hz]", "Frequency[Hz]", "Amplitude")
    #plt.xlim(0, 20000)
    #plot_it(sig_mult_f, f, 2, f"{name} Harmony After Multiply", "Frequency[Hz]", "Amplitude")
    #plt.show()
def mult_by_cos(signal, f, fAvg, time, std, name):
    cosine = np.cos(2* np.pi * fAvg * time) #Define Cosine wave in the average frequency
    sig_mult = cosine * signal #Multiply by the signal
    #Filter the new signal
    g_pass = 1
    g_stop = 20
    f_pass = [fAvg * 2 - (std*2), fAvg * 2 + (std*2)]  #Passband frequency [Hz]
    f_stop = [f_pass[0] / 2, fs / 2]  #Stopband frequency [Hz]
    order, wn = scisig.buttord(f_pass, f_stop, g_pass, g_stop, analog=False, fs=fs)  # Butterworth type
    b, a = scisig.butter(3, wn, 'band', analog=False, output='ba', fs=fs)
    sig_mult = scisig.filtfilt(b, a, sig_mult)  #Filter the signal
    w, h = scisig.freqz(b, a)  #Compute the frequency response of the digital filter
    w = w * fs / (2 * np.pi)  #Radians to Hz
    mult_sig_f = rfft(sig_mult)
    # Plot the butterworth filter and the filtered signal
    #plot_it(20 * np.log(abs(h)), w, 1, f"Butterworth Filter around {int(fAvg*2)}[Hz]", "Frequency[Hz]", "Amplitude[dB]")
    #plot_it(abs(mult_sig_f), xf, 2, f"{name} Harmony after filter", "Frequency[Hz]", "Amplitude")
    #plt.show()
    # Normalize the amplitude to the original one
    sig_mult = (sig_mult / max(abs(sig_mult))) * max(abs(sig))
    # Save filtered file
    #wavfile.write(f'{name}MultSig.wav', fs, sig_mult.astype(np.int16))
    return sig_mult

mult_1st = mult_by_cos(sig, xf, filt_avg, time, std, '1st')
mult_2nd = mult_by_cos(sig, xf, filt_avg2, time, std, '2nd')
mult_3rd = mult_by_cos(sig, xf, filt_avg3, time, std, '3rd')

########### 7 ###########
#Sum all the Multiplied Signals
sum_of_mult = mult_1st + mult_2nd + mult_3rd
#plot_it(sum_of_mult, time, 1, "Sum of Multiplications", "Time[sec]", "Amplitude")
#sum_of_mult_f = rfft(sum_of_mult)
#plot_it(sum_of_mult_f, xf, 2, "Sum of Multiplications in frequency domain", "Frequency[Hz]", "Amplitude")
#plt.show()
#wavfile.write('Sum_Of_Mult.wav', fs, sum_of_mult.astype(np.int16))

############# 8 ###########
#Recursive least squares filter (RLS)


#Gather sample rate and data from audio file
fs, recycled = wavfile.read("recycled_signal.wav")
channels = len(signal.shape)
rec = recycled
N = rec.shape[0]
t_audio = N / fs
print("Audio time:", t_audio, "[sec]") #Length of audio in seconds
time1 = np.linspace(0, t_audio, N) #Time axis
xf1 = rfftfreq(N, 1 / fs) #Frequency axis

recycled = (recycled / max(abs(recycled))) * max(abs(sig)) * 4

recycled_f = rfft(recycled)
plt.plot(xf1, abs(recycled_f))
plt.show()
wavfile.write("AfterRLS.wav", fs, filt_sig.astype(np.int16))






