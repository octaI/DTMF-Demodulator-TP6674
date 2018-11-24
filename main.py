import numpy as np
import scipy
import pyaudio
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt


def plot_signal():
    sample_rate, sample_array  = scipy.io.wavfile.read('modemDialing.wav')
    x_array = np.arange(len(sample_array))
    x_array_normalized = np.true_divide(x_array,sample_rate)

    plt.plot(x_array_normalized,sample_array)
    plt.savefig("graph.svg",dpi=1000, format='svg')
    plt.show()

    return sample_rate, sample_array, x_array_normalized


def spectrogram(x_axis,fs=1.0,):
    f,t,Sxx = scipy.signal.spectrogram(x_axis,fs)
    plt.figure()
    plt.pcolormesh(t,f,Sxx)
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.show()

def fundamental_dtmf_freq():
    dtmf_freqs_array = [697,770,852,941,1209,1336,1477,1633]
    print(np.gcd.reduce(dtmf_freqs_array))


def generate_tone(f1,f2,fs):
    t_array = np.arange(0,1,1/fs)
    print(len(t_array))
    omega_1 = 2*np.pi*f1
    omega_2 = 2*np.pi*f2
    first_sine = np.sin(omega_1*t_array)
    second_sine = np.sin(omega_2*t_array)
    sum_of_sines = first_sine + second_sine
    chunk = sum_of_sines * 0.25 #sino sale muy fuerte
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    stream.write(chunk.astype(np.float32).tostring())
    stream.stop_stream()
    stream.close()

    p.terminate()

    plt.figure()
    plt.plot(t_array,sum_of_sines)
    plt.show()


if __name__ == '__main__':
    fs,sample_array,normalized_x =plot_signal()
    spectrogram(sample_array,fs)
    generate_tone(697,1477,8000)
    generate_tone(697,1336,8000)