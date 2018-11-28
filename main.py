import numpy as np
import scipy
import pyaudio
from scipy import signal
import scipy.io.wavfile
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def plot_signal():
    sample_rate, sample_array  = scipy.io.wavfile.read('modemDialing.wav')
    x_array = np.arange(len(sample_array))
    x_array_normalized = np.true_divide(x_array,sample_rate)

    plt.plot(x_array_normalized,sample_array)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [pcm]')
    plt.show()

    return sample_rate, sample_array, x_array_normalized


def spectrogram(x_axis,fs=1.0,):
    f,t,Sxx = scipy.signal.spectrogram(x_axis,fs,window=('tukey',0.25),nperseg=256)
    plt.figure()
    plt.pcolormesh(t,f,Sxx)
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.show()

def fundamental_dtmf_freq():
    dtmf_freqs_array = [697,770,852,941,1209,1336,1477,1633]
    print(np.gcd.reduce(dtmf_freqs_array))


def generate_tone(f1,f2,fs,duration=1):
    silence = np.zeros(int(0.05*fs))
    t_array = np.arange(0,duration,1/fs)
    omega_1 = 2*np.pi*f1
    omega_2 = 2*np.pi*f2
    first_sine = np.sin(omega_1*t_array)
    second_sine = np.sin(omega_2*t_array)
    sum_of_sines = first_sine + second_sine
    sum_of_sines = np.divide(sum_of_sines,2)
    chunk = sum_of_sines * 0.25 #lower volume
    chunk = np.append(chunk,silence)
    return chunk

def play_tones(chunk,fs):
    while len(chunk) < fs:
        chunk = np.append(chunk, [0])  # pad with zeros
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    stream.write(chunk.astype(np.float32).tostring())
    stream.stop_stream()
    stream.close()

    p.terminate()

def generate_tones(digit_seq,fs,dur):
    dtmf_freqs = {'1': (1209, 697), '2': (1336, 697), '3': (1477, 697), 'A': (1633, 697),
                  '4': (1209, 770), '5': (1336, 770), '6': (1477, 770), 'B': (1633, 770),
                  '7': (1209, 852), '8': (1336, 852), '9': (1477, 852), 'C': (1633, 852),
                  '*': (1209, 941), '0': (1336, 941), '#': (1477, 941), 'D': (1633, 941)}
    resulting_chunk = np.empty((1,0))
    for digit in digit_seq:
        freqs = dtmf_freqs[digit]
        resulting_chunk =np.append(resulting_chunk,generate_tone(freqs[1],freqs[0],fs,dur))
    play_tones(resulting_chunk, fs)


def calculate_fft(sample_array,fs):
    N = len(sample_array)
    T = 1/fs
    yf = fft(sample_array)
    xf = np.linspace(0.0,1.0/(2.0*T),N//2)
    plt.figure()
    plt.plot(xf,2.0/N * np.abs(yf[0:N//2]))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Potencia [W/Hz]')
    plt.grid()
    plt.show()

def plot_digits_dft():
    digit_idxes = {1: (3.35, 3.45), 2: (3.48, 3.63), 3: (3.65, 3.78), 4: (3.8, 3.94), 5: (3.95, 4.08)}
    for key, val in digit_idxes.items():
        idx_1 = int(val[0] * fs)
        idx_2 = int(val[1] * fs)
        calculate_fft(sample_array[idx_1:idx_2], fs)


if __name__ == '__main__':
    fs,sample_array,normalized_x =plot_signal()
    spectrogram(sample_array,fs)
    #plot_digits_dft()
    digit_seq = '32327'
    generate_tones(digit_seq,fs,0.07)