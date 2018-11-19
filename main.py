import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt


def plot_signal():
    sample_rate, sample_array  = scipy.io.wavfile.read('modemDialing.wav')
    x_array = np.arange(len(sample_array))
    x_array_normalized = np.true_divide(x_array,sample_rate)
    plt.plot(x_array_normalized,sample_array)
    plt.savefig("graph.svg",dpi=1000, format='svg')
    plt.show()

def fundamental_dtmf_freq():
    dtmf_freqs_array = [697,770,852,941,1209,1336,1477,1633]
    print(np.gcd.reduce(dtmf_freqs_array))


if __name__ == '__main__':
    plot_signal()