import numpy as np
import scipy
import pyaudio
from scipy import signal
import scipy.io.wavfile
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from  matplotlib import patches

dtmf_freqs = {'1': (1209, 697), '2': (1336, 697), '3': (1477, 697), 'A': (1633, 697),
              '4': (1209, 770), '5': (1336, 770), '6': (1477, 770), 'B': (1633, 770),
              '7': (1209, 852), '8': (1336, 852), '9': (1477, 852), 'C': (1633, 852),
              '*': (1209, 941), '0': (1336, 941), '#': (1477, 941), 'D': (1633, 941)}

dtmf_freqs_array = [697, 770, 852, 941, 1209, 1336, 1477, 1633]


def zplane(b, a,fc, filename=None):
    """Plot the complex z-plane given a transfer function.
    """
    plt.figure()
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0,
             markeredgecolor='k', markerfacecolor='g',color='none')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0,
             markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5;
    plt.axis('scaled');
    plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1];
    # plt.xticks(ticks);
    # plt.yticks(ticks)

    if filename is None:
        plt.title('Diagrama de ceros y polos para filtro centrado en ' + str(fc))
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k



def plot_signal(x_axis,y_axis,xlabel=None,ylabel=None,title=None):
    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_superimposed_signal(x1,y1,x2,y2,xlabel=None,ylabel=None,title=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x1,y1,linewidth=2)
    plt.plot(x2,y2,'r-')
    plt.show()


def plot_two_signals(x1,y1,x2,y2,xlabel1=None,ylabel1=None,xlabel2=None,ylabel2=None,title=None,sub1=None,sub2=None):
    f,axarr = plt.subplots(2)
    f.suptitle(title)
    axarr[0].plot(x1,y1)
    axarr[0].set_title(sub1)
    axarr[0].set(xlabel=xlabel1,ylabel=ylabel1)
    axarr[1].plot(x2,y2)
    axarr[1].set(title=sub2,xlabel=xlabel2,ylabel=ylabel2)


def draw_signal():
    sample_rate, sample_array  = scipy.io.wavfile.read('modemDialing.wav')
    x_array = np.arange(len(sample_array))
    x_array_normalized = np.true_divide(x_array,sample_rate)

    return sample_rate, sample_array, x_array_normalized


def spectrogram(x_axis,fs=1.0,nfft=256,window=np.hanning,window_name='Hanning'):
    plt.figure()
    plt.specgram(x_axis,Fs=fs,scale_by_freq=True,NFFT=nfft,window=window(nfft),noverlap=nfft//2)
    plt.title(f'Espectrograma con ventana {window_name}')
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')

def fundamental_dtmf_freq():
    dtmf_freqs_array = [697,770,852,941,1209,1336,1477,1633]
    print(np.gcd.reduce(dtmf_freqs_array))

def generate_sine_wave(f,fs,duration):
    omega = 2*np.pi*f
    t_array = np.arange(0,duration,1/fs)

    return np.sin(omega*t_array)

def generate_tone(f1,f2,fs,duration=1):
    silence = np.zeros(int(0.08*fs)) #silence between each digit.
    first_sine = generate_sine_wave(f1,fs,duration)
    second_sine = generate_sine_wave(f2,fs,duration)
    sum_of_sines = first_sine + second_sine
    sum_of_sines = np.divide(sum_of_sines,2)
    chunk = sum_of_sines #lower volume
    chunk = np.append(chunk,silence)
    scaled_chunk = np.int16(chunk*32767)
    return scaled_chunk

def play_tones(chunk,fs):
    while len(chunk) < fs:
        chunk = np.append(chunk, [0])  # pad with zeros
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=fs,
                    output=True)
    stream.write(chunk.astype(np.int16).tostring())
    scipy.io.wavfile.write('ex7.wav',fs,chunk.astype(np.int16))
    stream.stop_stream()
    stream.close()

    p.terminate()

def generate_tones(digit_seq,fs,dur):

    resulting_chunk = np.empty((1,0))
    for digit in digit_seq:
        freqs = dtmf_freqs[digit]
        resulting_chunk =np.append(resulting_chunk,generate_tone(freqs[1],freqs[0],fs,dur))
    return resulting_chunk


def calculate_fft(sample_array,fs):
    N = len(sample_array)
    T = 1/fs
    yf = fft(sample_array)
    xf = np.linspace(0.0,1.0/(2.0*T),N//2)
    return xf,np.abs(yf[0:N//2])

def plot_digits_dft(sample_array,fs):
    digit_idxes = {1: (3.35, 3.45), 2: (3.48, 3.63), 3: (3.65, 3.78), 4: (3.8, 3.94), 5: (3.95, 4.08)}
    for key, val in digit_idxes.items():
        idx_1 = int(val[0] * fs)
        idx_2 = int(val[1] * fs)
        xf,yf = calculate_fft(sample_array[idx_1:idx_2], fs)
        plot_two_signals(np.arange(len(sample_array[idx_1:idx_2])),sample_array[idx_1:idx_2],xf,yf,
                         'Tiempo [s]','Amplitud [pcm]','Frecuencia [Hz]','Amplitud','Digito nro '+str(key) +' en tiempo/frecuencia','Tiempo','Frecuencia')
    xf, yf = calculate_fft(sample_array[int(3.35*fs):int(4.10*fs)], fs)
    plot_two_signals(np.arange(len(sample_array[int(3.35*fs):int(4.10*fs)]))/fs,sample_array[int(3.35*fs):int(4.10*fs)],xf,yf,
                     'Tiempo [s]','Amplitud [pcm]','Frecuencia [Hz]','Amplitud','Discado completo en tiempo/frecuencia','Tiempo','Frecuencia')

# def firwin_filter(sample_array,fs,cutoff_freq,N,window=None,plot=False):
#     freq_nyq = fs/2
#
#     if (window == None):
#         window = 'hamming' #default window for lfilter
#
#     h_lpf = signal.firwin(N,cutoff_freq/freq_nyq,pass_zero=False,window=window)
#
#     filtered_array = signal.lfilter(h_lpf,1.0,sample_array)
#
#     if (plot == True):
#         plt.figure()
#         plt.plot(h_lpf,'b',linewidth=2)
#         plt.title('Coeficientes del Filtro (%d puntos)' %N)
#         plt.grid(True)
#         plt.show()
#
#         plt.figure()
#         plt.clf()
#         w,h = signal.freqz(h_lpf,worN=8000)
#         plt.plot((w/np.pi)*freq_nyq,np.abs(h),linewidth=2)
#         plt.xlabel('Frecuencia [Hz]')
#         plt.ylabel('Ganancia')
#         plt.title('Respuesta en Frecuencia')
#         plt.ylim(-0.05,1.05)
#         plt.grid(True)
#         plt.show()
#     delay = 0.5 * (N-1) / fs
#
#     return filtered_array, delay

def normalize_signal(sample_array,filter_length=64):
    squared_signal = np.square(sample_array)
    ma_filter = np.true_divide(np.ones(filter_length),filter_length)
    signal_energy=np.convolve(squared_signal,ma_filter)
    signal_energy = signal_energy[(filter_length-1)//2:] #discard transient effects
    max_energy = np.max(signal_energy)
    return signal_energy/max_energy

def generate_kaiser_window(ripple_DB,desired_width,freq_nyq):

    width = desired_width / freq_nyq  # width of the passband
    N_kaiser, beta = signal.kaiserord(ripple_DB, width)

    return N_kaiser,beta


def classify_dft_frequencies(freq):
    # Allowing a tolerance of +-2.25% of central frequency
    # to classify possible frequencies
    tolerance = 0.0225
    for ideal_freq in dtmf_freqs_array:
        r = np.arange(int(ideal_freq*(1-tolerance)),int(ideal_freq*(1+tolerance)))
        if np.isin(round(freq),r):
            return ideal_freq

    return None

def get_freq_pair(tuple):
    for k,v in dtmf_freqs.items():
        if v == tuple:
            return k
    return None
def decode_dtmf_with_dft(signal_array,fs):
    squared_array = np.square(signal_array)

    filter_length= 64 #filter length, 0.14*8000 = 1020, N must be small enough so transient has little effect
    ma_filter = np.true_divide(np.ones(filter_length),filter_length)
    #plot_signal(np.arange(filter_length+20),np.pad(ma_filter,(10,10),mode='constant'),'Tiempo [s]','Amplitud','Filtro Moving Average')

    signal_energy_filtered = normalize_signal(signal_array,filter_length) #removed transient effects

    peak_energy = np.max(signal_energy_filtered)
    signal_energy_normalized = signal_energy_filtered/peak_energy #values between 0 and 1
    #plot_signal(np.arange(len(signal_energy_filtered))/fs,signal_energy_normalized)
    threshold = 0.7

    threshold_crosses = np.where(signal_energy_normalized > threshold)[0] #idxes where signal energy crosses threshold
    crosses_lengths = np.diff(threshold_crosses) #diff n+1 with n, starting from 0, to see differences. if diff is > 1,
    signal_edges = np.where(crosses_lengths > 1)[0]+1 #it's another signal.

    split_signal = np.split(threshold_crosses,signal_edges) #each set of points corresponds to one posible digit

    decoded_digits = np.empty([1,0])
    for enum_idx,point_set in enumerate(split_signal):
        freq_array = np.empty([1,0]) #decoded frequencies go here
        signal_part = signal_array[point_set]
        xf,yf = calculate_fft(signal_part,fs)
        if (yf.size == 0): continue
        normalized_yf = yf/np.max(yf)
        digit_threshold = 0.62
        possible_digits_yf = np.where(normalized_yf > digit_threshold )[0] #idxes where magnitude is greater than threshold
        possible_digits_xf = xf[possible_digits_yf]
        for possible_freq in possible_digits_xf:
            ideal_freq = classify_dft_frequencies(possible_freq)
            if ideal_freq is not None:
                freq_array = np.append(freq_array,[ideal_freq])
        freq_array = np.unique(freq_array) #maybe it has the same frequency more than once.
        if (len(freq_array) == 2):
            # plot_two_signals(point_set/fs,signal_array[point_set],xf,normalized_yf,'Tiempo [s]','Amplitud',
            #                  'Frecuencia [Hz]','Amplitud normalizada', 'Digito Posible nro '+str(enum_idx),'Tiempo','Frecuencia')
            # plot_signal(xf, normalized_yf, 'Frecuencia [Hz]', 'Amplitud Normalizada',
            #             'Digito Posible ' + str(enum_idx + 1))
            digit = get_freq_pair((freq_array[1],freq_array[0]))
            decoded_digits = np.append(decoded_digits,digit)

    return decoded_digits


def plot_freq_response(h_filter, filter_freq):
     plt.figure()
     w, hz = signal.freqz(h_filter, worN=8000)
     plt.title('Respuesta en Frecuencia y fase de filtro '+str(filter_freq))
     ax1 = plt.subplot(111)
     plt.plot(w, 20 * np.log10(abs(hz)), linewidth=2)
     plt.xlabel('Frecuencia [rad/sample]')
     plt.ylabel('Ganancia',color='b')
     ax2 = ax1.twinx()
     angles = np.unwrap(np.angle(hz))
     plt.plot(w,angles,'g')
     plt.ylabel('Fase',color='g')
     plt.grid(True)
     plt.show()




def passband_filter(fs,fc,band_width=30,rolloff_freq=16):
    fL = (fc-band_width)/fs
    fH = (fc+band_width)/fs
    b = rolloff_freq/fs
    M = int(np.ceil((4/b)))
    if not M % 2: M += 1 #so that it's uneven and delay is easier to calculate

    n = np.arange(M)
    hlpf_H = np.sinc(2 * fH * (n - (M - 1) / 2.))
    hlpf_H = hlpf_H / np.sum(hlpf_H)

    hlpf_L = np.sinc(2 * fL * (n - (M - 1) / 2.))
    hlpf_L = hlpf_L / np.sum(hlpf_L)

    kaiser_att = scipy.signal.kaiser_atten(M,b)
    kaiser_beta = scipy.signal.kaiser_beta(kaiser_att)



    h = hlpf_H - hlpf_L
    h *= np.blackman(M)
    #zplane(h,1,fc)
    #plot_freq_response(h,fc)
    hxf,hyf =calculate_fft(h,fs)
    #plot_two_signals(n/fs,h,hxf,hyf,'Tiempo[s]','Amplitud','Frecuencia [Hz]','Amplitud [W/Hz]','Gráfico en Tiempo y Frecuencia de Filtro de Freq '+str(fc) + ' con ventana Kaiser')
    return h,M+1




def decode_dtmf_filterbank(sample_array,fs,threshold=0.7):
    freqs_idxs = {} #store indexes where a certain frequency was detected
    for freq in dtmf_freqs_array:
        h_pb,N_pb = passband_filter(fs,freq,30,36)
        filtered_signal = np.convolve(sample_array,h_pb)[(N_pb-1)//2:] #discard transient effect

        # t_sample_array = np.arange(len(sample_array))/fs
        # plot_superimposed_signal(t_sample_array,sample_array,t_sample_array,filtered_signal[(N_pb-1)//2:],'Tiempo [s]','Amplitud','Filtrado de freq '+str(freq))

        normalized_signal = normalize_signal(filtered_signal,64)


        # xf, yf = calculate_fft(filtered_signal,fs)
        # plot_two_signals(np.arange(len(normalized_signal))/fs,normalized_signal,xf,yf,'Tiempo','Amplitud Normalizada','Frecuencia [Hz]','Amplitud','Filtro con freq '+str(freq))


        energy_crosses = np.where(normalized_signal > threshold)[0] #parts of the signal that cross the threshold
        freqs_idxs[freq] = []
        if (len(energy_crosses) >0 ):
            cross_lengths = np.diff(energy_crosses)
            possible_tones_idxes = np.where(cross_lengths > 200)[0]+1 #add 1 to correctly display idxes
            possible_tones_intervals = np.split(energy_crosses,possible_tones_idxes)
            for tone_interval in possible_tones_intervals:
                freqs_idxs[freq].append(tone_interval)

    guessed_sequence = []
    for low_freq in dtmf_freqs_array[0:4]:
        for high_freq in dtmf_freqs_array[4:]:
            for low_freq_interval in freqs_idxs[low_freq]:
                for high_freq_interval in freqs_idxs[high_freq]:
                    if len(np.intersect1d(high_freq_interval,low_freq_interval)) > 200:
                        guessed_freq = [k for k, v in dtmf_freqs.items() if v == (high_freq, low_freq)]
                        guessed_tuple = (low_freq_interval[0],guessed_freq[0])
                        guessed_sequence.append(guessed_tuple)

    dtype = [('index',int),('value', '|S2')]
    unordered_array = np.array(guessed_sequence,dtype=dtype)
    ordered_array = np.sort(unordered_array,order='index') #array ordered by indexes

    ordered_sequence = np.empty((1,0))

    for num,string in ordered_array:
      ordered_sequence = np.append(ordered_sequence,(string.decode('UTF-8'))) #decode npy string back to utf8

    return ordered_sequence

def pretty_print_array(array):
    return print(np.array2string(array, separator=','))

def ex_1(fs,sample_array,normalized_x):
    plot_signal(normalized_x, sample_array, 'Tiempo [s]', 'Amplitud [pcm]')

def ex_2(fs):
    test_tone = generate_tone(800, 1200, fs, 0.07)
    play_tones(test_tone,fs)
    plot_signal(np.arange(len(test_tone)) / fs, test_tone, 'Tiempo [s]', 'Amplitud [s]', 'Señal de prueba')

def ex_3(fs,sample_array,nperseg=256):
    spectrogram(sample_array,fs,nperseg,np.blackman,'Blackman')
    plot_digits_dft(sample_array,fs)

def ex_7(fs):
    digit_seq = '32327'
    chunk = generate_tones(digit_seq,fs,0.07)
    play_tones(chunk,fs)

def ex_8(fs,digit_seq):
    print(f'Original sequence: {digit_seq}')
    res_chunk = generate_tones(digit_seq, fs, 0.07)
    pretty_print_array(decode_dtmf_with_dft(res_chunk, fs))

def ex_9(fs,digit_seq):
    print(f'Original sequence: {digit_seq}')
    res_chunk = generate_tones(digit_seq,fs,0.07)
    t_chunk = np.true_divide(np.arange(len(res_chunk)),fs)
    pretty_print_array(decode_dtmf_filterbank(res_chunk,fs))

def ex_10(fs):
    for freq in dtmf_freqs_array:
        h,N_pb = passband_filter(fs,freq,30,36)
        plot_freq_response(h,freq)
        zplane(h,1.0,freq)

def ex_11(fs,digit_seq,noise_intensity):
    print(f'Original sequence: {digit_seq}')
    res_chunk = generate_tones(digit_seq,fs,0.07)
    res_chunk_noisy = res_chunk + np.random.randn(len(res_chunk))*noise_intensity
    t_chunk = np.true_divide(np.arange(len(res_chunk)),fs)
    plot_signal(t_chunk,res_chunk_noisy,'Tiempo[s]','Amplitud [pcm]', 'Secuencia con ruido')
    print('Filtro con DFT: ',end='')
    pretty_print_array(decode_dtmf_with_dft(res_chunk_noisy, fs))
    print('Banco de Filtros FIR: ',end='')
    pretty_print_array(decode_dtmf_filterbank(res_chunk_noisy,fs))

if __name__ == '__main__':
    fs,sample_array,normalized_x =draw_signal()
    digit_seq = 'ABCD123456789*#0'
    ex_3(fs,sample_array,512)