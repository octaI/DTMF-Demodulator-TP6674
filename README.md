# TP6674
Repositorio para el TP de Señales y Sistemas.

Repository that contains a special assignment for the subject Signals and systems.


It consists of a DTMF tone demodulator through 2 methods:
Decoding the sequence through DFT, using the FFT
Utilizing a filter bank, with filters designed through the Window method.

## Prerequisites

You will need to have pip3, as well as virtualenv installed.

Virtualenv can be installed by:

`pip3 install virtualenv`


## Running dtmf demoduler

1. First, create a new virtualenv. We will choose 'dtmfdecoder' as the name of our example virtualenv
but you can change the name to whichever you want to.

     `$ virtualenv dtmfdecoder`

2. Activate the virtualenv by executing

     `$ source dtmfdecoder/bin/activate`

3. Install the required libraries by executing
    
    `(dtmfdecoder) $ pip3 install -r requirements.txt`
    
4. Execute the script
    
    `(dtmfdecoder) $ python3 main.py  `
    
5. To uninstall, simply remove the dtmfdecoder folder created by virtualenv.



## Usage

Bundled with *main.py* there are some exercises that depict the usage of this tool. Bear in mind
that it is just an implementation and it is intended to be used in an investigational way.
You will find several utilitary functions as well as comented lines throughout the functions,
those serve as points where you would like a plot of a certain phenomenon or stage of the
decoding process.

You have two ways of testing the decoders:

* Utilizing the `generate_tones` function, which receives a string sequence of DTMF characters
and generates its corresponding output. You can also specify a filename and save that output
to a wav file, for external reproduction.

* Reading an existing wavfile with the `draw_signal` function. It will return the sampling frequency, as well as the sample array and 
the x axis, normalized by the sampling frequency so you plot in the time domain, if you want to.


**Note**: Keep in mind that, the tone generator is suited to generate the signals utilizing np.int16 data type.
If required,  you will need to change the data type to whatever you need (int32, float64, etc.).

**Note 2**: When utilizing the `generate_tones` function, it will generate the sequence and play
it using the PyAudio library. Sometimes it will not play correctly with PyAudio, but opening the resulting
.wav file with your player of choice will do.


## Project Info

This project was made as an special assignment for the Signals and Systems course at the Buenos Aires
University Faculty of Engineering (FIUBA - Facultad de Ingeniería de la Universidad de Buenos Aires)

## Credits

To Christopher Felton for his z-plot function which I slightly modified!

https://www.dsprelated.com/showcode/244.php
