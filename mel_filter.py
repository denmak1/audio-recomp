import sys
import math
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from scipy.fftpack import fft, fftfreq

def freq_to_mel(freq):
  return 1125.0 * math.log(1.0 + (float(freq)/700.0))
# END freq_to_mel

def mel_to_freq(mel):
  return 700.0 * (math.exp(float(mel)/1125.0) - 1.0)
# END mel_to_freq

def freq_to_fbin(freq, fft_size, samplerate):
  return math.floor((fft_size+1) * freq / samplerate)
# END freq_to_fbin

class MelFilter:
  # Vars:
  # frange - natural frequency range
  # num_windows - number of windows
  # samplerate
  # framesize
  # fft_size
  # input - input sound signal
  # mrange - mel-freq range
  # mfreqs - mel frequencies after dividing the mrange
  # nfreqs - natural frequencies after converting from mel scale
  # fbins - fft bins from the natural frequencies
  # max_amp - max amplitude

  def __init__(self, _frange, _num_windows, _samplerate, _framesize, _fft_size):
    self.frange = _frange
    self.num_windows = _num_windows
    self.samplerate = _samplerate
    self.framesize = _framesize
    self.fft_size = _fft_size

  def set_input(self, _input):
    self.input = _input

  def calc_mel_filters(self):
    num_samples = float(self.framesize)/1000.0 * float(self.samplerate)

    self.mrange = [freq_to_mel(x) for x in self.frange]
    print(self.mrange)

    # split mel-freq range into linear windows
    step_size = (self.mrange[1] - self.mrange[0]) / float(self.num_windows+1)

    self.mfreqs = []
    self.mfreqs.append(self.mrange[0])
    c_freq = self.mrange[0]
    for i in range(self.num_windows+1):
      c_freq = c_freq + step_size
      self.mfreqs.append(c_freq)

    print(self.mfreqs)

    # convert mel-freqs to natural frequencies
    self.nfreqs = [mel_to_freq(x) for x in self.mfreqs]
    print(self.nfreqs)

    # convert nat-freqs to freq bins based on fft
    self.fbins = \
      [freq_to_fbin(x, self.fft_size, self.samplerate) for x in self.nfreqs]
    print(self.fbins)

  def plot_fft(self):
    print("Plot fft")

    x_fft = fft(self.input)
    x_fft_mag = np.absolute(x_fft)                       # magnitudes
    f = np.linspace(0, self.samplerate, len(x_fft_mag))  # frequencies
    print("fft shapes:", f.shape, x_fft_mag.shape)

    # cut off frequencies after 2000 because guitar frequency ranges from
    # 80 to 1400 in std tuning and 24 frets
    #fft_bin_lim = freq_to_fbin([2000.0], self.fft_size, self.samplerate)
    #print("fft_bin_lim:", fft_bin_lim)
    #f = f[:fft_bin_lim]
    #x_fft_mag = x_fft_mag[:fft_bin_lim]

    # append 0 if not even length
    if (len(f)%2 != 0):
      f = np.append(f, [0.0])
      x_fft_mag = np.append(x_fft_mag, [0.0])

    f = np.split(f, 2)[0]
    x_fft_mag = np.split(x_fft_mag, 2)[0]
    self.max_amp = max(x_fft_mag)

    # plot magnitude vs frequency
    plt.plot(f, x_fft_mag)

  def plot_freqs(self):
    plt.clf()

    self.plot_fft()

    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (Hz)")

    # make points
    x = [0]
    y = [0]
 
    # start from 0
    top = True
    for f in self.nfreqs:
      if (top):
        x.append(f)
        y.append(self.max_amp)

        x.append(f)
        y.append(self.max_amp)

        top = False
      else:
        x.append(f)
        y.append(0)

        x.append(f)
        y.append(0)

        top = True

    x_2 = [self.nfreqs[0]]
    y_2 = [0]

    # start from 0
    top = False
    for f in self.nfreqs:
      if (top):
        x.append(f)
        y.append(self.max_amp)

        x.append(f)
        y.append(self.max_amp)

        top = False
      else:
        x.append(f)
        y.append(0)

        x.append(f)
        y.append(0)

        top = True

    for i in range(0, len(x), 2):
      plt.plot(x[i:i+2], y[i:i+2], 'ro-')

    for i in range(0, len(x_2), 2):
      plt.plot(x_2[i:i+2], y_2[i:i+2], 'ro-')

    plt.show()
  # END plot_freqs
# END MelFilter
