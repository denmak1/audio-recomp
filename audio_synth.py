import sys
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.feature
import sklearn
import peakutils

from mel_filter import *

# gets the dominant frequency from the audio segment
def get_dom_freq(sr, seg):
  print(sr, seg)

  # 1) estimate pitch from dominant frequency
  f, fft_mags = get_fft(seg, sr, False)

  fft_mags = list(fft_mags)
  peak_indices = peakutils.indexes(fft_mags)

  #for i in peak_indices:
  #  print(f[i], "with magnitude of", fft_mags[i])

  peak_idx = peak_indices[0]

  # 2) try to deduce pitch with autocorrelation from power specturm
  max_size = seg.shape[-1]

  # get power spectrum of segment
  seg_fft = fft(seg, n=2 * seg.shape[-1] + 1, axis=-1)
  power_spec = np.abs(seg_fft) ** 2

  # inverse fft to get time-domain of power spectrum
  seg_ifft = ifft(power_spec, axis=-1)

  # resize to segment size
  subslice = [slice(None)] * seg_ifft.ndim
  subslice[-1] = slice(max_size)
  seg_ifft = seg_ifft[tuple(subslice)]

  # ignore frequencies below 50 Hz and above 2000 Hz
  min_idx = sr / 2000.0
  max_idx = sr / 50.0
  seg_ifft[:int(min_idx)] = 0
  seg_ifft[int(max_idx):] = 0

  # dominant freq
  dom_freq = float(sr) / seg_ifft.argmax()

  # peak freq, magintude, dominant frequency from power spec
  return f[peak_idx], fft_mags[peak_idx], dom_freq
# END get_dom_freq

def pca(x, sr, show_plot):
  print("Perform PCA")

  frame_size = int(float(sr) * (20.0/1000.0))
  X = librosa.feature.mfcc(x, sr = sr, hop_length = frame_size)

  print("frame_size", frame_size)
  print("mfcc shape:", X.shape)

  X = sklearn.preprocessing.scale(X)
  print(X.mean())

  model = sklearn.decomposition.PCA(n_components = 2, whiten=True)
  model.fit(X.T)
  Y = model.transform(X.T)

  print(Y.shape)
  print(model.components_.shape)

  if (show_plot):
    plt.clf()
    plt.scatter(Y[:,0], Y[:,1], s=3)
    plt.show()

  return Y
# END pca

def nmf(x, sr):
  print("Perform NMF")

  S = librosa.stft(x)

  # display a spectogram
  plt.clf()
  librosa.display.specshow(librosa.amplitude_to_db(np.abs(S), ref=np.max),
                           y_axis='log',
                           x_axis='time')
  plt.title('spectrogram')
  plt.colorbar(format='%+2.0f dB')
  plt.tight_layout()
  plt.show()


  X, X_phase = librosa.magphase(S)
  n_components = 6
  W, H = librosa.decompose.decompose(X, n_components=n_components, sort=True)

  print("W shape =", W.shape)
  print("H shape =", H.shape)

  plt.clf()
  plt.plot(W[:,1])
  plt.show()

  return
# END nmf

def plot_signal(x, sr):
  print("Plot signal")

  plt.clf()
  plt.subplot(1, 1, 1);
  librosa.display.waveplot(x, sr = sr)
  #plt.show()

  print("signal:", x.shape, sr)

  # harmonic and percussive plot
  #y_harm, y_perc = librosa.effects.hpss(x)
  #librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5) 

  # TODO: adjust?
  hop_length = 256
  frame_length = 512

  # calculate energy
  energy = np.array([
    sum(abs(x[i:i+frame_length]**2))
    for i in range(0, len(x), hop_length)
  ])

  # normalize energy to between 0 and 1
  energy *= (max(x)/energy.max())

  print("energy:", energy.shape)

  # calculate RMSE
  rmse = librosa.feature.rms(x,
                             frame_length = frame_length,
                             hop_length = hop_length,
                             center = True)
  print("rmse:", rmse.shape)
  rmse = rmse[0]

  # overlay RMSE and energy on the signal graph
  frames = range(len(energy))
  t = librosa.frames_to_time(frames,
                             sr=sr,
                             hop_length = hop_length)

  print("time:", t.shape)

  #plt.subplot(1, 1, 2)
  plt.plot(t, rmse, color='y', label = "RMS")
  plt.plot(t, energy, color='r', label = "Energy", alpha=0.5)
  plt.ylabel("Intensity (per channel)")
  plt.legend()

  plt.show()
# END plot_signal

def get_fft(x, sr, show_plot):
  print("Get fft")

  x_fft = fft(x)
  x_fft_mag = np.absolute(x_fft)          # magnitudes
  f = np.linspace(0, sr, len(x_fft_mag))  # frequencies

  # cut off frequencies after 1600 because guitar frequency ranges from
  # 80 to 1400 in std tuning and 24 frets
  f = f[:5000]
  x_fft_mag = x_fft_mag[:5000]

  # plot magnitude vs frequency
  if (show_plot):
    plt.plot(f, x_fft_mag)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.show()

  return f, x_fft_mag
# END plot_fft

def onset_seg(x, sr, show_plot):
  print("Onset seg")

  # pick out onset frames
  hop_length = 512
  onset_frames = \
    librosa.onset.onset_detect(x,
                               sr = sr,
                               hop_length = hop_length)
  print(onset_frames)

  # convert frames to timestamps
  onset_times = \
    librosa.frames_to_time(onset_frames,
                           sr = sr,
                           hop_length = hop_length)
  print(onset_times)

  # plot spectrogram
  S = librosa.stft(x)
  logS = librosa.amplitude_to_db(np.abs(S))

  if (show_plot):
    librosa.display.specshow(logS,
                             sr = sr,
                             x_axis = 'time',
                             y_axis = 'log')

    # overlay onset frames
    plt.vlines(onset_times, 0, 10000, color='g')
    plt.show()

  # split source audio on the segments
  onset_samples = \
    librosa.frames_to_samples(onset_frames,
                              hop_length = hop_length)

  # append a frame of silence to segment 
  frame_size = min(np.diff(onset_samples))
  silent_frame = np.zeros(int(0.5 * sr))

  segments = []
  for i in onset_samples:
    segments.append(np.concatenate([x[i:i+frame_size], silent_frame]))

  # pad 0 and end of input to samples
  onset_samples = np.concatenate([onset_samples, [len(x)]])

  return sr, segments, onset_samples
# END onset_seg

def print_data(x, sr):
  print("Print Data")
  print("x =", x, "len =", len(x)) 
  print("sr =", sr)
# END print_data

def generate_synth(x, sr):
  # get segments
  sr, segments, seg_samples = onset_seg(x, sr, False)
  print("num segs, num samples =", len(segments), seg_samples)

  # for each seg, approximate pitch using dominant frequency
  digi_tones = []
  i = 1
  for s in segments:
    # pitch detect
    peak_freq, mag, dom_freq = get_dom_freq(sr, s)
    print(dom_freq, "with magnitude of", mag)

    time_len = librosa.get_duration(y = s, sr = sr)
    print(time_len)

    # calculate frame size
    seg_sample_size = seg_samples[i] - seg_samples[i-1]

    # generate tone
    n = np.arange(seg_sample_size)
    sine_sig = 0.2 * np.sin(2 * np.pi * dom_freq * n / float(sr))

    digi_tones.append(sine_sig)
    i += 1

  return np.concatenate(digi_tones)
# END generate_synth

def mel_filter(x, sr):
  # parameters to the mel filterbank finder
  frange = [10.0, 10000.0]
  num_windows = 6
  framesize_ms = 20          # milliseconds
  fft_size = 512

  # caclculate the frame length in samples
  frame_size = int(float(sr) * (float(framesize_ms)/1000.0))

  x = x[12000:12000+frame_size]

  mf = MelFilter(frange, num_windows, sr, framesize_ms, fft_size)
  mf.set_input(x)
  mf.calc_mel_filters()

  mf.plot_freqs()
# END mel_filter

def main():
  fname = sys.argv[1]
  print(fname)

  # load file
  # x = numpy array
  # sr = sampling rate
  x, sr = librosa.load(fname)
  print_data(x, sr)

  # get_fft(x, sr, True)

  # generate synth of input
  x_synth = generate_synth(x, sr)

  # write synth to file
  librosa.output.write_wav("audio_synth/" + fname.split("/")[1],
                           x_synth,
                           sr,
                           norm = True)

  # out of curiosity, draw the spectrogram
  onset_seg(x, sr, True)
  onset_seg(x_synth, sr, True)

  # print_data(x, sr)

  # plot_signal(x, sr)

  # pca(x, sr)

  # nmf(x, sr)

  # mel_filter(x, sr)
# END main

main()
