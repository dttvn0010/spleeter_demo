from scipy.io import wavfile
from tensorflow.contrib import predictor

input_file = "audio_example.wav"
waveform = wavfile.read(input_file)[1]
waveform = waveform/32768.0

model = predictor.from_saved_model('model')
result = model({'waveform': waveform, 'audio_id': input_file})
wavfile.write('vocals.wav', 44100, result['vocals'])
wavfile.write('accompaniment.wav', 44100, result['accompaniment'])
