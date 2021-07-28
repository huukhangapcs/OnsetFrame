temporal_sizes=[3, 3]
freq_sizes=[3, 3]
num_filters=[48, 96]
pool_sizes=[1,2]
dropout_keep_amts=[1.0, 0.75]
fc_size = 384
fc_dropout_keep_amt=0.5
onset_lstm_units=256
import librosa


MIN_MIDI_PITCH = librosa.note_to_midi('A0')
MAX_MIDI_PITCH = librosa.note_to_midi('C8')
MIDI_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1