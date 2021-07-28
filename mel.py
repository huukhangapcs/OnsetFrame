import tensorflow as tf
import math
import mel_configs
import mfcc_mel_utils
import numpy as np
import fractions

def _fixed_frame(signal, frame_length, frame_step, first_axis=False):
  """tflite-compatible tf.signal.frame for fixed-size input.
  Args:
    signal: Tensor containing signal(s).
    frame_length: Number of samples to put in each frame.
    frame_step: Sample advance between successive frames.
    first_axis: If true, framing is applied to first axis of tensor; otherwise,
      it is applied to last axis.
  Returns:
    A new tensor where the last axis (or first, if first_axis) of input
    signal has been replaced by a (num_frames, frame_length) array of individual
    frames where each frame is drawn frame_step samples after the previous one.
  Raises:
    ValueError: if signal has an undefined axis length.  This routine only
      supports framing of signals whose shape is fixed at graph-build time.
  """
  signal_shape = signal.shape.as_list()
  if first_axis:
    length_samples = signal_shape[0]
  else:
    length_samples = signal_shape[-1]
  if length_samples <= 0:
    raise ValueError('fixed framing requires predefined constant signal length')
  num_frames = max(0, 1 + (length_samples - frame_length) // frame_step)
  if first_axis:
    inner_dimensions = signal_shape[1:]
    result_shape = [num_frames, frame_length] + inner_dimensions
    gather_axis = 0
  else:
    outer_dimensions = signal_shape[:-1]
    result_shape = outer_dimensions + [num_frames, frame_length]
    # Currently tflite's gather only supports axis==0, but that may still
    # work if we want the last of 1 axes.
    gather_axis = len(outer_dimensions)

  subframe_length = fractions.gcd(frame_length, frame_step)  # pylint: disable=deprecated-method
  subframes_per_frame = frame_length // subframe_length
  subframes_per_hop = frame_step // subframe_length
  num_subframes = length_samples // subframe_length

  if first_axis:
    trimmed_input_size = [num_subframes * subframe_length] + inner_dimensions
    subframe_shape = [num_subframes, subframe_length] + inner_dimensions
  else:
    trimmed_input_size = outer_dimensions + [num_subframes * subframe_length]
    subframe_shape = outer_dimensions + [num_subframes, subframe_length]
  subframes = tf.reshape(
      tf.slice(
          signal,
          begin=np.zeros(len(signal_shape), np.int32),
          size=trimmed_input_size), subframe_shape)

  # frame_selector is a [num_frames, subframes_per_frame] tensor
  # that indexes into the appropriate frame in subframes. For example:
  # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
  frame_selector = np.reshape(
      np.arange(num_frames) * subframes_per_hop, [num_frames, 1])

  # subframe_selector is a [num_frames, subframes_per_frame] tensor
  # that indexes into the appropriate subframe within a frame. For example:
  # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
  subframe_selector = np.reshape(
      np.arange(subframes_per_frame), [1, subframes_per_frame])

  # Adding the 2 selector tensors together produces a [num_frames,
  # subframes_per_frame] tensor of indices to use with tf.gather to select
  # subframes from subframes. We then reshape the inner-most subframes_per_frame
  # dimension to stitch the subframes together into frames. For example:
  # [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
  selector = frame_selector + subframe_selector
  frames = tf.reshape(
      tf.gather(subframes, selector.astype(np.int32), axis=gather_axis),
      result_shape)
  return frames

def create_mel_graph(inputs, training=True):
    window_length_samples = int(round(mel_configs.window_length_seconds * mel_configs.sample_rate))
    hop_length_samples = int(round(mel_configs.hop_length_seconds * mel_configs.sample_rate))
    fft_length = 2**int(math.ceil(math.log(window_length_samples) / math.log(2.0)))
    
    #noise layer
    if mel_configs.noise_scale != 0.0 and training:
        net = tf.keras.layers.GaussianNoise(stddev=mel_configs.noise_scale)(inputs)
    else:
        net  = tf.keras.layers.Lambda(lambda x: x)(inputs)
    
    #`calculate magnitude_spectrogram` => [?, fft_length/2 + 1] tensor of spectrograms.
    net = tf.abs(
        tf.signal.stft(
          net,
          frame_length=window_length_samples,
          frame_step=hop_length_samples,
          fft_length=fft_length),
        name='magnitude_spectrogram')


    num_spectrogram_bins = int(net.shape[-1])
    linear_to_mel_weight_matrix = tf.constant(
                                        mfcc_mel_utils.SpectrogramToMelMatrix(mel_configs.num_mel_bins, num_spectrogram_bins,
                                        mel_configs.sample_rate, mel_configs.lower_edge_hz,
                                        mel_configs.upper_edge_hz).astype(np.float32),
                                        name='linear_to_mel_matrix')
    
    net = tf.matmul(net,linear_to_mel_weight_matrix, name='mel_spectrogram')
    log_offset = 0.001
    net = tf.math.log(net + log_offset, name='log_mel_spectrogram')
    # net = _fixed_frame(net,frame_length=mel_configs.frame_width,frame_step=mel_configs.frame_hop,first_axis=True)
    net = tf.signal.frame(
        net,
        frame_length=mel_configs.frame_width,
        frame_step=mel_configs.frame_hop,
        axis=0)

    return net