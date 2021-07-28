import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Dense, Bidirectional, LSTM
import model_configs as configs
def conv_net(inputs):
    net = inputs
    for (conv_temporal_size, conv_freq_size,
        num_filters, freq_pool_size, dropout_amt) in zip(
            configs.temporal_sizes, configs.freq_sizes, configs.num_filters,
            configs.pool_sizes, configs.dropout_keep_amts):
        
        net = Conv2D(num_filters, (conv_temporal_size, conv_freq_size))(net)
        net = BatchNormalization()(net)
        if freq_pool_size > 1:
            net = MaxPool2D((1, freq_pool_size), strides=(1, freq_pool_size))(net)
        if dropout_amt < 1:
            net =  Dropout(dropout_amt)(net)
    
    dims = tf.shape(net)
    net = tf.reshape(net, (dims[0], dims[1], net.shape[2] * net.shape[3]))
    net = Dense(configs.fc_size)(net)
    net = Dropout(configs.fc_dropout_keep_amt)(net)
    return net


def acoustic_model(inputs, lstm_units):
    """Acoustic model that handles all specs for a sequence in one window."""
    conv_output = conv_net(inputs)
    if lstm_units:
        return Bidirectional(LSTM(lstm_units))(conv_output)

    else:
        return conv_output

def model_fn(inputs, labels=None, mode=None):
    """Builds the acoustic model."""
    
    #Onset
    onset_outputs = acoustic_model(inputs,
          lstm_units=configs.onset_lstm_units)
    onset_probs = Dense(
          configs.MIDI_PITCHES,
          activation="sigmoid")(onset_outputs)
    return onset_probs
    




