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

def model_fn(inputs,is_training=True):
    """Builds the acoustic model."""
    #Onset
    onset_stack = conv_net(inputs)
    onset_stack = Bidirectional(LSTM(configs.onset_lstm_units, return_sequences=True))(onset_stack)
    onset_stack = Dense(configs.MIDI_PITCHES, activation="sigmoid")(onset_stack) 
    #Frame
    frame_stack = conv_net(inputs)
    frame_stack = Dense(configs.MIDI_PITCHES, activation="sigmoid")(frame_stack) 
    #Concat
    concat_inputs = tf.keras.layers.Concatenate()([tf.stop_gradient(onset_stack), frame_stack])
    combined_stack = Bidirectional(LSTM(configs.onset_lstm_units))(concat_inputs)
    combined_stack = Dense(configs.MIDI_PITCHES, activation="sigmoid")(combined_stack) 


    # activation_outputs = acoustic_model(
    #         inputs,
    #         lstm_units=0)
    # activation_probs = Dense(
    #         constants.MIDI_PITCHES,
    #         activation="sigmoid")(activation_outputs)
    
    return combined_stack
    




