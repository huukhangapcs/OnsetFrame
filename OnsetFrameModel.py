import tensorflow as tf
from tensorflow.keras.initializers    import VarianceScaling
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Activation, Dense, Bidirectional, LSTM
def conv_net(inputs, withLstm):
    net = Conv2D(48,3, padding='same',use_bias=False, kernal_initializer=VarianceScaling())(inputs)
    net = BatchNormalization(scale=False)(net)
    net = Activation('relu')(net)
    net = Conv2D(48,3, padding='same',use_bias=False, kernal_initializer=VarianceScaling())(net)
    net = BatchNormalization(scale=False)(net)
    net = Activation('relu')(net)
    net = MaxPool2D((1, 2))(net)
    net = Conv2D(96,3, padding='same',use_bias=False, kernal_initializer=VarianceScaling())(net)
    net = BatchNormalization(scale=False)(net)
    net = Activation('relu')(net)
    net = MaxPool2D((1, 2))(net)
    net = Dense(768, use_bias=False, kernel_initializer=VarianceScaling())(net)
    net = BatchNormalization(scale=False)(net)
    net = Activation('relu')(net)
    if withLstm: outputs = Bidirectional(LSTM(256, kernel_initializer=VarianceScaling(), # (2)
                                              return_sequences=True))(net)
    net = Dense(88,activation='sigmoid')
    return net

def model_fn(inputs):
    onset_stack = conv_net(inputs, withLstm=True)
    frame_stack = conv_net(inputs, withLstm=False)
    concat_inputs = tf.keras.layers.Concatenate()([onset_stack, frame_stack])
    combined_stack = Bidirectional(LSTM(256, kernel_initializer=VarianceScaling(), return_sequences=True))(concat_inputs)
    combined_stack = Dense(88,activation='sigmoid')(combined_stack)
    return combined_stack

