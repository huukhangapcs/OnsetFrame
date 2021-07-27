from mel import create_mel_graph
import tensorflow as tf
import OnsetFrameModel
inputs = tf.keras.Input((1600,))
net = create_mel_graph(inputs) 
net = OnsetFrameModel.model_fn(net)
model = tf.keras.Model(inputs, net)
print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
