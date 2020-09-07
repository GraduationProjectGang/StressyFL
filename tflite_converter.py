import tensorflow as tf

org_model = tf.keras.models.load_model('78Model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(org_model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('78_model.tflite', 'wb') as f:
    f.write(tflite_model)