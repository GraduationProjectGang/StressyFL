import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

org_model = tf.keras.models.load_model('best_model_2_10.h5')

org_model.save('test_model', save_format='tf')

converter = tf.lite.TFLiteConverter.from_saved_model('test_model')
tflite_model = converter.convert()

with tf.io.gfile.GFile('final_model.tflite', 'wb') as f:
    f.write(tflite_model)