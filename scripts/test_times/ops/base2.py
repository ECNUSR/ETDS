''' ETDS '''
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
try:
    from scripts.test_times.convert import convert_model
except Exception:
    sys.path.append(sys.path[0].replace('scripts/test_times/ops', ''))
    from scripts.test_times.convert import convert_model


def add():
    ''' ETDS inference stage model '''
    input = layers.Input(shape=(None, None, 3))
    out = layers.Conv2D(32, 3, padding='same')(input)
    for _ in range(4):
        out = layers.Conv2D(32, 3, padding='same')(out)
    out = layers.Conv2D(27, 3, padding='same')(out)
    out = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 3))(out)
    out = layers.Add()([out, out])
    return models.Model(inputs=input, outputs=out)


def main():
    ''' main '''
    # build model
    model = add()
    model.summary()

    # prepare
    os.makedirs('experiments/test_times', exist_ok=True)
    os.makedirs('experiments/test_times/ops', exist_ok=True)
    os.makedirs('experiments/test_times/ops/base', exist_ok=True)
    os.makedirs('experiments/test_times/ops/base/int8', exist_ok=True)

    # to tflite
    model.save('/tmp/base/base', overwrite=True, include_optimizer=True, save_format='tf')
    convert_model('/tmp/base/base', 'experiments/test_times/ops/base/base2.tflite', 2, input_channel=3, time=True)

    # to int8 tflite
    convert_model('/tmp/base/base', 'experiments/test_times/ops/base/int8/base2.tflite', 2, input_channel=3, mode='int8', time=True)


if __name__ == '__main__':
    main()
