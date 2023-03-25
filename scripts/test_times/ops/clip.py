''' ETDS '''
import os
import sys
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
try:
    from scripts.test_times.convert import convert_model
except Exception:
    sys.path.append(sys.path[0].replace('scripts/test_times/ops', ''))
    from scripts.test_times.convert import convert_model


def clip():
    ''' ETDS inference stage model '''
    input = layers.Input(shape=(None, None, 3))
    out = layers.Conv2D(32, 3, padding='same')(input)
    for _ in range(4):
        out = layers.Conv2D(32, 3, padding='same')(out)
    out = layers.Conv2D(27, 3, padding='same')(out)
    out = layers.Lambda(lambda x: K.clip(x, 0., 255.))(out)
    return models.Model(inputs=input, outputs=out)


def main():
    ''' main '''
    # build model
    model = clip()
    model.summary()

    # prepare
    os.makedirs('experiments/test_times', exist_ok=True)
    os.makedirs('experiments/test_times/ops', exist_ok=True)
    os.makedirs('experiments/test_times/ops/clip', exist_ok=True)
    os.makedirs('experiments/test_times/ops/clip/int8', exist_ok=True)

    # to tflite
    model.save('/tmp/clip/clip', overwrite=True, include_optimizer=True, save_format='tf')
    convert_model('/tmp/clip/clip', 'experiments/test_times/ops/clip/clip.tflite', 2, input_channel=3, time=True)

    # to int8 tflite
    convert_model('/tmp/clip/clip', 'experiments/test_times/ops/clip/int8/clip.tflite', 2, input_channel=3, mode='int8', time=True)


if __name__ == '__main__':
    main()
