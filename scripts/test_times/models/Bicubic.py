''' BICUBIC '''
import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
try:
    from scripts.test_times.convert import convert_model
except Exception:
    sys.path.append(sys.path[0].replace('scripts/test_times/models', ''))
    from scripts.test_times.convert import convert_model


def bicubic(scale: int):
    ''' bicubic model '''
    input = layers.Input(shape=(360, 640, 3))
    x = layers.Lambda(lambda x: tf.image.resize(x, size=[360*scale, 640*scale]))(input)
    x = layers.Lambda(lambda x: K.clip(x, 0., 255.))(x)
    return models.Model(inputs=input, outputs=x)


def main():
    ''' main '''
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', required=True, type=int, help='scale')
    args = parser.parse_args()

    # build model
    model = bicubic(args.scale)
    model.summary()

    # prepare
    name = f'X{args.scale}'
    os.makedirs('experiments/test_times', exist_ok=True)
    os.makedirs('experiments/test_times/models', exist_ok=True)
    os.makedirs('experiments/test_times/models/BICUBIC', exist_ok=True)
    os.makedirs('experiments/test_times/models/BICUBIC/int8', exist_ok=True)

    # to tflite
    model.save(f'/tmp/BICUBIC/{name}', overwrite=True, include_optimizer=True, save_format='tf')
    convert_model(f'/tmp/BICUBIC/{name}', f'experiments/test_times/models/BICUBIC/{name}.tflite', args.scale, time=True)

    # to int8 tflite
    convert_model(f'/tmp/BICUBIC/{name}', f'experiments/test_times/models/BICUBIC/int8/{name}.tflite', args.scale, mode='int8', time=True)


if __name__ == '__main__':
    main()
