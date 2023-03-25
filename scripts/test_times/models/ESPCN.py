''' ESPCN '''
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

def espcn(scale: int):
    ''' espcn model '''
    input = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(64, 3, padding='same', activation='tanh')(input)
    x = layers.Conv2D(32, 3, padding='same', activation='tanh')(x)
    x = layers.Conv2D((scale**2)*3, 3, padding='same')(x)
    x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)
    x = K.sigmoid(x)
    return models.Model(inputs=input, outputs=x)


def main():
    ''' main '''
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', required=True, type=int, help='scale')
    args = parser.parse_args()

    # build model
    model = espcn(args.scale)
    model.summary()

    # prepare
    name = f'X{args.scale}'
    os.makedirs('experiments/test_times', exist_ok=True)
    os.makedirs('experiments/test_times/models', exist_ok=True)
    os.makedirs('experiments/test_times/models/ESPCN', exist_ok=True)
    os.makedirs('experiments/test_times/models/ESPCN/int8', exist_ok=True)

    # to tflite
    model.save(f'/tmp/ESPCN/{name}', overwrite=True, include_optimizer=True, save_format='tf')
    convert_model(f'/tmp/ESPCN/{name}', f'experiments/test_times/models/ESPCN/{name}.tflite', args.scale, time=True)

    # to int8 tflite
    convert_model(f'/tmp/ESPCN/{name}', f'experiments/test_times/models/ESPCN/int8/{name}.tflite', args.scale, mode='int8', time=True)


if __name__ == '__main__':
    main()
