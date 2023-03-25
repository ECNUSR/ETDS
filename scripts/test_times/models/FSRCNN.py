''' FSRCNN '''
import os
import sys
import argparse
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
try:
    from scripts.test_times.convert import convert_model
except Exception:
    sys.path.append(sys.path[0].replace('scripts/test_times/models', ''))
    from scripts.test_times.convert import convert_model


def fsrcnn(scale: int):
    ''' fsrcnn model '''
    input = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(56, 5, padding='same')(input)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(12, 1, padding='same')(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    for _ in range(4):
        x = layers.Conv2D(12, 3, padding='same')(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(56, 1, padding='same')(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2DTranspose(3, 9, strides=scale, padding='same')(x)
    out = layers.Lambda(lambda x: K.clip(x, 0., 255.))(x)
    return models.Model(inputs=input, outputs=out)


def main():
    ''' main '''
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', required=True, type=int, help='scale')
    args = parser.parse_args()

    # build model
    model = fsrcnn(args.scale)
    model.summary()

    # prepare
    name = f'X{args.scale}'
    os.makedirs('experiments/test_times', exist_ok=True)
    os.makedirs('experiments/test_times/models', exist_ok=True)
    os.makedirs('experiments/test_times/models/FSRCNN', exist_ok=True)
    os.makedirs('experiments/test_times/models/FSRCNN/int8', exist_ok=True)

    # to tflite
    model.save(f'/tmp/FSRCNN/{name}', overwrite=True, include_optimizer=True, save_format='tf')
    convert_model(f'/tmp/FSRCNN/{name}', f'experiments/test_times/models/FSRCNN/{name}.tflite', args.scale, time=True)

    # to int8 tflite
    convert_model(f'/tmp/FSRCNN/{name}', f'experiments/test_times/models/FSRCNN/int8/{name}.tflite', args.scale, mode='int8', time=True)


if __name__ == '__main__':
    main()
