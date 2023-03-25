''' convert '''
from functools import partial
import numpy as np
import tqdm
import cv2
import tensorflow as tf


def representative_dataset_gen(scale: int, input_channel: int = 3):
    ''' representative dataset gen func '''
    for i in tqdm.tqdm(range(801, 901), desc='rep_dataset'):
        lr_path = f'datasets/DIV2K/valid/LR/bicubic/X{scale}/original/{i:04d}x{scale}.png'
        lr = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
        lr = lr.astype(np.float32)
        lr = np.expand_dims(lr, 0)
        yield [lr[:, :, :, :input_channel]]


def representative_dataset_gen_time(scale: int, input_channel: int = 3):
    ''' representative dataset gen func for time '''
    lr_path = f'datasets/DIV2K/valid/LR/bicubic/X{scale}/original/0801x{scale}.png'
    lr = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
    lr = lr.astype(np.float32)
    lr = np.expand_dims(lr, 0)
    yield [lr[:, 0:360, 0:640, :input_channel]]


def convert_model(model_path, tflite_path, scale: int, input_channel: int = 3, mode: str = 'fp32', time=False):
    ''' convert model fp32 '''
    assert mode in ['fp32', 'fp16', 'int8']
    model = tf.saved_model.load(model_path)
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, 360, 640, input_channel] if time else [1, None, None, input_channel])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter=True
    if mode == 'int8':
        converter.experimental_new_quantizer=True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = partial(representative_dataset_gen_time if time else representative_dataset_gen, scale, input_channel=input_channel)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    if mode == 'fp16':
        converter.experimental_new_quantizer=True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
