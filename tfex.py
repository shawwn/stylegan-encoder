import tensorflow as tf
from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

_devices = None
_has_gpu = False
_has_tpu = False
_cores = None

def init():
  global _devices
  global _cores
  global _has_gpu
  global _has_tpu
  if _devices is None:
    _devices = tf.get_default_session().list_devices()
    _has_gpu = len([x.name for x in _devices if ':GPU' in x.name]) > 0
    _has_tpu = len([x.name for x in _devices if ':TPU' in x.name]) > 0
  if _cores is None:
    if _has_tpu:
      _cores = [x.name for x in _devices[2:10]]
    elif _has_gpu:
      _cores = [x.name for x in _devices if ':GPU' in x.name]
  return _has_gpu

def get_cores():
  init()
  return _cores

def has_gpu():
  init()
  return _has_gpu

def has_tpu():
  init()
  return _has_tpu

def device(name=''):
  if name is None:
    return tf.device(None)
  if 'gpu' in name:
    if has_gpu():
      return tf.device(name)
    if has_tpu():
      *start, idx = name.split(':')
      idx = int(idx)
      return tf.device(_cores[idx])
  if 'cpu' in name:
    return tf.device(name)
  return nullcontext()

