import tensorflow as tf
from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def get_device(idx=0):
  #return tf.device('/gpu:%d' % idx)
  #return tf.device('/cpu:%d' % idx)
  return nullcontext()
  
