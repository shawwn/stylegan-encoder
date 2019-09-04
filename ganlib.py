import tensorflow as tf
import pickle
import dnnlib.tflib as tflib
import dnnlib
from training.networks_stylegan import G_style, D_basic
from vgg16_zhang_perceptual import lpips_network

def load_model(
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', # karras2019stylegan-ffhq-1024x1024.pkl
    session = None,
    cache_dir = 'cache'):
  session = session or tf.get_default_session()
  with session.as_default():
    with dnnlib.util.open_url(url, cache_dir='cache') as f:
      _G, _D, _Gs = pickle.load(f)
    G = tflib.Network(_G.name, G_style, **_G.static_kwargs)
    G.copy_vars_from(_G)
    D = tflib.Network(_D.name, D_basic, **_D.static_kwargs)
    D.copy_vars_from(_D)
    Gs = tflib.Network(_Gs.name, G_style, **_Gs.static_kwargs)
    Gs.copy_vars_from(_Gs)
    return G, D, Gs

def load_perceptual(
    url = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', # vgg16_zhang_perceptual.pkl
    session = None,
    cache_dir = 'cache'):
  session = session or tf.get_default_session()
  with dnnlib.util.open_url(url, cache_dir='cache') as f:
    _P = pickle.load(f)
  with session.as_default():
    P = tflib.Network(_P.name, lpips_network, **_P.static_kwargs)
    P.copy_vars_from(_P)
    return P

