import numpy as np
import collections
import tensorflow as tf
import pickle
import dnnlib.tflib as tflib
import dnnlib
from tqdm import tqdm

tflib.init_tf()
sess = tf.get_default_session()    

url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
#f = open('karras2019stylegan-ffhq-1024x1024.pkl', 'rb')
f = dnnlib.util.open_url(url, cache_dir='cache')
_G, _D, _Gs = pickle.load(f)
print('Loaded karras2019stylegan-ffhq-1024x1024.pkl')

# Load weights into memory
weights = {
    'G': {'args': _G.static_kwargs, 'vars': collections.OrderedDict()},
    'D': {'args': _D.static_kwargs, 'vars': collections.OrderedDict()},
    'Gs': {'args': _Gs.static_kwargs, 'vars': collections.OrderedDict()}
}
with sess.as_default():  
    print('Loading generator...')
    for i, (key, weight_tensor) in enumerate(tqdm(_G.vars.items())):
        weights['G']['vars'][key] = weight_tensor.eval()
    print('Loading discriminator...')
    for i, (key, weight_tensor) in enumerate(tqdm(_D.vars.items())):
        weights['D']['vars'][key] = weight_tensor.eval()
    print('Loading synthesizer...')
    for i, (key, weight_tensor) in enumerate(tqdm(_Gs.vars.items())):
        weights['Gs']['vars'][key] = weight_tensor.eval()
print(' ({}) weights found. '.format(sum([len(x) for x in weights.values()])))

pickle.dump(weights, open('weights.pkl', 'wb'))
print('Saved original StyleGAN weights to disk.')
