import streamlit as st
import os.path
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as md
from matplotlib.colors import LogNorm
from matplotlib.colors import cnames

import tensorflow as tf
from tensorflow import keras

def load_model(model_name):
    # st.write(st.session_state['model'])

    modelreg = {
        'gan_01_v0':  'gan_01_v0'
    }

    if st.session_state['model']['name'] != model_name or st.session_state['model']['name'] is None:
    # if True:
        p = modelreg['gan_01_v0']
        st.session_state['model']['name'] = model_name
        spath = f'./models/{p}/'
        spath = os.path.abspath(spath)
        gen_use = keras.models.load_model(spath)
        st.session_state['model']['model'] = gen_use
        st.session_state['model']['store'] = {}
    else:
        pass


