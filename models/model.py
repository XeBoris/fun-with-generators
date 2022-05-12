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
    else:
        pass


# We need a function which creates the lantent / noise vector for us.
def generate_latent(batch_size=1,
                    latent_size=10,
                    dim=2):
    """
    Generate a latent (noise) vector as input to generator. This
    latent vector does not include the style! Style is defined
    outside this function.
    """
    nm = np.random.normal(0, 1, (batch_size*latent_size, dim))

    return nm

def generate_fake(model=None,
                  batch_size=1,
                  signal_length=10,
                  style_enc=[{}]):
    """
    We generate a fake signal from a generator model
    """

    # We need a latent vector:
    z = generate_latent(batch_size=batch_size,
                        latent_size=signal_length,
                        dim=2)

    # The generated signal is always known as false, here
    # comes the label:
    label = np.zeros((batch_size*signal_length, 1))


    # Encode the style from a <style_length> x 1 vector to a
    # <signal_length> x <style_length> vector
    char = np.full((batch_size*signal_length, len(style_enc)), style_enc)

    #print(z.shape)
    #print(char.shape)
    z_prime = np.hstack((z, char))

    #print(z)
    #print(char.T)
    if model is None:
        return _, label, char, z
    else:
        m = model(z_prime)
        return m, label, char, z

def h_plot_2(fake, latent, i_batch, signal_length, title="no title"):
    total_length = len(fake)
    batch_size = int(total_length/signal_length)

    jb = i_batch*signal_length
    je = (i_batch+1)*signal_length

    fig, ax = init_plot_a(figsize=(8,5), col=1, row=1)
    ax[0].scatter(latent[jb:je, 0], latent[jb:je, 1],
                  color="green", s=5,
                  label="Noise")
    ax[0].set_title("Latent vector")
    ax[0].set_xlabel("x-noise")
    ax[0].set_ylabel("y-noise")

    fig, ax = init_plot_a(figsize=(8,5), col=1, row=1)
    ax[0].set_title(f"{title}: Batch {i_batch}/{batch_size} - Lenght: {signal_length}")
    # ax[0].scatter(signal[jb:je, 0], signal[jb:je, 1],
    #               color="green", s=5,
    #               label="Original")
    ax[0].scatter(fake[jb:je, 0], fake[jb:je, 1],
                  color="red", s=2,
                  label="GAN")

    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend()
    st.write(fig)

def init_plot_a(  title="title",
                  x_title="xtitle",
                  y_title="ytitle",
                  figsize=(8,8),
                  row = 1,
                  col = 1,
                  ):

    nb_total = row*col
    #    ax = fig.add_subplot(422)
    fig = plt.figure(figsize=figsize)
    ax = []
    for i in range(nb_total):
        cnt = "{0}{1}{2}".format(str(nb_total), str(col), str(i+1) )
        print(cnt)
        ax.append(fig.add_subplot(int(cnt)))
        ax[i].set_title(title)
        ax[i].set_xlabel(x_title)
        ax[i].set_ylabel(y_title)
    #plt.show()
    return fig, ax