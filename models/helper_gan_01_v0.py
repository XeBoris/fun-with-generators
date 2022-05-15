import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

#We need a real signal. Here we go for a function which generates them for us.
def generate_real(batch_size=1,
                  signal_length=10,
                  style_enc=[{}],
                  use_linspace="rand"):
    """
    We need a signal generator

    :param batch_size: int
        defines how many batches of the same signal type is produced
    :param signal_length: int
        defines the signal length
    :param style_enc: list
        encodes a certain style of the generated signals

    """
    if isinstance(use_linspace, str) and use_linspace == "linear":
        ls_beg = -10
        ls_end = 10
        ls_bins = signal_length
        X1 = []
        for i in range(batch_size):
            il = np.linspace(start=ls_beg, stop=ls_end, num=ls_bins)
            X1.append(il)
        X1 = np.array(X1)
    elif isinstance(use_linspace, str) and use_linspace == "rand":
        X1 = np.random.uniform(low=-10,
                               high=10,
                               size=(batch_size, signal_length))
    else:
        X1 = use_linspace

    if style_enc[0] == 1:
        Y = np.sin(X1)
    elif style_enc[0] == 2:
        Y = X1*X1
    elif style_enc[0] == 3:
        Y = np.exp(X1)

    X1 = X1.reshape(batch_size*signal_length,1)
    Y = Y.reshape(batch_size*signal_length, 1)
    xy = np.hstack((X1,Y))

    # The generated signal here is of label 1. We 'know' that those
    # x/y coordinates are true.
    label = np.ones((batch_size*signal_length, 1))

    # Encode the style from a <style_length> x 1 vector to a
    # <signal_length> x <style_length> vector
    char = np.full((batch_size*signal_length, len(style_enc)), style_enc)

    return xy, label, char

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

    z_prime = np.hstack((z, char))

    if model is None:
        return _, label, char, z
    else:
        m = model(z_prime)
        return m, label, char, z

def generate_fake_lite(model=None,
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

    z_prime = np.hstack((z, char))

    if model is None:
        return [], label, char, z
    else:
        gen, gen_in, gen_out = model
        gen.resize_tensor_input(gen_in[0]["index"], [int(batch_size*signal_length), 4])
        gen.allocate_tensors()

        z_prime = z_prime.astype(np.float32)

        gen.set_tensor(gen_in[0]['index'], z_prime)
        gen.invoke()
        r = gen.get_tensor(gen_out[0]['index'])
        r = np.squeeze(r)

        return r, label, char, z


def plot_xy(fake=np.zeros((1,1)),
            signal=np.zeros((2,2))
            ):
    try:
        x_fake = fake.numpy().T[0]
        y_fake = fake.numpy().T[1]
    except:
        x_fake = fake.T[0]
        y_fake = fake.T[1]
    try:
        x_sig = signal.numpy().T[0]
        y_sig = signal.numpy().T[1]
    except:
        x_sig = signal.T[0]
        y_sig = signal.T[1]


    fig = make_subplots(rows=1, cols=2)
    s1 = go.Scatter(x=x_fake,
                    y=y_fake,
                    mode="markers",
                    name="GAN (1D)",
                    )
    fig.add_trace(s1, row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)

    if len(x_sig) > 1:
        s2 = go.Scatter(x=x_sig,
                        y=y_sig,
                        mode="markers",
                        name="parable",
                        )
        fig.add_trace(s2, row=1, col=1)

        diff = np.abs((y_sig - y_fake)/y_sig)
        s3 = go.Scatter(x=x_sig,
                        y=diff,
                        mode="markers",
                        name="(Y_sig - Y_gan)/y_sig",
                        )
        fig.add_trace(s3, row=1, col=2)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="residual", row=1, col=2, type="log")

        if 'histo_diff' not in st.session_state['model']['store']:
            st.session_state['model']['store']['histo_diff'] = []
            st.session_state['model']['store']['histo_diff'].append(diff)
        else:
            st.session_state['model']['store']['histo_diff'].append(diff)

    fig.update_layout(
        width=1024,
        height=450,
        xaxis_title="x",
        yaxis_title="y",
        title={
            'text': "Compare an analytical parable with generated one:",
        }
    )
    st.write(fig)

    kk = st.session_state['model']['store']['histo_diff']
    kk = np.array(kk)
    # st.write(.shape)
    kk = kk.T
    kksum = np.sum(kk, axis=1)


    if len(kk)>0:
        fig = make_subplots(rows=1, cols=2)
        s1 = go.Scatter(x=x_fake,
                        y=kksum/len(kk.T),
                        mode="markers",
                        name="sum",
                        )
        fig.add_trace(s1, row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text=f"SUM(residual)/{len(kk.T)}", row=1, col=1, type="log")

        s2 = go.Histogram(x=kksum/len(kk.T),
                          #histnorm='probability',
                          # xbins=dict( # bins used for histogram
                          #     start=0,
                          #     end=1e6,
                          #     size=1000
                          # ),
                          )
        fig.add_trace(s2, row=1, col=2)
        fig.update_xaxes(title_text=f"SUM(residual)/{len(kk.T)}",
                         # type="log",
                         row=1, col=2)
        fig.update_yaxes(title_text="Counts",
                         type="log",
                         row=1, col=2)


        fig.update_layout(
            width=1024,
            height=450,
            title={
                'text': f"Summary of {len(kk.T)} generated parables from the GAN 1D",
            }
        )
        st.write(fig)