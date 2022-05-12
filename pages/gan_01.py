import streamlit as st
from models.model import load_model
from models.model import generate_fake, h_plot_2

import datetime





def write(show_token=None, sidebar={}, sh=None):

    sidebar["gan_01_bar"].write()
    st.markdown("# A simple 1D GAN example")
    st.markdown(
        """Hello user!
        """
    )
    load_model(model_name='model_gan_01')

    gen_use = st.session_state['model']['model']

    char_dict = {'sinus': 1, 'para': 2, 'exp': 3}
    batch_size = 1
    signal_length=1000
    style_enc = [char_dict['para'], 1]
    fake_xy, fake_label, fake_char, fake_latent = generate_fake(model=gen_use,
                                                                batch_size=batch_size,
                                                                signal_length=signal_length,
                                                                style_enc=style_enc)

    if st.button('Generate'):
        h_plot_2(fake=fake_xy,
                 latent=fake_latent,
                 i_batch=0,
                 signal_length=signal_length,
                 title="test")
    else:
        pass


if __name__ == "__main__":
    write()
