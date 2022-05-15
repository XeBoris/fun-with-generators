import streamlit as st
from models.model import load_lite_model
from models.helper_gan_01_v0 import generate_real, generate_fake_lite
from models.helper_gan_01_v0 import plot_xy
import datetime





def write(show_token=None, sidebar={}, sh=None):

    sidebar["gan_01_bar"].write()
    st.markdown("# A simple 1D GAN example")
    st.markdown(
        """Hello user!
        """
    )
    # load_model(model_name='model_gan_01')
    load_lite_model(model_name='gan_01_v1')

    model_lite = st.session_state['model']['model']

    # This page is uses some specific dictionary content in the
    # session_stage object to avoid loss during reload. You need to
    # setup model/store upfront with the right keywords:
    if 'histo_diff' not in st.session_state['model']['store']:
        st.session_state['model']['store']['histo_diff'] = []

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.button('Generate')

    with col2:
        if st.button('Reset') or len(st.session_state['model']['store']['histo_diff']) > 50:
            st.session_state['model']['store']['histo_diff'] = []
        else:
            pass


    char_dict = {'sinus': 1, 'para': 2, 'exp': 3}
    batch_size = 1
    signal_length = 200
    style_enc = [char_dict['para'], 1]
    fake_xy, fake_label, fake_char, fake_latent = generate_fake_lite(model=model_lite,
                                                                batch_size=batch_size,
                                                                signal_length=signal_length,
                                                                style_enc=style_enc)

    # st.write(fake_xy)

    x_ = fake_xy.T[0]
    real_xy, real_label, real_style = generate_real(batch_size=batch_size,
                                                    signal_length=signal_length,
                                                    style_enc=style_enc,
                                                    use_linspace=x_)
    plot_xy(fake_xy, real_xy)
    # plot_xy(real_xy)




if __name__ == "__main__":
    write()
