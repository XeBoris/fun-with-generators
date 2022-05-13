import streamlit as st
import os
import sys
import pathlib
import argparse
import pandas as pd
import numpy as np
import logging
import datetime


sys.path.append(str(pathlib.Path().absolute()).split("/model")[0] + "/model")


# pre-init session states:
#We safe the argparse object in the session_state for later use
st.set_page_config(
    page_title="Test-Me-App",
    page_icon=None,
    # layout="centered",
    layout="wide"
    #initial_sidebar_state="collapsed",
)

import pages.home
import pages.page1
import pages.gan_01
import sitebars.basic
import sitebars.site_page1
import sitebars.gan_01_bar

PAGES = {
    "Home": pages.home,
    "Simple 1D GAN": pages.gan_01,
    # "Page 1": pages.page1
}

SIDEBARS = {
    "basic": sitebars.basic,
    "gan_01_bar": sitebars.gan_01_bar,
    "site_page1": sitebars.site_page1
}



if 'meta' not in st.session_state:
    st.session_state['meta'] = {}

if 'load_date' not in st.session_state['meta']:
    st.session_state['meta']['load_date'] = datetime.datetime.now()

if 'model' not in st.session_state:
    st.session_state['model'] = {}
    st.session_state['model']['name'] = None
    st.session_state['model']['model'] = None
    st.session_state['model']['store'] = {}


def main():

    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page.write(sidebar=SIDEBARS)


if __name__ == "__main__":
    main()
