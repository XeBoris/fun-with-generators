import streamlit as st
import os
import sys
import pathlib
import argparse
import pandas as pd
import numpy as np
import logging
import datetime


import streamlit_authenticator as stauth

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
import sitebars.basic
import sitebars.site_page1

PAGES = {
    "Home": pages.home,
    "Page 1": pages.page1
}

SIDEBARS = {
    "basic": sitebars.basic,
    "site_page1": sitebars.site_page1
}



if 'meta' not in st.session_state:
    st.session_state['meta'] = {}

if 'load_date' not in st.session_state['meta']:
    st.session_state['meta']['load_date'] = datetime.datetime.now()


def main():

    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page.write(sidebar=SIDEBARS)


if __name__ == "__main__":
    main()
