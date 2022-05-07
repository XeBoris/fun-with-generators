import streamlit as st

import datetime

def write(show_token=None, sidebar={}, sh=None):

    sidebar["basic"].write()
    st.markdown("# Hello!")
    st.markdown(
        """Hello user!
        """
    )
    delta_time = datetime.datetime.now() - st.session_state['meta']['load_date']

    st.write(f"This app is up since {delta_time.seconds} seconds.")


if __name__ == "__main__":
    write()
