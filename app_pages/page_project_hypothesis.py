import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We suspect  powdery mildew leaves have clear marks/signs, "
        f"typically on the surface of the leaves, that can differentiate them from an Healthy leaf. \n\n"
        f"* An Image Montage shows that typically a mildew leaves  has white_like subtance  on the surface. "
        f"Average Image, Variability Image and Difference between Averages studies did not reveal "
        f"any clear pattern to differentiate one from another."

    )