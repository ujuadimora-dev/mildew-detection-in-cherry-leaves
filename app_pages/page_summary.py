import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n"
        f"* Powdery mildew is a fungal disease that affects many different types of plants.\n"
        f"* These white, powdery spots emerge in a circular pattern on the leaves, stems, and even the fruit.\n"
        f"* Powdery mildew often grows on the upper surface of the leaves, although it can also develop on the undersides.\n"
        f"* Read More: [Planet Natural - Powdery Mildew](https://www.planetnatural.com/pest-problem-solver/plant-disease/powdery-mildew/)\n"
        f"* The available dataset contains +4 thousand images taken from the client's crop fields."
    )

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/ujuadimora-dev/mildew-detection-in-cherry-leaves/blob/main/README.md)."
    )

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in having a study to differentiate "
        f"a cherry leaf infected with powdery mildew and a healthy leaf visually.\n"
        f"* 2 - The client is interested in telling whether a given cherry leaf is infected with powdery mildew or not."
    )
