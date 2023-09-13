import streamlit as st
import matplotlib.pyplot as plt



def page_project_hypothesis_body():
    st.write("### Hypothesis 1 and validation")

    st.success(
        f"Infected leaves have clear marks differentiating them from the healthy leaves."
    )
    st.info(
        f"We suspect cherry leaves affected by powdery mildew have clear marks, "
        f"An Image Montage shows that typically a mildew leaves  has white_like subtance  on the surface."
    )
    st.write("To visualize a thorough investigation of infected and healthy leaves visit the Leaves Visualiser tab.")
     
    st.warning(
        f"The model was able to detect such differences and learn how to differentiate and generalize in order to make accurate predictions."
        f" A good model trains its ability to predict classes on a batch of data without adhering too closely to that set of data."
        f" In this way the model is able to generalize and predict future observation reliably because it didn't 'memorize' the relationships between features and labels"
        f" as seen in the training dataset but the general pattern from feature to labels. "
    )


    st.write("### Hypotesis 2 and validation")

    st.success(
        f" Here from tensorflow.keras.callbacks import EarlyStopping for the training of the Model. "
    )
    st.info(
        f"The EarlyStopping callback is used to monitor the validation loss during training."
        f" If the validation loss does not improve for three consecutive epochs the training will"
        f"stopped early to prevent overfitting and save time"
        f" The learning curve shows the accuracy and error rate on the training and validation dataset while the model is training.\n\n"
        f"The learning model stop after the 12th Epochs"
    )
    st.warning(
        f"In this case the ```EarlyStopping``` function performed better. "
    )
    
    # file paths to your images within the "streamlit_images" folder
    aculate_plt_path = "stream_images/acuulat_plt.png"
    loss_plt_path = "stream_images/plt.png"

    # Display the images using st.image()
    st.image(aculate_plt_path, caption="Aculate Plot")
    st.image(loss_plt_path, caption="Loss Plot")

    

    

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/ujuadimora-dev/mildew-detection-in-cherry-leaves/blob/main/README.md).")