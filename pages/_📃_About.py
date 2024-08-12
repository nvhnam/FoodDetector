import streamlit as st

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":microscope:"
)

st.title(":bookmark_tabs: About FoodDetector")
st.divider()
st.markdown('''_A one-stop website for anyone that want to remotely track the eating activity of their family member._
##### FoodDetector will have 3 main sections as follows:
            
- `main`: The homepage of this website for uploading media files to perform inferencing and download its results in real-time.           
- `About`: The overview of FoodDetector sections and functions.
- `Dataset`: Details of the VietFood57 dataset.''')
st.markdown('''
The source code of this project will be available on [Github](https://github.com/nvhnam/FoodDetector).

---
#### Functions
            
FoodDetector can inferencing on these 4 main sources of media:

- **Image**: uploading image file (suffix `.jpg`, `.png`,...) from the user local machine or the image url that being hosted online can also be used. After the prediction process, 2 buttons will be shown for download the prediction results as an images file with bounding boxes or a CSV file that records the detected dishes. The results file will be generated when user clicked the button and both will have the name format of `"%date-%month-%year_%hour-%minute".jgp/csv`.
- **Video**: uploading video file (suffix `.mp4`, `.mpeg4`,...) from the user local machine. Besides, ___Youtube___ video or shorts URL link are also supported for prediction on the fly. Unlike images, the results file of this part will only be a csv file that record all the dishes that appeared accross all frames.
- **Webcam**: as this project is deployed on [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud) so, to open user local webcam then [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) was used for handling the connection and choosing the webcam input. However, no result files will be generated as this process can be up running for a while.
- **IP camera**: a _RTSP_ address of the user camera must be provided and that camera must be configured beforehand to be able to receive connection from outside network. 
''')