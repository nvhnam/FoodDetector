import streamlit as st
import time

from utils import _display_detected_frame, detect_camera, detect_image, detect_video, detect_webcam, load_onnx_model, load_model

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":microscope:"
)

with st.sidebar:
        st.header("Adjust the confident score :triangular_flag_on_post:")
        confidence = float(st.slider(
            label="",label_visibility="collapsed", min_value=10, max_value=100, value=50 
        ))/ 100
        st.subheader("Quick note :memo:")
        st.markdown("Consideration for selecting the best suited confident score:")
        st.markdown('''* **High confident score (>= 50%)**:  
                    Set a higher threshold will make the model to predict with a higher accuracy detection but it will have a :green[low recall] as fewer object will be detected because of the :green[high precision] constraint.''')
        st.markdown('''* **Low confident score (< 50%)**:  
                    Set a lower threshold will enable the model to detect more object - :green[high recall] because of the :green[low precision] constraint.''')
        st.divider()
        st.markdown('''Made by [@nvhnam](https://github.com/nvhnam)''')

with st.container():
    st.title("Welcome to _:green[FoodDetector]_ :male-detective:")
    st.divider()

    st.markdown('''
FoodDetector uses the _YOLOv10m_ pretrained models for fine-tuning with `VietFood57`, a new custom-made Vietnamese food dataset created for detecting local dishes and achieved a `mAP50` of `0.934`.  
It can be used to detect <a href="/Dataset" target="_blank" style="color: #4CAF50; font-weight: bold; font-style: italic; text-decoration: none;">`57`</a> Vietnamese dishes from a picture, video, webcam, and an IP camera through RTSP.
''', unsafe_allow_html=True)

    model1 = load_model()
    model = load_onnx_model()

    tab1, tab2, tab3, tab4 = st.tabs(["Image", "Video", "Webcam", "IP Camera"])

    with tab1:
        st.header("Image Upload :frame_with_picture:")
        uploaded_file = st.file_uploader("Choose a picture", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            detect_image(confidence, model=model1, uploaded_file=uploaded_file)
            
            # detections = detect_image_onnx(model, uploaded_file, confidence)

        st.subheader("Enter a picture URL 	:link:")
        with st.form("picture_form"):
            col1, col2 = st.columns([0.8, 0.2], gap="medium")
            with col1:
                picture_url = st.text_input("", label_visibility="collapsed", placeholder="https://ultralytics.com/images/bus.jpg")
            with col2:
                submitted = st.form_submit_button("Predict", use_container_width=True)
        if submitted and picture_url:
            detect_image(confidence, model=model1, uploaded_file=picture_url, url=True)            

    with tab2:
        st.header("Video Upload :movie_camera:")
        uploaded_clip = st.file_uploader("Choose a clip", accept_multiple_files=False, type=['mp4'])
        if uploaded_clip:
            detect_video(conf=confidence, uploaded_file=uploaded_clip)

        else: 
            st.subheader("Enter YouTube URL :tv:")
            tube = st.empty()
            with st.form("youtube_form"):
                col1, col2 = st.columns([0.8, 0.2], gap="medium")
                with col1:
                    youtube_url = st.text_input("", label_visibility="collapsed", placeholder="https://youtu.be/LNwODJXcvt4")
                with col2:
                    submitted = st.form_submit_button("Predict", use_container_width=True)
            if submitted and youtube_url:            
                _display_detected_frame(conf=confidence, model=model1, 
                                        st_frame=tube,
                                        youtube_url=youtube_url)

    with tab3:
        st.header("Webcam :camera:")
        detect_webcam(confidence, model=model1)

    with tab4:
        st.header("IP Camera :video_camera:")
        st.text("Enter your Camera (RTSP) address: ")
        col1, col2 = st.columns([1, 4])
        with col1: 
            st.text("rtsp://admin:")
        with col2:
            with st.form("ip_camera_form"):
                address = st.text_input("", label_visibility="collapsed", placeholder="hd543211@192.168.14.106:554/Streaming/channels/101")
                col1, col2 = st.columns([3, 0.8])
                with col1:
                    submitted = st.form_submit_button("Connect")
                with col2:
                    cancel = st.form_submit_button("Disconnect")
                if submitted:
                    if address:
                        detect_camera(confidence, model1, address=address)
                    else:
                        st.error("Please enter a valid RTSP camera URL")
                if cancel:
                    if address:
                        detect_camera(confidence, model1, address="")
                        st.toast("Disconnected", icon="✅")