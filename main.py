import streamlit as st
import time

from utils import detect_image, detect_video, detect_webcam, download_predicted_image, load_model

with st.sidebar:
    st.title(":hamburger: Menu")
    st.divider()
    st.page_link("pages/dataset.py", label="Dataset", icon="📦")
    st.page_link("pages/dataset.py", label="About", icon="📃")
    st.page_link("pages/dataset.py", label="Record", icon="🖇️")
    st.divider()

    confidence = float(st.slider(
        "Model Confidence Score: ", 30, 100, 50 
    ))/ 100

with st.container():
    st.title("Welcome to _:green[FoodDetector]_ :pizza:")
    st.divider()

    st.markdown('''FoodDetector use the _YOLOv8_ pretrained model for fine-tuning with the new food dataset which was collected and annotated.  
                It can be used to detect food from a picture, video, webcam and an IP camera.''')
    
    model = load_model()

    tab1, tab2, tab3, tab4 = st.tabs(["Image", "Video", "Webcam", "IP Camera"])

    with tab1:
        st.header("Image Upload :frame_with_picture:")
        uploaded_file = st.file_uploader("Choose a picture", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            detect_image(confidence, model, uploaded_file)
            
        st.header("Take a picture now :camera_with_flash:")
        show_section = st.checkbox(":point_left: Toggle to open the camera")
        if show_section:
            picture = st.camera_input("")
            if picture:
                detect_image(confidence, model, picture)
        

    with tab2:
        st.header("Video Upload :movie_camera:")
        uploaded_clip = st.file_uploader("Choose a clip", accept_multiple_files=False, type=['mp4'])

        if uploaded_clip:
            detect_video(confidence, model, uploaded_clip)
        
    with tab3:
        st.header("Webcam :camera:")
        show_section = st.checkbox(":point_left: Toggle to open the webcam")
        if show_section:
            detect_webcam(confidence, model)

    with tab4:
        st.header("IP Camera :video_camera:")
        st.text("Enter your Camera (RTSP) IP address: ")
        col1, col2 = st.columns([1, 4])
        with col1: 
            st.text("rtsp://admin:")
        with col2:
            with st.form("ip_camera_form"):
                # rtsp://admin:N@m20011!@192.168.50.15:554/Streaming/channels/101
                address = st.text_input("", label_visibility="collapsed", placeholder="hd543211@192.168.14.106:554/Streaming/channels/101")
                col1, col2 = st.columns([3, 0.8])
                with col1:
                    submitted = st.form_submit_button("Connect")
                with col2:
                    cancel = st.form_submit_button("Disconnect")
                if submitted:
                    if address:
                        with st.spinner("Loading..."):
                            time.sleep(2)
                        detect_webcam(confidence, model, address=address, rtsp=True)
                    else:
                        st.error("Please enter a valid RTSP camera URL")
                if cancel:
                    if address:
                        detect_webcam(confidence, model, address)