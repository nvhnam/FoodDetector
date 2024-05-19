import streamlit as st
import time

# from pages.sidebar import SideBar
from utils import detect_image, detect_video, detect_webcam, load_model

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":pizza:"
)

with st.sidebar:
        st.header("Adjust the confident score :triangular_flag_on_post:")
        confidence = float(st.slider(
            label="",label_visibility="collapsed", min_value=30, max_value=100, value=50 
        ))/ 100
        st.subheader("Quick note :memo:")
        st.markdown("Consideration for selecting the best suited confident score:")
        st.markdown('''* **High confident score (>= 50%)**:  
                    Set a higher threshold will make the model to predict with a higher accuracy detection but it will have a :green[low recall] as fewer object will be detected because of the :green[high precision] constraint.''')
        st.markdown('''* **Low confident score (< 50%)**:  
                    Set a lower threshold will enable the model to detect more object - :green[high recall] because of the :green[low precision] constraint.''')
        st.divider()
        st.caption("Made by :blue[@nvhnam01]")

with st.container():
    st.title("Welcome to _:green[FoodDetector]_ :pizza:")
    st.divider()

    st.markdown('''FoodDetector use the _YOLOv8_ pretrained model for the hyperparameters fine-tuning with the new food dataset which was collected and annotated.  
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