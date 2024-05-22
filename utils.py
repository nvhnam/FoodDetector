from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile

import numpy as np
from io import BytesIO

import csv
import re
import requests
import datetime
import os
import io
import base64
import plotly.graph_objects as go

def create_fig(image, detected=False):

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data_uri = base64.b64encode(buffer.getvalue()).decode()
    
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{image_data_uri}",
            x=0,
            y=image.size[1],
            xref="x",
            yref="y",
            sizex=image.size[0],
            sizey=image.size[1],
            layer="below"
        )
    )
    
    fig.update_layout(
        xaxis_range=[0, image.size[0]],
        yaxis_range=[0, image.size[1]],
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                showarrow=False,
                text="Detected Image" if detected else "Original Image",
                xref="paper",
                yref="paper"
            )
        ]
    )
    
    return fig

def extract_shorts_id(url):
    is_youtube_short =  re.search(r'(?:youtube\.com/shorts/|youtu\.be/)([a-zA-Z0-9_-]+)', url)
    if is_youtube_short:
        return is_youtube_short.group(1)
    return None

def _display_detected_frames(conf, model, st_frame, image, youtube_url, rtsp=False):
    if rtsp:
        res = model(source=image, conf=conf, imgsz=640, save=False, device="cpu", stream=True)
    else:
        youtube_id = extract_shorts_id(url=youtube_url)
        if youtube_id and youtube_url:
            valid_url = f"https://www.youtube.com/watch?v={youtube_id}"
            res = model(source=valid_url, stream=True, conf=conf, imgsz=640, save=True, device="cpu")
            for i, r in enumerate(res):
                im_bgr = r.plot() 
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  
                im_rgb_resized = im_rgb.resize((640, 640))          
            st_frame.image(im_rgb_resized, caption='Predicted Video', use_column_width=True)  
        else: 
            res = model(source=image, stream_buffer=True, stream=True, vid_stride=7280, conf=conf, imgsz=640, save=True, device="cpu")
            current_time = datetime.datetime.now()
            time_format = current_time.strftime("%d-%m-%Y_%H-%M")
            for i, r in enumerate(res):
                im_bgr = r.plot() 
                im_rgb = Image.fromarray(im_bgr[..., ::-1]) 
                im_rgb_resized = im_rgb.resize((640,640), reducing_gap=1.0)
                r.save(filename=f"runs/detect/videos/{time_format}.jpg") 
            st_frame.image(im_rgb_resized, caption='Predicted Video', use_column_width=True)
            # pass

def display_clip(uploaded_clip, confidence, model):
    if uploaded_clip:
        # youtube_id = extract_shorts_id(video_url)
        # if youtube_id:
        #     video_url = f"https://www.youtube.com/watch?v={youtube_id}"
        # print(video_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_clip.read())
            temp_file_path = temp_file.name
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            st.error("Error opening video stream or file")
            return
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  
        # cap.set(cv2.CAP_PROP_FPS, 60)
        try: 
            st_frame = st.empty()
            while True:
                success, frame = cap.read()
                if success:             
                    _display_detected_frames(confidence, model, st_frame, frame, youtube_url="")
                else:
                    break
            cap.release()
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")

@st.cache_resource
def load_model():
    # modelpath = r"./model/YOLOv8m_1.pt"
    # modelpath = r"./model/YOLOv8s_1_new_VN_SGD_tuned.pt"
    modelpath = r"./model/YOLOv8s_2_new_VN_SGD_YOLO_tuned.pt"
    model = YOLO(modelpath)

    return model

def resize_image(image):
    return image.resize((640, 640))

def detect_image_result(detected_image, model):
    boxes = detected_image[0].boxes

    if boxes:
        detected_img_arr_RGB = detected_image[0].plot()[:, :, ::1]  
        detected_img_arr_BGR = detected_image[0].plot()[:, :, ::-1]  

        fig_detected = create_fig(detected_img_arr_BGR, detected=True)
        st.plotly_chart(fig_detected, use_container_width=True)

        current_time = datetime.datetime.now()
        time_format = current_time.strftime("%d-%m-%Y_%H-%M")
        
        result_filename = os.path.join("runs/detect/images/", f"{time_format}.jpg")
        cv2.imwrite(result_filename, detected_img_arr_RGB)
        with open(result_filename, 'rb') as file:
            byte_im = file.read()

        col1, col2 = st.columns([0.7, 0.3], gap="medium")
        with col1:
            st.markdown("**Predicted Image**")
        with col2:
            st.download_button(label="Download the image",
                                data=byte_im,
                                mime="image/jpg",
                                file_name=f"{time_format}.jpg")
            
        detection_results = ""
        count_dict = {}

        food_names = []
        confidences = []
        counts = []

        for box in boxes:
            class_id = model.names[box.cls[0].item()]
            food_names.append(class_id)
            conf = int(round(box.conf[0].item(), 2)*100)
            confidences.append(conf)

            detection_results += f"<b style='color: cyan;'>Food name:</b> {class_id}<br><b style='color: cyan;'>Confidence:</b> {conf}%<br>---<br>"
            if class_id in count_dict:
                count_dict[class_id] += 1
            else:
                count_dict[class_id] = 1
        for object_type, count in count_dict.items():
            counts.append(count)
            detection_results += f"<b style='color: cyan;'>Count of {object_type}:</b> {count}<br>"

        scrollable_textbox = f"""
            <div style="
                font-family: 'Source Code Pro','monospace';
                font-size: 16px;
                overflow-y: scroll;
                padding: 10px;
                width: 400px;
                height: 400px;
            ">
                {detection_results}
            </div>
        """

        st.markdown("""### Results:""")
        st.markdown(scrollable_textbox, unsafe_allow_html=True)

        rows = zip(food_names, confidences, counts)

        csv_filename = os.path.join("runs/detect/texts/", f"{time_format}.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Food Name(s)", "Confidence(%)", "Count"])
            writer.writerows(rows)

    else:
        st.markdown("""### No food detected""")
        st.markdown("""
            The model did not detect any foods in the uploaded image.  
            Please try with a different image or adjust the model's 
            confidence threshold in the sidebar and try again.
        """)


def detect_image(conf, model, uploaded_file, url=False):
    if uploaded_file is not None:
        if url==False:
            uploaded_image = Image.open(uploaded_file)
        else:
            response = requests.get(uploaded_file)
            response.raise_for_status()
            uploaded_image = Image.open(BytesIO(response.content))

        resized_uploaded_image = resize_image(uploaded_image)
        st.image(resized_uploaded_image, output_format="JPEG", use_column_width=False)

        col1, col2 = st.columns([0.8, 0.2], gap="large")
        with col1:
            st.markdown("**Original Image**")
        with col2:
            if url==False:
                predict_button = st.button("Predict", use_container_width=True, type="primary")
        if url==False and predict_button:
            with st.spinner("Running..."):
                detected_image = model(resized_uploaded_image, conf=conf, imgsz=640)
                detect_image_result(detected_image, model)
        elif uploaded_file and url:
            with st.spinner("Running..."):
                detected_image = model(resized_uploaded_image, conf=conf, imgsz=640)
                detect_image_result(detected_image, model)

def detect_video(conf, model, uploaded_file):
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        execution_button = st.button("Predict", key="predict_button")

        if 'execution_completed' not in st.session_state:
            st.session_state['execution_completed'] = False

        if 'play_button_clicked' not in st.session_state:
            st.session_state['play_button_clicked'] = False

        st_frame = st.empty()  
        temp_output_file = None  

        if execution_button:  
            with st.spinner("FoodDetector is predicting..."):
                temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_cap = cv2.VideoCapture(tfile.name)
                frame_width, frame_height = int(vid_cap.get(3)), int(vid_cap.get(4))
                out = cv2.VideoWriter(temp_output_file.name, fourcc, vid_cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

                progress_bar = st.progress(0)
                    
                frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
                st.info(f"This video frame rate: {frame_rate} FPS")  

                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

                for frame_num in range(total_frames):
                    success, image = vid_cap.read()
                    if success:
                        if frame_num % 15 == 0:
                            processed_frame = _display_detected_frames(conf, model, st_frame, image, youtube_url="")
                            out.write(processed_frame)
                        progress_bar.progress((frame_num + 1) / total_frames)
                    else:
                        break

                processed_frames_info = f"Total Frames: {total_frames}"
                st.info(processed_frames_info) 

                out.release()  
                vid_cap.release()  

                st.success('Execution complete!')
                st.session_state['execution_completed'] = True

                

def detect_webcam(conf, model, address="", rtsp=False):
    if rtsp:
        vid_cap = cv2.VideoCapture('rtsp://admin:' + address)
        # vid_cap = cv2.VideoCapture('http://:' + address)
    else: 
        vid_cap = cv2.VideoCapture(0)
    
    vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  
    vid_cap.set(cv2.CAP_PROP_FPS, 4)

    try: 
        st_frame = st.empty()
        while True:
            success, image = vid_cap.read()
            if success:
                mirrored_frame = cv2.flip(image, 1)            
                _display_detected_frames(conf, model, st_frame, mirrored_frame, rtsp=rtsp, youtube_url="")
            else:
                break
        # vid_cap.release()
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
    finally:
        vid_cap.release()
