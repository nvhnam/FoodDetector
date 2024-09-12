import av
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer, VideoTransformerBase

import numpy as np
from io import BytesIO
import queue

import time
from collections import deque

import csv
import re
import requests
import datetime
import os
import io
import base64
import plotly.graph_objects as go

from class_names import class_names

def styling_css():
    with open('./assets/css/general-style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
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

def convert_youtube_url(url):
    pattern = r"(?:https?://)?(?:www\.)?(?:youtube\.com/shorts/|youtube\.com/watch\?v=|youtu\.be/)([\w\-]{11})"
    match = re.search(pattern, url)
    
    if match:
        video_id = match.group(1)
        return f"https://youtu.be/{video_id}"
    return None


def _display_detected_frame(conf, model, st_frame, youtube_url=""):
    if youtube_url:
        youtube_id = convert_youtube_url(youtube_url)
        if youtube_id:
            valid_url = youtube_id
            st.toast("Connecting", icon="🕒")
            try:
                results = model(source=valid_url, stream=True, conf=conf, imgsz=640, save=True, device="cpu", vid_stride=5)
                displayed_dishes = set()
                total_nutrition = {
                    "Calories": 0,
                    "Fat": 0,
                    "Saturates": 0,
                    "Sugar": 0,
                    "Salt": 0
                }
                detection_results = ""
                new_detections = False
                nutrition_data = []
                current_time = datetime.datetime.now()
                time_format = current_time.strftime("%d-%m-%Y")
                
                stop_button = st.button("Stop")
                stop_pressed = False

                frame_count = 0
                start_time = time.time()
                for r in results:
                    for pred in r.boxes: 
                        class_id = int(pred.cls[0].item())
                        class_name = class_names[int(class_id)]["name"]
                        confident = int(round(pred.conf[0].item(), 2)*100)
                        serving = class_names[int(class_id)]["serving_type"]

                        if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                            detection_results += f"<b style='color: black;'>Class name:</b> {class_name}<br><b style='color: black;'>Confidence:</b> {confident}%<br>---<br>"
                            displayed_dishes.add(class_name)
                            new_detections = True
                        elif class_name not in displayed_dishes:
                            nutrition = class_names[int(class_id)]["nutrition"]
                            if nutrition:
                                displayed_dishes.add(class_name)
                                new_detections = True

                                calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                                fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                                saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                                sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                                salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                                percentage_contribution = calculate_nutrient_percentage(nutrition)

                                nutrition_str = (
                                    f"<span>Calories: {nutrition.get('Calories')} kcal ({percentage_contribution['Calories']:.1f}%) - {calories_desc}</span>, "
                                    f"<span style='color: {fat_color};'>Fat: {nutrition.get('Fat')} g ({percentage_contribution['Fat']:.1f}%) - {fat_desc}</span>, "
                                    f"<span style='color: {saturates_color};'>Saturates: {nutrition.get('Saturates')} g ({percentage_contribution['Saturates']:.1f}%) - {saturates_desc}</span>, "
                                    f"<span style='color: {sugar_color};'>Sugar: {nutrition.get('Sugar')} g ({percentage_contribution['Sugar']:.1f}%) - {sugar_desc}</span>, "
                                    f"<span style='color: {salt_color};'>Salt: {nutrition.get('Salt')} g ({percentage_contribution['Salt']:.1f}%) - {salt_desc}</span>"
                                )

                                detection_results += (
                                    f"<b style='color: black;'>Food name:</b> {class_name}<br>"
                                    f"<b style='color: black;'>Confidence:</b> {confident}%<br>"
                                    f"<b style='color: black;'>Nutrition ({serving}):</b> {nutrition_str}<br>---<br>"
                                )

                                for key in total_nutrition:
                                    if key in nutrition:
                                        total_nutrition[key] += nutrition[key]

                            nutrition_data.append((
                                class_name,
                                serving,
                                confident,
                                nutrition.get('Calories'),
                                nutrition.get('Fat'),
                                nutrition.get('Saturates'),
                                nutrition.get('Sugar'),
                                nutrition.get('Salt')
                            ))
                    im_bgr = r.plot() 
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 1.0:
                        fps = frame_count / elapsed_time
                        start_time = time.time()
                        frame_count = 0
                    cv2.putText(im_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4) 

                    im_rgb = Image.fromarray(im_bgr[..., ::-1])  
                    im_rgb_resized = im_rgb.resize((640, 640))        
                    st_frame.image(im_rgb_resized, caption='Predicted Video', use_column_width=True)

                    if stop_button:
                        stop_pressed = True
                        stop_button = None
                        break
                
                st.markdown("""### Results:""")
                if new_detections:
                    scrollable_textbox = f"""
                        <div style="
                            font-family: 'Source Code Pro','monospace';
                            font-size: 16px;
                            overflow-y: scroll;
                            padding: 10px;
                            width: auto;
                            height: auto;
                        ">
                            {detection_results}
                        </div>
                    """
                    st.markdown(scrollable_textbox, unsafe_allow_html=True)
                    
                displayed_dishes.clear()

                total_nutrition_str = (
                                f"<b style='color: black;'>Total Nutrition:</b> "
                                f"Calories: {total_nutrition['Calories']:.1f} kcal, "
                                f"Fat: {total_nutrition['Fat']:.1f} g, "
                                f"Saturates: {total_nutrition['Saturates']:.1f} g, "
                                f"Sugar: {total_nutrition['Sugar']:.1f} g, "
                                f"Salt: {total_nutrition['Salt']:.1f} g<br>"
                            )
                st.markdown(total_nutrition_str, unsafe_allow_html=True)
                # rows = zip(food_names1, confidences1)

                # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir="/tmp") as csv_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
                    csv_filename = csv_file.name
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Food Name", "Serving", "Confidence (%)", "Calories (kcal)", "Fat (g)", "Saturates (g)", "Sugar (g)", "Salt (g)"])
                    writer.writerows(nutrition_data)
                with open(csv_filename, "rb") as file:
                    the_csv = file.read()  
                
                st.toast("Prediction completed. Results saved to CSV.", icon="✅")
                download_csv = st.download_button(label="Download Predictions CSV",
                                data=the_csv,
                                file_name=f"{time_format}.csv", 
                                use_container_width=True)
                if download_csv:
                    os.remove(csv_filename)
            except ConnectionError as e:
                st.error(f"Failed to open YouTube video stream: {e}")
        else:
            st.error("Invalid YouTube URL or unable to extract YouTube ID.")
    else:
        st.error("YouTube URL is required.")

@st.cache_resource
def load_model():
    modelpath = r"./model/yolov10/YOLOv10m_new_total_VN_5_SGD.pt"
    
    model = YOLO(modelpath)
    return model

def resize_image(image):
    return image.resize((640, 640))

# Define color variables
COLOR_HIGH = "#FF4A3F"    # Red for high values
COLOR_MEDIUM = "#FECB02"  # Yellow for medium values
COLOR_LOW = "#85BB2F"     # Green for low values

def get_nutri_score_color(nutrient, value, serving_type):
    if serving_type == "per 100g":
        if nutrient == "Calories":
            if value < 100:
                return COLOR_LOW, "Low"
            elif value < 200:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Fat":
            if value < 3:
                return COLOR_LOW, "Low"
            elif value < 17.5:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Saturates":
            if value < 1.5:
                return COLOR_LOW, "Low"
            elif value < 5:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Sugar":
            if value < 5:
                return COLOR_LOW, "Low"
            elif value < 22.5:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Salt":
            if value < 0.3:
                return COLOR_LOW, "Low"
            elif value < 1.5:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
   
    elif serving_type == "1 serving":
        if nutrient == "Calories":
            if value < 150:
                return COLOR_LOW, "Low"
            elif value < 300:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Fat":
            if value < 5:
                return COLOR_LOW, "Low"
            elif value < 21:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Saturates":
            if value < 2:
                return COLOR_LOW, "Low"
            elif value < 6:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Sugar":
            if value < 6:
                return COLOR_LOW, "Low"
            elif value < 27:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Salt":
            if value < 0.4:
                return COLOR_LOW, "Low"
            elif value < 1.8:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"


def calculate_nutrient_percentage(nutrition):
    total_nutrition_value = (
        nutrition.get('Calories', 0) + 
        nutrition.get('Fat', 0) + 
        nutrition.get('Saturates', 0) + 
        nutrition.get('Sugar', 0) + 
        nutrition.get('Salt', 0)
    )
    if total_nutrition_value == 0:
        return {key: 0 for key in nutrition}

    percentage = {
        "Calories": (nutrition.get('Calories', 0) / total_nutrition_value) * 100,
        "Fat": (nutrition.get('Fat', 0) / total_nutrition_value) * 100,
        "Saturates": (nutrition.get('Saturates', 0) / total_nutrition_value) * 100,
        "Sugar": (nutrition.get('Sugar', 0) / total_nutrition_value) * 100,
        "Salt": (nutrition.get('Salt', 0) / total_nutrition_value) * 100,
    }
    return percentage

def detect_image_result(detected_image, model):
    boxes = detected_image[0].boxes

    if boxes:
        detected_img_arr_RGB = detected_image[0].plot()[:, :, ::1]
        detected_img_arr_BGR = detected_image[0].plot()[:, :, ::-1]
        fig_detected = create_fig(detected_img_arr_BGR, detected=True)
        st.plotly_chart(fig_detected, use_container_width=True)

        current_time = datetime.datetime.now()
        time_format = current_time.strftime("%d-%m-%Y")

        # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir='/tmp') as img_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=tempfile.gettempdir()) as img_file:
            img_filename = img_file.name
            cv2.imwrite(img_filename, detected_img_arr_RGB)
        with open(img_filename, 'rb') as file:
            the_img = file.read()
        
            detection_results = ""
            count_results = ""
            count_dict = {}
            food_names = []
            nutrition_data = []
            confidences = []
            counts = []
            total_nutrition = {
                "Calories": 0,
                "Fat": 0,
                "Saturates": 0,
                "Sugar": 0,
                "Salt": 0
            }

            total_nutrition_placeholder = st.empty()

            for box in boxes:
                # class_id = model.names[box.cls[0].item()]
                class_id = int(box.cls[0].item())
                class_name = class_names[int(class_id)]["name"] 
                food_names.append(class_name)
                conf = int(round(box.conf[0].item(), 2)*100)
                confidences.append(conf)
                serving = class_names[int(class_id)]["serving_type"]

                if class_id in count_dict:
                    count_dict[class_id] += 1
                else:
                    count_dict[class_id] = 1

                if class_name == "Con nguoi (Human)":
                    detection_results += f"<b style='color: black;'>Class name:</b> {class_name}<br><b style='color: black;'>Confidence:</b> {conf}%<br>---<br>"
                else:
                    nutrition = class_names[int(class_id)]["nutrition"]
                    if nutrition:
                        calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                        fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                        saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                        sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                        salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                        percentage_contribution = calculate_nutrient_percentage(nutrition)


                        nutrition_str = f"""
<div class="each-nutri-container">
    <div  id="calo-each-nutri-box" class="each-nutri-box" style="background-color: transparent;">
        <span class="each-nutri-name">Calories</span><br>
        <p class="each-nutri-number">{nutrition.get('Calories')} kcal</p>
        <span id="calo-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution['Calories']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {fat_color};">
        <span class="each-nutri-name">Fat</span><br>
        <p class="each-nutri-number">{nutrition.get('Fat')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Fat']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {saturates_color};">
        <span class="each-nutri-name">Saturates</span><br>
        <p class="each-nutri-number">{nutrition.get('Saturates')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Saturates']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {sugar_color};">
        <span class="each-nutri-name">Sugar</span><br>
        <p class="each-nutri-number">{nutrition.get('Sugar')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Sugar']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {salt_color};">
        <span class="each-nutri-name">Salt</span><br>
        <p class="each-nutri-number">{nutrition.get('Salt')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Salt']:.1f}%</span>
    </div>
</div>
                            """


                    detection_results += (
                    f"""<p class="item-header"><b>{count_dict[class_id]} ({conf}%):</b> {class_name}</p>
                    <p class="nutrition-header">Nutrition ({serving})</p>
                    <p class="nutrition-facts">{nutrition_str}</p>
                    <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                    """)

                    for key in total_nutrition:
                        if key in nutrition:
                            total_nutrition[key] += nutrition[key]

                    nutrition_data.append((
                        class_name,
                        serving,
                        conf,
                        nutrition.get('Calories'),
                        nutrition.get('Fat'),
                        nutrition.get('Saturates'),
                        nutrition.get('Sugar'),
                        nutrition.get('Salt')
                    ))

            total_nutrition_str = f"""
    <h5 class="total-nutrition-title">Total Nutrition Values</h5>
    <div class="total-nutrition-container">
        <div class="total-nutri-box" id="calo-box">
            <span class="total-nutri-num">{total_nutrition['Calories']:.1f} kcal</span><br>
            <span class="total-nutri-value-name">Calories</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-num">{total_nutrition['Fat']:.1f} gram</span><br>
            <span class="total-nutri-value-name">Fat</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-num">{total_nutrition['Saturates']:.1f} gram</span><br>
            <span class="total-nutri-value-name">Saturates</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-num">{total_nutrition['Sugar']:.1f} gram</span><br>
            <span class="total-nutri-value-name">Sugar</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-num">{total_nutrition['Salt']:.1f} gram</span><br>
            <span class="total-nutri-value-name">Salt</span>
        </div>
    </div>
"""


            # detection_results += total_nutrition_str
            total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)

            for object_type, count in count_dict.items():
                the_name = class_names[object_type]["name"]
                counts.append(count)
                # detection_results += f"<b style='color: black;'>Count of {the_name}:</b> {count}<br>"
                count_results += f"""
                <p class="total-count-result">{the_name}: {count}</p>"""
                

            scrollable_textbox = f"""<div id="result-nutri-container">{detection_results}</div>"""
            
            st.markdown("""<br>
                        <h5>Results:</h5>""", unsafe_allow_html=True)
            
            st.markdown(count_results, unsafe_allow_html=True)
            st.markdown(scrollable_textbox, unsafe_allow_html=True)

            # rows = zip(food_names, confidences, counts)
            # rows = zip(nutrition_data)

            # with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir='/tmp') as csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
                csv_filename = csv_file.name
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Food Name", "Serving", "Confidence (%)", "Calories (kcal)", "Fat (g)", "Saturates (g)", "Sugar (g)", "Salt (g)"])
                writer.writerows(nutrition_data)
            with open(csv_filename, 'rb') as file:
                the_csv = file.read()
        col1, col2 = st.columns(2, gap="large")
        with col1:    
            download_pic = st.download_button(label="Download Predicted Image",
                                    data=the_img,
                                    mime="image/jpg",
                                    file_name=f"{time_format}.jpg", 
                                    use_container_width=True)
            if download_pic:
                os.remove(img_filename)
        with col2: 
            download_csv = st.download_button(label="Download Predictions CSV", 
                               data=the_csv, 
                               file_name=f"{time_format}.csv", 
                               use_container_width=True)
            if download_csv:
                os.remove(csv_filename)
        st.divider()

    else:
        st.markdown("""<h5 id="no-food">No food detected</h5>
                    <p id="no-food-descr">The model did not detect any foods in the uploaded image.  
            Please try with a different image or adjust the model's 
            confidence threshold and try again.</p>
                    """, unsafe_allow_html=True)
        


def detect_image(conf, uploaded_file, model, url=False):
    if uploaded_file is not None:
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
        
        if "is_reset" not in st.session_state:
            st.session_state.is_reset = False
        
        if "show_image" not in st.session_state:
            st.session_state.show_image = True

        reset_button = None
        predict_button = None
        
        def toggle_button(reset = False):
            st.session_state.button_clicked = not st.session_state.button_clicked
            st.session_state.show_image = not st.session_state.show_image
            if reset == True:
                st.session_state.is_reset = not st.session_state.is_reset
        
        if url==False:
            uploaded_image = Image.open(uploaded_file)
        else:
            response = requests.get(uploaded_file)
            response.raise_for_status()
            uploaded_image = Image.open(BytesIO(response.content))

        resized_uploaded_image = resize_image(uploaded_image)

        original_image = st.empty()

        if st.session_state.show_image and not st.session_state.is_reset and not st.session_state.button_clicked:   
            original_image = st.image(resized_uploaded_image, output_format="JPEG", use_column_width=True)

        if not st.session_state.is_reset:
            col1, col2 = st.columns([0.8, 0.2], gap="large")
            with col1:
                if st.session_state.show_image and not st.session_state.button_clicked and not original_image == st.empty():
                    st.markdown("**Original Image**")
                elif not st.session_state.show_image and st.session_state.button_clicked:
                    st.markdown("**Predicted Image**")
            with col2:
                if not st.session_state.button_clicked:
                    predict_button = st.button("Predict", use_container_width=True, type="primary", on_click=toggle_button)
                else:
                    reset_button = st.button("Reset", use_container_width=True, type="primary", on_click=toggle_button, args=[True])
                    uploaded_file = None
        if st.session_state.show_image and st.session_state.is_reset and not st.session_state.button_clicked:
            st.session_state.is_reset = False

        if st.session_state.button_clicked and not reset_button:
            with st.spinner("Running..."):
                detected_image = model.predict(resized_uploaded_image, conf=conf, imgsz=640)
                detect_image_result(detected_image, model)        
        # elif uploaded_file and url:
        #     with st.spinner("Running..."):
        #         detected_image = model(resized_uploaded_image, conf=conf, imgsz=640)
        #         detect_image_result(detected_image, model)

        
        # elif url==False and not st.session_state.button_clicked and not predict_button:
        #     original_image.image(resized_uploaded_image, output_format="JPEG", use_column_width=True)

def detect_camera(conf, model, address):
    vid_cap = cv2.VideoCapture('rtsp://admin:' + address)
    vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  
    vid_cap.set(cv2.CAP_PROP_FPS, 15)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    
    while True:
        if vid_cap.isOpened():
            st.toast("Connected", icon="✅")
            break
        else:
            vid_cap.release()
            return
    try: 
        st_frame = st.empty()

        displayed_dishes = set()

        total_nutrition = {
            "Calories": 0,
            "Fat": 0,
            "Saturates": 0,
            "Sugar": 0,
            "Salt": 0
        }

        st.markdown("""### Results:""")

        total_nutrition_placeholder = st.empty()

        frame_count = 0
        start_time = time.time()
        while True:   
            success, image = vid_cap.read()
            if success:
                # mirrored_frame = cv2.flip(image, 1)
                results = model.track(source=image, conf=conf, imgsz=640, save=False, device="cpu", stream=True)


                new_detections = False  
                detection_results = ""

                for r in results:
                    im_bgr = r.plot()
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 1.0:
                        fps = frame_count / elapsed_time
                        start_time = time.time()
                        frame_count = 0
                    cv2.putText(im_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                    im_rgb = Image.fromarray(im_bgr[..., ::-1])
                    st_frame.image(im_rgb, caption='Camera IP', use_column_width=True)

                    for pred in r.boxes:
                        class_id = int(pred.cls[0].item())
                        class_name = class_names[int(class_id)]["name"]
                        confident = int(round(pred.conf[0].item(), 2)*100)
                        serving = class_names[int(class_id)]["serving_type"]

                        if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                            detection_results += f"<b style='color: black;'>Class name:</b> {class_name}<br><b style='color: black;'>Confidence:</b> {confident}%<br>---<br>"
                            displayed_dishes.add(class_name)
                            new_detections = True
                        elif class_name not in displayed_dishes:
                            nutrition = class_names[int(class_id)]["nutrition"]
                            if nutrition:
                                displayed_dishes.add(class_name)
                                new_detections = True

                                calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                                fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                                saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                                sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                                salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                                percentage_contribution = calculate_nutrient_percentage(nutrition)

                                nutrition_str = (
                                    f"<span>Calories: {nutrition.get('Calories')} kcal ({percentage_contribution['Calories']:.1f}%) - {calories_desc}</span>, "
                                    f"<span style='color: {fat_color};'>Fat: {nutrition.get('Fat')} g ({percentage_contribution['Fat']:.1f}%) - {fat_desc}</span>, "
                                    f"<span style='color: {saturates_color};'>Saturates: {nutrition.get('Saturates')} g ({percentage_contribution['Saturates']:.1f}%) - {saturates_desc}</span>, "
                                    f"<span style='color: {sugar_color};'>Sugar: {nutrition.get('Sugar')} g ({percentage_contribution['Sugar']:.1f}%) - {sugar_desc}</span>, "
                                    f"<span style='color: {salt_color};'>Salt: {nutrition.get('Salt')} g ({percentage_contribution['Salt']:.1f}%) - {salt_desc}</span>"
                                )

                                detection_results += (
                                    f"<b style='color: black;'>Food name:</b> {class_name}<br>"
                                    f"<b style='color: black;'>Confidence:</b> {confident}%<br>"
                                    f"<b style='color: black;'>Nutrition ({serving}):</b> {nutrition_str}<br>---<br>"
                                )

                                for key in total_nutrition:
                                    if key in nutrition:
                                        total_nutrition[key] += nutrition[key]
                if new_detections:
                    scrollable_textbox = f"""
                        <div style="
                            font-family: 'Source Code Pro','monospace';
                            font-size: 16px;
                            overflow-y: scroll;
                            padding: 10px;
                            width: auto;
                            height: auto;
                        ">
                            {detection_results}
                        </div>
                    """
                    st.markdown(scrollable_textbox, unsafe_allow_html=True)
                total_nutrition_str = f"""
                    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                        <div style="border: 1px solid black; padding: 10px; border-radius: 5px; text-align: center; width: 150px;">
                            <span><b>Calories:</b></span><br>
                            <span>{total_nutrition['Calories']:.1f} kcal</span>
                        </div>
                        <div style="border: 1px solid black; padding: 10px; border-radius: 5px; text-align: center; width: 150px;">
                            <span><b>Fat:</b></span><br>
                            <span>{total_nutrition['Fat']:.1f} g</span>
                        </div>
                        <div style="border: 1px solid black; padding: 10px; border-radius: 5px; text-align: center; width: 150px;">
                            <span><b>Saturates:</b></span><br>
                            <span>{total_nutrition['Saturates']:.1f} g</span>
                        </div>
                        <div style="border: 1px solid black; padding: 10px; border-radius: 5px; text-align: center; width: 150px;">
                            <span><b>Sugar:</b></span><br>
                            <span>{total_nutrition['Sugar']:.1f} g</span>
                        </div>
                        <div style="border: 1px solid black; padding: 10px; border-radius: 5px; text-align: center; width: 150px;">
                            <span><b>Salt:</b></span><br>
                            <span>{total_nutrition['Salt']:.1f} g</span>
                        </div>
                    </div>
                    """

                total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)
            else:
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
    finally:
        vid_cap.release()
        displayed_dishes.clear()

from typing import List, NamedTuple
result_queue = queue.Queue(maxsize=12)

class Detection(NamedTuple):
    class_id: int
    class_name: str
    confident: float
    serving: str

class VideoTransformer(VideoTransformerBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.prev_time = time.time()
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        mirrored_frame = cv2.flip(img, 1)
        results = self.model(source=mirrored_frame, conf=self.conf, imgsz=640, save=False, device="cpu", stream=True, vid_stride=80)
        # results = self.model.track(source=mirrored_frame, conf=self.conf, imgsz=640, save=False, device="cpu", stream=True)
        detections = []
        for r in results:
            im_bgr = r.plot()
            for pred in r.boxes:
                class_id = int(pred.cls[0].item())
                detections.append(
                    Detection(
                        class_id = int(pred.cls[0].item()),
                        class_name = class_names[int(class_id)]["name"],
                        confident = int(round(pred.conf[0].item(), 2)*100),
                        serving = class_names[int(class_id)]["serving_type"],
                    )
                )

        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time

        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            self.prev_time = current_time
            self.frame_count = 0
        else:
            fps = self.frame_count / elapsed_time
  

        cv2.putText(im_rgb, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not result_queue.full():
            result_queue.put(detections)

        return av.VideoFrame.from_ndarray(im_rgb, format="rgb24")   



def detect_webcam(conf, model):
    webrtc_ctx = webrtc_streamer(
        key="webcam_1",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(conf, model),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.markdown("""### Results:""")
        total_nutrition_placeholder = st.empty()
        results_placeholder = st.empty()
        
        while True:
            total_nutrition = {
                "Calories": 0,
                "Fat": 0,
                "Saturates": 0,
                "Sugar": 0,
                "Salt": 0
            }
            detections = result_queue.get()

            detection_results = ""
            new_detections = False
            displayed_dishes = set()

            for detection in detections:
                class_id = detection.class_id
                class_name = detection.class_name
                confident = detection.confident
                serving = detection.serving
                
                if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                    detection_results += f"<b style='color: black;'>Class name:</b> {class_name}<br><b style='color: black;'>Confidence:</b> {confident}%<br>---<br>"
                    displayed_dishes.add(class_name)
                    new_detections = True
                elif class_name not in displayed_dishes:
                    nutrition = class_names[int(class_id)]["nutrition"]
                    if nutrition:
                        displayed_dishes.add(class_name)
                        new_detections = True

                        calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                        fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                        saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                        sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                        salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                        percentage_contribution = calculate_nutrient_percentage(nutrition)

                        nutrition_str = (
                            f"<span>Calories: {nutrition.get('Calories')} kcal ({percentage_contribution['Calories']:.1f}%) - {calories_desc}</span>, "
                            f"<span style='color: {fat_color};'>Fat: {nutrition.get('Fat')} g ({percentage_contribution['Fat']:.1f}%) - {fat_desc}</span>, "
                            f"<span style='color: {saturates_color};'>Saturates: {nutrition.get('Saturates')} g ({percentage_contribution['Saturates']:.1f}%) - {saturates_desc}</span>, "
                            f"<span style='color: {sugar_color};'>Sugar: {nutrition.get('Sugar')} g ({percentage_contribution['Sugar']:.1f}%) - {sugar_desc}</span>, "
                            f"<span style='color: {salt_color};'>Salt: {nutrition.get('Salt')} g ({percentage_contribution['Salt']:.1f}%) - {salt_desc}</span>"
                        )

                        detection_results += (
                            f"<b style='color: black;'>Food name:</b> {class_name}<br>"
                            f"<b style='color: black;'>Confidence:</b> {confident}%<br>"
                            f"<b style='color: black;'>Nutrition ({serving}):</b> {nutrition_str}<br>---<br>"
                        )
                        for key in total_nutrition:
                                    if key in nutrition:
                                        total_nutrition[key] += nutrition[key]

            results_placeholder.markdown(detection_results, unsafe_allow_html=True)
        
            total_nutrition_str = (
                f"<b style='color: black;'>Total Nutrition:</b> "
                f"Calories: {total_nutrition['Calories']:.1f} kcal, "
                f"Fat: {total_nutrition['Fat']:.1f} g, "
                f"Saturates: {total_nutrition['Saturates']:.1f} g, "
                f"Sugar: {total_nutrition['Sugar']:.1f} g, "
                f"Salt: {total_nutrition['Salt']:.1f} g<br>"
            )

            total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)
    
            displayed_dishes.clear()
    
    result_queue.queue.clear()




import onnxruntime as ort

model_path = "./model/yolov10/YOLOv10m_new_total_VN_5_SGD.onnx"


@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0

    img = img.astype(np.float16)


    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img



def postprocess(outputs, frame, original_size, conf, model):
    h, w, _ = original_size
    for output_array in outputs:
        for output in output_array[0]:
            x1, y1, x2, y2, score, class_id = output[:6]
            if score > conf: 
                x1, y1, x2, y2 = x1 * w / 640, y1 * h / 640, x2 * w / 640, y2 * h / 640
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if class_id < len(class_names):
                    class_name = class_names[int(class_id)]["name"]
                label = f"{class_name}: {score:.2f}"

                font_scale = 1.0  
                thickness = 3     
                
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        return frame

def detect_video(conf, uploaded_file):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
            temp_input_file.write(uploaded_file.read())
            temp_input_file_path = temp_input_file.name
        detect_from_file(conf=conf, video_file=temp_input_file_path)

def detect_from_file(conf, video_file):
    session, input_name, output_name = load_onnx_model()
    if video_file:
        cap = cv2.VideoCapture(video_file)

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%d-%m-%Y")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)

    # with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir='/tmp') as mp4_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=tempfile.gettempdir()) as mp4_file:
        mp4_filename = mp4_file.name
        out = cv2.VideoWriter(mp4_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    with open(mp4_filename, "rb") as file:
        the_mp4 = file.read()

    st_frame = st.empty()

    frames1 = []
    displayed_dishes = set()
    nutrition_data = []
    
    total_nutrition = {
        "Calories": 0,
        "Fat": 0,
        "Saturates": 0,
        "Sugar": 0,
        "Salt": 0
    }

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        rewind_button = st.button("Rewind 10s", use_container_width=True)
    with col2:
        stop_button = st.button("Stop", use_container_width=True)
        stop_pressed = False
    with col3:
        fast_forward_button = st.button("Fast-forward 10s", use_container_width=True)

    frame_count = 0
    start_time = time.time()
        
    stop_pressed = False
    skip_frames = 0

    st.markdown("""### Results:""")

    while True:
        if skip_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames)
            skip_frames = 0
        if rewind_button:
            skip_frames = -int(fps * 10) 
        if fast_forward_button:
            skip_frames = int(fps * 10)   
        if stop_button:
            stop_pressed = True

        ret, frame = cap.read()
        if not ret or stop_pressed:
            break

        input_frame = preprocess(frame)
        outputs = session.run([output_name], {input_name: input_frame})
        detected_frame = postprocess(outputs, frame, original_size, conf, model=session)

        new_detections = False  
        detection_results = ""

        for output_array in outputs:        
            for output in output_array[0]:
                x1, y1, x2, y2, score, class_id = output[:6]
                if score > conf:
                    if class_id < len(class_names):
                        class_name = class_names[int(class_id)]["name"]
                    label = f"{class_name}: {score:.2f}"
                    frames1.append(frame_count)
                    confident = round(score*100)
                    serving = class_names[int(class_id)]["serving_type"]

                    if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                        detection_results += f"<b style='color: black;'>Class name:</b> {class_name}<br><b style='color: black;'>Confidence:</b> {confident}%<br>---<br>"
                        displayed_dishes.add(class_name)
                        new_detections = True
                    elif class_name not in displayed_dishes:
                        nutrition = class_names[int(class_id)]["nutrition"]
                        if nutrition:
                            displayed_dishes.add(class_name)
                            new_detections = True

                            calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                            fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                            saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                            sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                            salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                            percentage_contribution = calculate_nutrient_percentage(nutrition)

                            nutrition_str = (
                                f"<span>Calories: {nutrition.get('Calories')} kcal ({percentage_contribution['Calories']:.1f}%) - {calories_desc}</span>, "
                                f"<span style='color: {fat_color};'>Fat: {nutrition.get('Fat')} g ({percentage_contribution['Fat']:.1f}%) - {fat_desc}</span>, "
                                f"<span style='color: {saturates_color};'>Saturates: {nutrition.get('Saturates')} g ({percentage_contribution['Saturates']:.1f}%) - {saturates_desc}</span>, "
                                f"<span style='color: {sugar_color};'>Sugar: {nutrition.get('Sugar')} g ({percentage_contribution['Sugar']:.1f}%) - {sugar_desc}</span>, "
                                f"<span style='color: {salt_color};'>Salt: {nutrition.get('Salt')} g ({percentage_contribution['Salt']:.1f}%) - {salt_desc}</span>"
                            )

                            detection_results += (
                                f"<b style='color: black;'>Food name:</b> {class_name}<br>"
                                f"<b style='color: black;'>Confidence:</b> {confident}%<br>"
                                f"<b style='color: black;'>Nutrition ({serving}):</b> {nutrition_str}<br>---<br>"
                            )

                            for key in total_nutrition:
                                if key in nutrition:
                                    total_nutrition[key] += nutrition[key]

                        nutrition_data.append((
                            class_name,
                            serving,
                            confident,
                            nutrition.get('Calories'),
                            nutrition.get('Fat'),
                            nutrition.get('Saturates'),
                            nutrition.get('Sugar'),
                            nutrition.get('Salt')
                        ))

                    frames1.append(frame_count)

        if new_detections:
            scrollable_textbox = f"""
                <div style="
                    font-family: 'Source Code Pro','monospace';
                    font-size: 16px;
                    overflow-y: scroll;
                    padding: 10px;
                    width: auto;
                    height: auto;
                ">
                    {detection_results}
                </div>
            """
            st.markdown(scrollable_textbox, unsafe_allow_html=True)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        cv2.putText(detected_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        im_rgb = Image.fromarray(detected_frame[..., ::-1])
        st_frame.image(im_rgb, caption='Predicted video', use_column_width=True)
        out.write(detected_frame)

        if stop_button:
            stop_pressed = True
            stop_button = None
            break

    cap.release()
    out.release()
    displayed_dishes.clear()

    total_nutrition_str = (
                    f"<b style='color: black;'>Total Nutrition:</b> "
                    f"Calories: {total_nutrition['Calories']:.1f} kcal, "
                    f"Fat: {total_nutrition['Fat']:.1f} g, "
                    f"Saturates: {total_nutrition['Saturates']:.1f} g, "
                    f"Sugar: {total_nutrition['Sugar']:.1f} g, "
                    f"Salt: {total_nutrition['Salt']:.1f} g<br>"
                )
    st.markdown(total_nutrition_str, unsafe_allow_html=True)

    # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir="/tmp") as csv_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
        csv_filename = csv_file.name
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Food Name", "Serving", "Confidence (%)", "Calories (kcal)", "Fat (g)", "Saturates (g)", "Sugar (g)", "Salt (g)"])
        writer.writerows(nutrition_data)
    with open(csv_filename, "rb") as file:
        the_csv = file.read()
    
    col1, col2 = st.columns(2, gap="large")
    with col1:    
        download_video = st.download_button(label="Download Processed Video",
                                data=the_mp4,
                                mime="video/mp4",
                                file_name=f"{timestamp}.mp4", 
                                use_container_width=True)
        if download_video:
            os.remove(mp4_filename)
    with col2: 
        download_csv = st.download_button(label="Download Predictions CSV", 
                            data=the_csv, 
                            file_name=f"{timestamp}.csv", 
                            use_container_width=True)
        if download_csv:
            os.remove(csv_filename)