import streamlit as st
import time
import base64 
from pathlib import Path
from streamlit_navigation_bar import st_navbar
from utils import _display_detected_frame, detect_camera, detect_image, detect_video, detect_webcam, load_onnx_model, load_model

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":microscope:"
)

# import streamlit as st
# from PIL import Image

# image = Image.open('./pages/bg-about-cuisine.jpg')

# st.image(image)
# st.markdown(f"""
# <style>
#     .stImage  {{
#         position: relative;
#         width: 100%;
#         height: calc(100px + 7vw);
#         overflow: hidden;
#     }}
# """, unsafe_allow_html=True)

# st.markdown(f"""
# <h1 class="header-title">📑 About FoodDetector</h1>
#             """, unsafe_allow_html=True)         

def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Convert your image to base64
img_path = './assets/img/bg-about-cuisine.png'
img_base64 = img_to_base64(img_path)

st.markdown(f"""
<div class="header-container">
    <img src="data:image/jpg;base64,{img_base64}" class="header-image">
    <div class="header-overlay">
        <div class="header-title">Welcome to FoodDetector 🕵️</div>
        <div class="header-subtitle">An easy way to detect Vietnamese dishes!</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Import img
# def img_bg_cover(img_path):
#     with open(img_path, 'rb') as img_file:
#         return base64.b64encode(img_file.read()).decode('utf-8')

# current_path = Path(__file__).parent
# img_path = current_path / 'pages' / 'img' / 'bg-about-cuisine.jpg'
# img_cover = img_bg_cover(img_path)

# # Cover
# st.markdown(f"""
# <style>
#     .header-container {{
#         position: relative;
#         width: 100%;
#         height: calc(100px + 7vw);
#         overflow: hidden;
#     }}
#     .header-image {{
#         position: absolute;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         background-image: url(data:image/jpg;base64,{img_cover});
#         background-size: cover;
#         background-position: center top;
#     }}
    
#     .header-title {{
#     position: absolute;
#     bottom: 0;
#     /* left: 50%;
#     transform: translateX(-50%); */
#     width: 100%;
#     color: white;
#     background-color: rgba(0, 0, 0, 0.6);
#     padding: 5px 10px;
#     letter-spacing: 1px;
#     font-weight: 800;
#     }}
# </style>

# <div class="header-container">
#     <div class="header-image"></div>
#     <h1 class="header-title">📑 About FoodDetector</h1>
# </div>
# """, unsafe_allow_html=True)
# End Cover

def render_content():         
    confidence = 0.5

    with st.container():
        # st.title("Welcome to _:green[FoodDetector]_ :male-detective:")
        st.divider()

    #     st.markdown('''
    # FoodDetector uses the _YOLOv10m_ pretrained models for fine-tuning with `VietFood57`, a new custom-made Vietnamese food dataset created for detecting local dishes and achieved a `mAP50` of `0.934`.  
    # It can be used to detect <a href="/Dataset" target="_blank" style="color: #4CAF50; font-weight: bold; font-style: italic; text-decoration: none;">`57`</a> Vietnamese dishes from a picture, video, webcam, and an IP camera through RTSP.
    # ''', unsafe_allow_html=True)

        st.markdown(f'''
    <ul class="define">
        <li class="define-li">FoodDetector uses the <strong>YOLOv10m</strong> pretrained models for fine-tuning with <code>VietFood57</code>, 
        a new custom-made Vietnamese food dataset created for detecting local dishes and achieved a <code>mAP50</code> of <code>0.934</code>.</li>
        <li class="define-li">It can be used to detect <a href="/Dataset" target="_self">57</a> Vietnamese dishes from a picture, video, webcam, and an IP camera through RTSP.</li>
    </ul>
                    ''', unsafe_allow_html=True)


        st.divider()

        st.markdown(f'''

    <style>
        #quick-note {{
        margin-left: 0;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }}

    .title-text-score {{
        font-weight: 700;
        border-radius: 5px;
        background-color: var(--grey);
        padding: 0.5rem;
        display: inline;
    }}

    </style>
    <h4>Adjust the confident score 🚩</h4>
    <p class="define" id="quick-note"><strong>Quick note 📝</strong>: consideration for selecting the best suited confident score:</p>
    <div class="adjust-section">
        <p class="define title-text-score">High confident score (>= 50%):</p>
        <p class="define subtitle-text-score">Set a higher threshold will make the model to predict with a higher accuracy detection but it will have a low recall as fewer object will 
        be detected because of the high precision constraint.</p>
        <p class="define title-text-score">Low confident score (< 50%):</p>        
        <p class="define subtitle-text-score">Set a lower threshold will enable the model to detect more object - 
    high recall because of the low precision constraint.</p>
    </div>     
                ''', unsafe_allow_html=True)


        model1 = load_model()
        model = load_onnx_model()

        tab1, tab2, tab3, tab4 = st.tabs(["Image", "Video", "Webcam", "IP Camera"])

        with tab1:
            st.subheader("Image Upload :frame_with_picture:")

            # Accordion
            expander = st.expander("Instructions: Image upload and URL")  
            expander.write('''

    - Uploading image files from the user's local machine or using an image URL is supported.
    - After the prediction process, two buttons will appear to download the results as an image file with bounding boxes or a CSV file.
    - The results are generated when the user clicks the button and are named in the format: `"%date-%month-%year_%hour-%minute".jpg/csv`.
            ''', unsafe_allow_html=True)
            st.markdown(f'''
    <style>
    [data-testid="stExpanderDetails"] ul li {{
        font-size: calc(12px + 0.1vw);
        margin: 1rem 0 1rem 1.5rem;
        color: black
    }}
    .stExpander p {{
        font-size: calc(13px + 0.1vw);
        font-weight: 700;
        color: var(--brown);
        padding-left: 0.5rem;
    }}
    .st-emotion-cache-1h9usn1 {{
        background-color: var(--button-color-yellor);
        font-size: calc(16px +1vw);
    }}

    [data-testid="stExpanderDetails"] {{
        background-color: var(--grey-light);
        border-radius: 8px;
    }}
    </style>
                        ''', unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Choose a picture", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

            if uploaded_file:
                detect_image(confidence, model=model1, uploaded_file=uploaded_file)

                # detections = detect_image_onnx(model, uploaded_file, confidence)

            st.markdown('<br><br>', unsafe_allow_html=True)
            st.subheader("Enter a picture URL 	:link:")
            with st.form("picture_form"):
                col1, col2 = st.columns([0.8, 0.2], gap="medium")
                with col1:
                    picture_url = st.text_input("Label", label_visibility="collapsed", placeholder="https://ultralytics.com/images/bus.jpg")
                with col2:
                    submitted = st.form_submit_button("Predict", use_container_width=True)
            if submitted and picture_url:
                detect_image(confidence, model=model1, uploaded_file=picture_url, url=True)            

        with tab2:
                        
            st.subheader("Video Upload :movie_camera:")
            expander = st.expander("Instructions: Video upload and URL")  
            expander.write('''
- Video: upload video files (`.mp4, .mpeg4, etc.`) from the user's local machine.
- Youtube video or shorts URL links are supported for real-time prediction.
- The results will be in a CSV file recording all dishes detected across all frames (no image results).
            ''', unsafe_allow_html=True)
            
            uploaded_clip = st.file_uploader("Choose a clip", accept_multiple_files=False, type=['mp4'])
            if uploaded_clip:
                detect_video(conf=confidence, uploaded_file=uploaded_clip)

            else: 
                st.subheader("Enter YouTube URL :tv:")
                tube = st.empty()
                with st.form("youtube_form"):
                    col1, col2 = st.columns([0.8, 0.2], gap="medium")
                    with col1:
                        youtube_url = st.text_input("Label", label_visibility="collapsed", placeholder="https://youtu.be/LNwODJXcvt4")
                    with col2:
                        submitted = st.form_submit_button("Predict", use_container_width=True)
                if submitted and youtube_url:            
                    _display_detected_frame(conf=confidence, model=model1, 
                                            st_frame=tube,
                                            youtube_url=youtube_url)

        with tab3:
            
            st.header("Webcam :camera:")
            expander = st.expander("Instructions: Webcam connection")  
            expander.write('''
- Webcam: [Streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) is used to handle local webcam connection due to deployment on [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud).
- Users can choose their webcam input for live detection.
- No result files will be generated as the process may run continuously.

            ''', unsafe_allow_html=True)
            detect_webcam(confidence, model=model1)

        with tab4:
            
            st.header("IP Camera :video_camera:")
            expander = st.expander("Instructions: IP Camera connection")  
            expander.write('''
- IP Camera: A RTSP address of the user’s camera must be provided.
- The camera must be configured beforehand to allow connection from an external network.
            ''', unsafe_allow_html=True)    
            
            st.text("Enter your Camera (RTSP) address: ")
            col1, col2 = st.columns([1, 4])
            with col1: 
                st.text("rtsp://admin:")
            with col2:
                with st.form("ip_camera_form"):
                    address = st.text_input("Label", label_visibility="collapsed", placeholder="hd543211@192.168.14.106:554/Streaming/channels/101")
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
    
    st.markdown('''
    <div>
        <a href="#top-section" class="top-button" onclick="smoothScroll(event, 'top-section')">⬆</a>                
    </div>
    
    <script>
    function smoothScroll(event, targetId) {
        event.preventDefault();
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }
    }
    </script>
                ''', unsafe_allow_html=True)

# Nav bar
def navbar(active_page):
    return f"""
   
    <div class="custom-navbar">
        <div class="nav-items">
            <a href="#" target="_self" class="nav-item {'active' if active_page == 'Home' else ''}">🏠 Home</a>
            <a href="/dataset" target="_self" class="nav-item {'active' if active_page == 'About' else ''}">📄 About</a>
        </div>
        <a href="https://github.com" target="_blank" class="nav-item github-icon">
            <!-- GitHub SVG icon here -->
        </a>
    </div>
    """

def styling_css():
    with open('./assets/css/general-style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
        
def home_page():
    st.markdown(navbar('Home'), unsafe_allow_html=True)
    

def about_page():
    st.markdown(navbar('About'), unsafe_allow_html=True)
    

# Main app logic
def main():
        # Get the current page from the URL
    styling_css()
    query_params = st.query_params
    path = query_params.get("page", ["home"])[0].lower()
    
    # Always render the navbar
    st.markdown(navbar('Home' if path == 'home' else 'About'), unsafe_allow_html=True)
    
    if path == "about":
        st.markdown('<h1 style="color: white; font-size: 40px;">About Section</h1>', unsafe_allow_html=True)
        st.write("This is the About section. Here you can add information about your project or organization.")
    else:
        render_content()

if __name__ == "__main__":
    main()