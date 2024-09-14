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

st.markdown('''
    <div id="top-section"></div>
    ''', unsafe_allow_html=True)
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Convert your image to base64
img_path = './assets/img/bg-about-cuisine.png'
img_base64 = img_to_base64(img_path)

# Convert your image to base64
img_path_nutrition = './assets/img/nutrition-table.png'
img_base64_nutrition = img_to_base64(img_path_nutrition)

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
    with st.container():
        # st.title("Welcome to _:green[FoodDetector]_ :male-detective:")
        st.divider()

    #     st.markdown('''
    # FoodDetector uses the _YOLOv10m_ pretrained models for fine-tuning with `VietFood57`, a new custom-made Vietnamese food dataset created for detecting local dishes and achieved a `mAP50` of `0.934`.  
    # It can be used to detect <a href="/Dataset" target="_blank" style="color: #4CAF50; font-weight: bold; font-style: italic; text-decoration: none;">`57`</a> Vietnamese dishes from a picture, video, webcam, and an IP camera through RTSP.
    # ''', unsafe_allow_html=True)

        st.markdown(f'''
    <ul class="define introduction" style="margin-top: 0; margin-bottom: 0;">
        <li class="define-li home-page">FoodDetector uses the <strong>YOLOv10m</strong> pretrained models for fine-tuning with <code>VietFood57</code>, 
        a new custom-made Vietnamese food dataset created for detecting local dishes and achieved a <code>mAP50</code> of <code>0.934</code>.</li>
        <li class="define-li home-page">It can be used to detect <a href="/dataset" target="_self">57</a> Vietnamese dishes from a picture, video, webcam, and an IP camera through RTSP.</li>
    </ul>
                    ''', unsafe_allow_html=True)


        st.divider()

        st.markdown(f'''
                    <h4>Adjust the confident score 🚩</h4>
                    ''', unsafe_allow_html=True)
        confidence = float(st.slider(
            label="",label_visibility="collapsed", min_value=10, max_value=100, value=50 
        ))/ 100
        
        st.markdown(f'''
    <style>
        #quick-note {{
        margin-left: 0;
        margin-bottom: 0.5rem;
    }}
    
    p#quick-note.define {{
        margin-top: 0;
    }}

    .title-text-score {{
        font-weight: 700;
        border-radius: 5px;
        background-color: var(--grey);
        padding: 0.5rem;
        display: inline;
    }}

    p.define.subtitle-text-score {{
        margin-bottom: 1rem;
    }}
    
    </style>
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

        st.divider()
        st.markdown(f'''
                    <h4>Nutrition value score📊</h4>
                    ''', unsafe_allow_html=True)

        st.markdown(f'''
    <ul class="define nutrition" style="margin-top: 0; margin-bottom: 0;">
        <li class="define-li home-page">Our nutrition values are based on the <strong>Traffic Light system</strong>🚦.</li>
        <li class="define-li home-page">All nutrition information provided is approximate.</li>
    </ul>
                    ''', unsafe_allow_html=True)
        

        
        expander = st.expander("See more")  
        expander.write(f'''
<div class="nutrition-container">
    <img src="data:image/jpg;base64,{img_base64_nutrition}" class="nutrition-img">
    <ul class="nutrition-explain">
        <li class="nutrition-explain-details"><strong class="color-section" id="green">Green (Low)</strong>: Very healthy. Enjoy without worry.</li>
        <li class="nutrition-explain-details"><strong class="color-section" id="yellow">Yellow (Medium)</strong>: Consume in moderation or combine with healthier options.</li>
        <li class="nutrition-explain-details"><strong class="color-section" id="red">Red (High)</strong>: Limit consumption and look for healthier alternatives.</li>
    </ul>
    <p class="nutrition-explain-details">For more information, please refer to the <a href="https://www.nutricalc.co.uk/case-study/case-study-uk-traffic-light-front-of-pack-colour-thresholds/">NutriCalc</a>,
    <a href="https://heas.health.vic.gov.au/resources/government-guidelines/traffic-light-system/">Healthy Eating Advisory Service</a></p>

<style>
    li.nutrition-explain-details {{
        margin-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
    }}
    
    p.nutrition-explain-details {{
        font-weight: 400 !important;
        margin: 1rem 0 !important;
    }}
    
    img.nutrition-img {{
        margin-top: 1rem;
        margin-bottom: 1rem;
        width: 90%;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }} 
    
    .color-section {{
        padding: 3px 6px;
        border-radius: 5px;
    }}
    
    #green {{
        background-color: var(--green-nu);
    }}
    
    #yellow {{
        background-color: var(--yellow-nu);
    }}
    
    #red {{
        background-color: var(--red-nu);
    }}
</style>
''', unsafe_allow_html=True)
        
        st.markdown(f'''<br><br>''', unsafe_allow_html=True)        
        model1 = load_model()
        model = load_onnx_model()

        st.markdown("""
    <style>
    /* Style the tab labels */
    button[data-baseweb="tab"] {
        padding: calc(8px + 0.2vw) calc(8px + 0.5vw);
        gap: 0;

    }
    button[data-baseweb="tab"] p {
        font-size: calc(9px + 0.3vw) !important;
        font-weight: 500 !important;        
    }
    
    div[data-baseweb="tab-list"] {
        gap: 0;
    }
    /* Style the active tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--button-color-yellow); /* Active tab color */
        border-radius: 8px 7px 0 0;
        color: black;
    }

    /* Style the inactive tabs */
    button[data-baseweb="tab"][aria-selected="false"] {
        color: var(--grey-code-expander);
    }
    
    div[data-baseweb="tab-border"] {
    }
    </style>
""", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Image", "Video", "Webcam", "IP Camera"])

        with tab1:
            st.subheader("Image Upload :frame_with_picture:")

            # Accordion
            expander = st.expander("Instructions: Image upload and URL")  
            expander.write('''
    - Uploading image files from the user's local machine or using an image URL is supported.
    - After the prediction process, two buttons will appear to download the results as an image file with bounding boxes or a CSV file.
    - The results are generated when the user clicks the button and are named in the format: `"%date-%month-%year".jpg/csv`.
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
- Video: upload video files `(.mp4, .mpeg4, etc.)` from the user's local machine.
- Youtube video or shorts URL links are supported for real-time prediction.
- The results will be in a CSV file recording all dishes detected across all frames (no image results).
            ''', unsafe_allow_html=True)
            
            uploaded_clip = st.file_uploader("Choose a clip", accept_multiple_files=False, type=['mp4'])
            if uploaded_clip:
                detect_video(conf=confidence, uploaded_file=uploaded_clip)

            else:
                st.markdown('<br><br>', unsafe_allow_html=True) 
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
            with st.form("ip_camera_form"):
                col1, col2 = st.columns([2, 8])
                with col1:
                    st.write("rtsp://admin:")  # Place the static text on the same line
                with col2:
                    address = st.text_input(
                        "Label", 
                        label_visibility="collapsed", 
                        placeholder="hd543211@192.168.14.106:554/Streaming/channels/101"
                    )
                    
                col1, col2 = st.columns([2, 1.35])
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
        <a href="#top-section" class="top-button" onclick="smoothScroll(event, 'top-section')">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" width="16" height="16">
            <path d="M240.971 130.524l194.343 194.343c9.373 9.373 9.373 24.569 0 33.941l-22.667 22.667c-9.357 9.357-24.522 9.375-33.901.04L224 227.495 69.255 381.516c-9.379 9.335-24.544 9.317-33.901-.04l-22.667-22.667c-9.373-9.373-9.373-24.569 0-33.941L207.03 130.525c9.372-9.373 24.568-9.373 33.941-.001z"/>
        </svg>
        </a>                
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
            <a href="/main" target="_self" class="nav-item {'active' if active_page == 'Home' else ''}">🏠 Home</a>
            <a href="/dataset" target="_self" class="nav-item {'active' if active_page == 'About' else ''}">📄 About</a>
        </div>
        <a href="https://github.com/nvhnam/FoodDetector" target="_blank" class="nav-item">
            <svg id="github-icon" height="32" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="32" data-view-component="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" fill="currentColor"></path>
            </svg>
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