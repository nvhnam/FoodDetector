import streamlit as st
import base64 
from pathlib import Path

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":microscope:"
)

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
        <div class="header-title2">📃 About 📃</div>
    </div>
</div>
""", unsafe_allow_html=True)

def render_content():  
    # st.title(":package: Dataset")
    st.divider()
    st.markdown('''
    <h4 class="dataset-page">VietFood57: A Dataset for Vietnamese Food Detection</h4>
    <ul class="define dataset-page">
        <li class="define-li dataset-page">This dataset contains <code>22,920</code> images with <code>58</code> classes which included an extra 
        class for recognizing human faces as the purpose of this research is to detect and monitor people eating activity so 
        being able to know the human existence during the detection can give a more wholesome result. After all, the eating duration 
        can also be derived from human detection along with the dishes.</li>
        <li class="define-li dataset-page">VietFood57 is divided in <code>70%</code>/<code>20%</code>/<code>10%</code> with <code>16,045</code> 
        images for <code>train</code> set, <code>4,585</code> images for <code>test</code> set and <code>2,290</code> images for <code>valid</code> set.</li>
    </ul>

    ''', unsafe_allow_html=True)
    st.markdown('''<br>''', unsafe_allow_html=True)
    markdown_table_1 = """
    | Class ID | Food Names                               |
    |----------|------------------------------------------|
    | 0        | Bánh canh (Vietnamese thick noodle soup) |
    | 1        | Bánh chưng (Square sticky rice cake)     |
    | 2        | Bánh cuốn (Rolled rice pancake)          |
    | 3        | Bánh khọt (Mini savory pancakes)         |
    | 4        | Bánh mì (Vietnamese baguette sandwich)   |
    | 5        | Bánh tráng (Rice paper)                  |
    | 6        | Bánh tráng trộn (Rice paper salad)       |
    | 7        | Bánh xèo (Vietnamese sizzling pancake)   |
    | 8        | Bò kho (Beef stew)                       |
    | 9        | Bò lá lốt (Grilled beef wrapped in betel leaves) |
    | 10       | Bông cải (Cauliflower)                   |
    | 11       | Bún (Rice vermicelli)                    |
    | 12       | Bún bò Huế (Spicy beef noodle soup)      |
    | 13       | Bún chả (Grilled pork with vermicelli)   |
    | 14       | Bún đậu (Vermicelli with tofu)           |
    | 15       | Bún mắm (Fermented fish noodle soup)     |
    | 16       | Bún riêu (Crab noodle soup)              |
    | 17       | Cá (Fish)                                |
    | 18       | Cà chua (Tomato)                         |
    | 19       | Cà pháo (Pickled eggplant)               |
    | 20       | Cà rốt (Carrot)                          |
    | 21       | Canh (Soup)                              |
    | 22       | Chả (Vietnamese pork roll)               |
    | 23       | Chả giò (Spring rolls)                   |
    | 24       | Chanh (Lime)                             |
    | 25       | Cơm (Rice)                               |
    """

    markdown_table_2 = """
    | Class ID | Food Names                               |
    |----------|------------------------------------------|
    | 26       | Cơm tấm (Broken rice)                    |
    | 27       | Con người (Human)                        |
    | 28       | Củ kiệu (Pickled scallion head)          |
    | 29       | Cua (Crab)                               |
    | 30       | Đậu hũ (Tofu)                            |
    | 31       | Dưa chua (Pickled vegetables)            |
    | 32       | Dưa leo (Cucumber)                       |
    | 33       | Gỏi cuốn (Fresh spring rolls)            |
    | 34       | Hamburger                                |
    | 35       | Heo quay (Roast pork)                    |
    | 36       | Hủ tiếu (Clear rice noodle soup)         |
    | 37       | Khổ qua thịt (Stuffed bitter melon soup) |
    | 38       | Khoai tây chiên (French fries)           |
    | 39       | Lẩu (Hotpot)                             |
    | 40       | Lòng heo (Pork offal)                    |
    | 41       | Mì (Egg noodles)                         |
    | 42       | Mực (Squid)                              |
    | 43       | Nấm (Mushroom)                           |
    | 44       | Ốc (Snails)                              |
    | 45       | Ớt chuông (Bell pepper)                  |
    | 46       | Phở (Vietnamese noodle soup)             |
    | 47       | Phô mai (Cheese)                         |
    | 48       | Rau (Vegetables)                         |
    | 49       | Salad (Salad)                            |
    | 50       | Thịt bò (Beef)                           |
    | 51       | Thịt gà (Chicken)                        |
    | 52       | Thịt heo (Pork)                          |
    | 53       | Thịt kho (Braised pork)                  |
    | 54       | Thịt nướng (Grilled meat)                |
    | 55       | Tôm (Shrimp)                             |
    | 56       | Trứng (Egg)                              |
    | 57       | Xôi (Sticky rice)                        |
    """


    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(markdown_table_1)

    with col2:
        st.markdown(markdown_table_2)
    st.divider()
    st.markdown('''
    <h4 class="dataset-page">🔍 Data Gathering 🔍</h4>
    <p class="define dataset-page">These pictures were collected from different sources to ensure its variety and complexity.</p>
    <ul class="define dataset-page">
        <li class="define-li dataset-page"><code>Google, Facebook, Shopee Food</code>: Most of the images were gathered from these platforms by searching the dish name with some keyword like "food review" or "cooking".</li>
        <li class="define-li dataset-page"><code>Youtube</code>: Frames from the video or shorts were extracted with the help from the <a href="https://roboflow.com/" target="_blank">Roboflow</a> annotation tools.</li>
        <li class="define-li dataset-page"><code>Personal Collection</code>: Some images were personally taken by using a smartphone to simulate the real-world situation of food detection.</li>
    </ul>
    ''', unsafe_allow_html=True)

    st.divider()
    st.markdown('''
    <h4 class="dataset-page">✍️ Data Annotation ✍️</h4>
    <p class="define dataset-page">The bounding box annotation and labeling process was done by using <a href="https://roboflow.com/" target="_blank">Roboflow</a> tools. To speed up the process, a YOLOv10m model 
    was trained on a subset of the dataset and used for the <code>Auto Label</code> feature to help automatically annotate the remaining images before double-checking it manually.</p>
    ''', unsafe_allow_html=True)

    st.divider()
    st.markdown('''
    <h4 class="dataset-page">⚙️ Data Processing ⚙️</h4>
    <p class="define dataset-page">Some augmentation techniques were used to make sure the model can generalize well and to resolve the imbalance volume between classes.</p>
    <ul class="define dataset-page">
        <li class="define-li dataset-page"><code>Bounding box cropping</code>: Minimum zoom of <code>5%</code> and a maximum of <code>20%</code>.</li>
        <li class="define-li dataset-page"><code>Bounding box flip</code>: Flip vertically.</li>
        <li class="define-li dataset-page"><code>Brightness adjustments</code>: Between <code>-15%</code> and <code>+15%</code>.</li>
        <li class="define-li dataset-page"><code>Mosaic augmentation</code></li>
    </ul>
    <p class="define dataset-page">Overall, the total images obtained for training the model after the augmentation process are 66,593 images.</p>
    ''', unsafe_allow_html=True)

# Nav bar
def navbar(active_page):
    return f"""
   
    <div class="custom-navbar">
        <div class="nav-items">
            <a href="/main" target="_self" class="nav-item {'active' if active_page == 'Home' else ''}">🏠 Home</a>
            <a href="#" target="_self" class="nav-item {'active' if active_page == 'About' else ''}">📄 About</a>
        </div>
        <a href="https://github.com" target="_blank" class="nav-item github-icon">
            <!-- GitHub SVG icon here -->
        </a>
    </div>
    """

def home_page():
    st.markdown(navbar('Home'), unsafe_allow_html=True)
    

def about_page():
    st.markdown(navbar('About'), unsafe_allow_html=True)
    
def styling_css():
    with open('./assets/css/general-style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    styling_css()
    query_params = st.query_params
    path = query_params.get("page", ["home"])[0].lower()
    
    # Determine the active page
    active_page = 'About' if path == "about" else 'Home'
    
    # Always render the navbar with the correct active page
    st.markdown(navbar(active_page), unsafe_allow_html=True)
    
    if path == "about":
        st.markdown('<h1 style="color: white; font-size: 40px;">About Section</h1>', unsafe_allow_html=True)
        st.write("This is the About section. Here you can add information about your project or organization.")
    else:
        render_content()
    
if __name__ == '__main__':
    main()
        