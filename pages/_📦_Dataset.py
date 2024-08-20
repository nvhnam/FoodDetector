import streamlit as st

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":microscope:"
)

st.title(":package: Dataset")
st.divider()
st.markdown('''
#### VietFood57: A Dataset for Vietnamese Food Detection.
This dataset contains `22,920` images with `58` classes which included an extra class for recognizing human faces as the purpose of this research is to detect and monitor people eating activity so being able to know the human existence during the detection can give a more wholesome result. After all, the eating duration can also be derived from human detection along with the dishes.

VietFood57 is divided in `70%`/`20%`/`10%` with `16,045` images for ***train*** set, `4,585` images for ***test*** set and `2,290` images for ***valid*** set.
''')
markdown_table_1 = """
| Class ID | Food Names                               |
|----------|------------------------------------------|
| 0        | Banh_canh (Vietnamese thick noodle soup) |
| 1        | Banh_chung (Square sticky rice cake)     |
| 2        | Banh_cuon (Rolled rice pancake)          |
| 3        | Banh_khot (Mini savory pancakes)         |
| 4        | Banh_mi (Vietnamese baguette sandwich)   |
| 5        | Banh_trang (Rice paper)                  |
| 6        | Banh_trang_tron (Rice paper salad)       |
| 7        | Banh_xeo (Vietnamese sizzling pancake)   |
| 8        | Bo_kho (Beef stew)                       |
| 9        | Bo_la_lot (Grilled beef wrapped in betel leaves) |
| 10       | Bong_cai (Cauliflower)                   |
| 11       | Bun (Rice vermicelli)                    |
| 12       | Bun_bo_Hue (Spicy beef noodle soup)      |
| 13       | Bun_cha (Grilled pork with vermicelli)   |
| 14       | Bun_dau (Vermicelli with tofu)           |
| 15       | Bun_mam (Fermented fish noodle soup)     |
| 16       | Bun_rieu (Crab noodle soup)              |
| 17       | Ca (Fish)                                |
| 18       | Ca_chua (Tomato)                         |
| 19       | Ca_phao (Pickled eggplant)               |
| 20       | Ca_rot (Carrot)                          |
| 21       | Canh (Soup)                              |
| 22       | Cha (Vietnamese pork roll)               |
| 23       | Cha_gio (Spring rolls)                   |
| 24       | Chanh (Lime)                             |
| 25       | Com (Rice)                               |
"""

markdown_table_2 = """
| Class ID | Food Names                               |
|----------|------------------------------------------|
| 26       | Com_tam (Broken rice)                    |
| 27       | Con_nguoi (Human)                        |
| 28       | Cu_kieu (Pickled scallion head)          |
| 29       | Cua (Crab)                               |
| 30       | Dau_hu (Tofu)                            |
| 31       | Dua_chua (Pickled vegetables)            |
| 32       | Dua_leo (Cucumber)                       |
| 33       | Goi_cuon (Fresh spring rolls)            |
| 34       | Hamburger                                |
| 35       | Heo_quay (Roast pork)                    |
| 36       | Hu_tieu (Clear rice noodle soup)         |
| 37       | Kho_qua_thit (Stuffed bitter melon soup) |
| 38       | Khoai_tay_chien (French fries)           |
| 39       | Lau (Hotpot)                             |
| 40       | Long_heo (Pork offal)                    |
| 41       | Mi (Egg noodles)                         |
| 42       | Muc (Squid)                              |
| 43       | Nam (Mushroom)                           |
| 44       | Oc (Snails)                              |
| 45       | Ot_chuong (Bell pepper)                  |
| 46       | Pho (Vietnamese noodle soup)             |
| 47       | Pho_mai (Cheese)                         |
| 48       | Rau (Vegetables)                         |
| 49       | Salad (Salad)                            |
| 50       | Thit_bo (Beef)                           |
| 51       | Thit_ga (Chicken)                        |
| 52       | Thit_heo (Pork)                          |
| 53       | Thit_kho (Braised pork)                  |
| 54       | Thit_nuong (Grilled meat)                |
| 55       | Tom (Shrimp)                             |
| 56       | Trung (Egg)                              |
| 57       | Xoi (Sticky rice)                        |
"""


col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(markdown_table_1)

with col2:
    st.markdown(markdown_table_2)
st.divider()
st.markdown('''
#### Data Gathering
These pictures were collected from different sources to ensure its variety and complexity.
- `Google`, `Facebook`, `Shopee Food`: most of the images were gathered from these plaforms by searching the dish name with some keyword like "food review" or "cooking".
- `Youtube`: frames from the video or shorts were extracted with the help from the [Roboflow](https://roboflow.com/) annotation tools.
- `Personal Collection`: some images were personally taken by using smart phone to simulate the real-world situation of food detection.
#### Data Annotation
The bounding box annotation and labeling process was done by using [Roboflow](https://roboflow.com/) tools. To speed up the process, a YOLOv10m model was trained on a subset of the dataset was used for the `Auto Label` feature to help automatically annotate the remaining images before double-checking it manually.
#### Data Processing
Some augmentation techniques were used to make sure the model can generalize well and to resolve the imbalance volume between classes. 
- `Bounding box cropping`: minimum zoom of `5%` and a maxium of `20%`.
- `Bounding box flip`: flip vertically.
- `Brightness adjustments`: between `-15%` and `+15%`.
- `Mosaic augmentation`
Overall, the total images obtain for training model after the augmentation process are 31,495 images. 
''')