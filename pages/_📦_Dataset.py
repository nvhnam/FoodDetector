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
| Class ID | Food Names            |
|----------|------------------------|
| 0        | Banh_canh              |
| 1        | Banh_chung             |
| 2        | Banh_cuon              |
| 3        | Banh_khot              |
| 4        | Banh_mi                |
| 5        | Banh_trang             |
| 6        | Banh_trang_tron        |
| 7        | Banh_xeo               |
| 8        | Bo_kho                 |
| 9        | Bo_la_lot              |
| 10       | Bong_cai               |
| 11       | Bun                    |
| 12       | Bun_bo_Hue             |
| 13       | Bun_cha                |
| 14       | Bun_dau                |
| 15       | Bun_mam                |
| 16       | Bun_rieu               |
| 17       | Ca                     |
| 18       | Ca_chua                |
| 19       | Ca_phao                |
| 20       | Ca_rot                 |
| 21       | Canh                   |
| 22       | Cha                    |
| 23       | Cha_gio                |
| 24       | Chanh                  |
| 25       | Com                    |
| 26       | Com_tam                |
| 27       | Con_nguoi              |
| 28       | Cu_kieu                |
"""

markdown_table_2 = """
| Class ID | Food Names            |
|----------|------------------------|
| 29       | Cua                    |
| 30       | Dau_hu                 |
| 31       | Dua_chua               |
| 32       | Dua_leo                |
| 33       | Goi_cuon               |
| 34       | Hamburger              |
| 35       | Heo_quay               |
| 36       | Hu_tieu                |
| 37       | Kho_qua_thit           |
| 38       | Khoai_tay_chien        |
| 39       | Lau                    |
| 40       | Long_heo               |
| 41       | Mi                     |
| 42       | Muc                    |
| 43       | Nam                    |
| 44       | Oc                     |
| 45       | Ot_chuong              |
| 46       | Pho                    |
| 47       | Pho_mai                |
| 48       | Rau                    |
| 49       | Salad                  |
| 50       | Thit_bo                |
| 51       | Thit_ga                |
| 52       | Thit_heo               |
| 53       | Thit_kho               |
| 54       | Thit_nuong             |
| 55       | Tom                    |
| 56       | Trung                  |
| 57       | Xoi                    |
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