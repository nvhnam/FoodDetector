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