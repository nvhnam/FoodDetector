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
Overall, the total images obtain for training model after the augmentation process are 66,593 images. 
''')