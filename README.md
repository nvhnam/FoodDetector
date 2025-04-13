# 🕵️‍♂️ FoodDetector

**FoodDetector** is a real-time web-based application for detecting and recognizing Vietnamese dishes using a custom YOLOv10 model trained on the VietFood67 - the **largest** Vietnamese food dataset. This system empowers users with instant nutritional feedback, aiding in dietary awareness and health-conscious decision-making.

---

## ✨ Project Highlights

- Real-time Vietnamese dish detection via images, videos, webcam input and IP camera (RTSP).
- Nutritional breakdown for each detected dish: Calories, Fat, Saturates, Sugar, Salt.
- Traffic light system for nutrient awareness.
- Powered by a custom-trained YOLOv10 model on our largest Vietnamese food image dataset [VietFood67 dataset](https://www.kaggle.com/datasets/thomasnguyen6868/vietfood68).
- Developed using **Python**, **Streamlit**, and **OpenCV**.

---

## 📚 Publications

This project has been the foundation of two research papers:

1. **"Now I Know What I am Eating: Real-time Tracking and Nutritional Insights Using VietFood67 to Enhance User Experience"**  
   🏆 *Best Paper Runner-up Award at SOICT 2024*  
   🔗 [View Paper](https://drive.google.com/file/d/19FcdIjc2kdT4ocdUMtztcZSqOx39qTUe/view)
   
2. **"It’s Yummy: Real-Time Detection and Recognition of Vietnamese Dishes"**  
   📌 *Presented at ICCIT 2024, British University Vietnam (BUV)*  
   🔗 [View Paper](https://drive.google.com/file/d/15oVhSYscpNW5pSEFjiXLNHrGNyDi6vjt/view) 

---

## 👨‍💻 Contributors

- **Nguyen Viet Hoang Nam** (Project Lead, Web Developer, YOLOv10 Trainer, VietFood Dataset Gathering)  
- **Tran Bao Tu** (UI/UX Designer, Poster, Slides Creator)  
- **Ton That Minh Vu** (Dataset Gathering)  
- **Dr. Vi Chi Thanh** (Research Supervisor & Guidance)

---

## 🧠 Technologies Used

| Area             | Tech Stack                        |
|------------------|-----------------------------------|
| Model Training   | Python, YOLOv10                   |
| Deployment       | Streamlit                         |
| Nutritional Data | Custom JSON + Traffic Light System|
| Visualization    | Matplotlib, Streamlit Components  |

---

## 📁 Dataset

We created and released the **VietFood67** dataset for training and evaluation, containing 68 classes (an extra class for human face detection) and 33k images of common Vietnamese dishes with annotated bounding boxes.

📦 [View VietFood67 on Kaggle](https://www.kaggle.com/datasets/thomasnguyen6868/vietfood68)

---

## ⭐ Support This Project

If you find **FoodDetector** or the **VietFood67** dataset helpful in your research or projects:

- 🌟 Please consider giving this repository a **star** on [GitHub](https://github.com/nvhnam/FoodDetector).
- 📊 Star the [VietFood67 dataset on Kaggle](https://www.kaggle.com/datasets/thomasnguyen6868/vietfood68) to show your support.
- 📄 **Cite our papers** in your publications to help us continue our research and development.

> 🆓 The FoodDetector and **VietFood67** dataset are free to use for research and educational purposes **with proper citation**. Commercial use or redistribution is **not permitted**.

---

## 🚀 Features

- Upload or stream food media (image, video, webcam, IP camera via RTSP).
- Real-time detection with bounding boxes and labels.
- Nutritional values shown per dish and total per meal.
- User-friendly nutrient traffic light indicators.
- Designed for low-resource environments (runs without GPU).

---

## 🛠️ Getting Started

> 🚀 **Latest Version:** Please use the [`v2` branch](https://github.com/nvhnam/FoodDetector/tree/v2) before proceeding, as it includes all the newest features and improvements.

### Requirements
- Python 3.8+
- Streamlit
- OpenCV
- ONNX Runtime
- Pandas, Numpy, etc.

### Run Locally
<pre>
git clone https://github.com/nvhnam/FoodDetector.git
cd FoodDetector
git checkout v2 
pip install -r requirements.txt
streamlit run app.py
</pre>
   
---

## 📈 Future Work

- 📱 **Mobile App with AR**: Building a mobile version featuring AR overlays that display 3D real-time nutrient values directly on detected dishes.  
  > 🔍 *Currently seeking passionate collaborators with experience in **Unity** and **AR development** to bring this vision to life!*

- 🧠 Integration with AI nutritionist agents (CrewAI, LangChain) for personalized meal recommendations.
  > 🔍 *Research is currently in progress.*

- 🏥 Real-time health feedback based on user demographics (age, gender, height, weight, eating patterns).
- 🍲 Expand the VietFood67 dataset with more regional Vietnamese dishes for greater diversity and recognition accuracy.


---

📩 Contact

For questions or collaborations:

- 📧 Email: nvhnam01@gmail.com
- 📝 LinkedIn: https://www.linkedin.com/in/nvhnam01/
