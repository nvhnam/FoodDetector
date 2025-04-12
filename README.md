# 🍜 FoodDetector

**FoodDetector** is a real-time web-based application for detecting and recognizing Vietnamese dishes using a custom YOLOv10 model trained on the VietFood67 - the **largest** Vietnamese food dataset. This system empowers users with instant nutritional feedback, aiding in dietary awareness and health-conscious decision-making.

---

## ✨ Project Highlights

- Real-time Vietnamese dish detection via images, videos, webcam input and IP camera (RTSP).
- Nutritional breakdown for each detected dish: Calories, Fat, Saturates, Sugar, Salt.
- Traffic light system for nutrient awareness.
- Powered by a custom-trained YOLOv10 model on our largest Vietnamese food image dataset [VietFood67 dataset](https://www.kaggle.com/).
- Developed using **Python**, **Streamlit**, and **OpenCV**.

---

## 📚 Publications

This project has been the foundation of two research papers:

1. **"It’s Yummy: Real-Time Detection and Recognition of Vietnamese Dishes"**  
   📌 *Presented at ICCIT 2024, British University Vietnam (BUV)*  
   🔗 [View Paper](https://drive.google.com/file/d/15oVhSYscpNW5pSEFjiXLNHrGNyDi6vjt/view) 

2. **"Now I Know What I am Eating: Real-time Tracking and Nutritional Insights Using VietFood67 to Enhance User Experience"**  
   🏆 *Best Paper Runner-up Award at SOICT 2024*  
   🔗 [View Paper](https://drive.google.com/file/d/19FcdIjc2kdT4ocdUMtztcZSqOx39qTUe/view) 

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

We created and released the **VietFood67** dataset for training and evaluation, containing 67 classes and 33k images of common Vietnamese dishes with annotated bounding boxes.

📦 [View VietFood67 on Kaggle](#) (currently uploading the dataset, please wait as the zip file is 26 GB)

---

## ⭐ Support This Project

If you find **FoodDetector** or the **VietFood67** dataset helpful in your research or projects:

- 🌟 Please consider giving this repository a **star** on [GitHub](https://github.com/nvhnam/FoodDetector).
- 📊 Star the [VietFood67 dataset on Kaggle](#) to show your support.
- 📄 **Cite our papers** in your publications to help us continue our research and development.

> 🆓 The **VietFood67** dataset is free to use for research and educational purposes **with proper citation**. Commercial use or redistribution is **not permitted**.

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
git clone https://github.com/yourusername/FoodDetector.git
cd FoodDetector
git checkout v2 
pip install -r requirements.txt
streamlit run app.py

---

## 📈 Future Work
- Mobile app version with AR overlay for 3D real-time nutrient values display. (Currently looking for collaborators to work on Unity)
- Integration with AI nutritionist agents (CrewAI, LangChain).
- Real-time user health feedback based on demographics.
- Expand dataset with more regional dishes.

---

📩 Contact

For questions or collaborations:

- 📧 Email: nvhnam01@gmail.com
- 📝 LinkedIn: https://www.linkedin.com/in/nvhnam01/
