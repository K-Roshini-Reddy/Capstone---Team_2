### 🩺 **Pneumonia Disease Detection Using Vision Transformers**  
 **AI-powered medical imaging system for pneumonia detection using deep learning!** 
![image](https://github.com/user-attachments/assets/49f772bf-9c4a-453a-994c-d002b8e2635e)

![image](https://github.com/user-attachments/assets/2a17fb5c-9c42-46e0-9b36-66d413d45c05)

   
 ---

## 🎥 Project Demo
[Watch the Demo Video Here](https://video.kent.edu/media/Capstone%20Project%20Demo/1_u6w5bck1)


---

## 🏥 **About the Project**  
This **capstone project** aims to develop a **deep-learning model** to detect pneumonia from chest X-rays using **Vision Transformers (ViT) and Convolutional Neural Networks (CNNs)**. The model achieves **90.65% accuracy**, significantly reducing false positives compared to traditional methods. The system is deployed using **Flask and AWS**, enabling real-time medical image analysis.

---

## 🚀 **Tech Stack**
| Technology  | Usage |
|------------|--------------------------------|
| **Programming Language** | Python |
| **Deep Learning Frameworks** | TensorFlow, PyTorch |
| **Computer Vision** | OpenCV |
| **Model Architecture** | Vision Transformers (ViT), CNN |
| **Backend** | Flask |
| **Deployment** | Flask API, Docker, Local Host|

---

## 🎯 **Key Features**
✅ **Automated Pneumonia Detection** – AI-powered image analysis  
✅ **Deep Learning Models** – CNNs & Vision Transformers  
✅ **High Accuracy** – 90.65% detection accuracy  
✅ **False-Positive Reduction** – 25% improvement over traditional models  
✅ **Real-Time Processing** – Flask backend for quick predictions  

---

##  **Installation & Setup**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/SampathKumarKolichalam/Pneumonia-Detection-ViT.git
cd Pneumonia-Detection-ViT
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Train the Deep Learning Model**
```sh
python train_model.py
```
*(Make sure to have a dataset of chest X-rays in the appropriate format.)*

### **4️⃣ Run the Flask Web Application**
```sh
python app.py
```
Now, open **http://127.0.0.1:5000/** in your browser. 🚀

### **5️⃣ Running with Docker (Optional)**
```sh
docker build -t pneumonia-detection .
docker run -p 5000:5000 pneumonia-detection
```
---

## 🔥 **Machine Learning Model**
### **1️⃣ Convolutional Neural Network (CNN)**
- Used for **feature extraction** and **spatial pattern recognition**  
- Pretrained models like **ResNet, EfficientNet** used for transfer learning  

### **2️⃣ Vision Transformers (ViT)**
- Used for **image classification** and **disease detection**  
- Improves accuracy by capturing **global dependencies** in images  

---

## 📊 **Data Preprocessing & Feature Engineering**
✅ **Chest X-ray Image Preprocessing** – Noise removal & augmentation  
✅ **Resizing & Normalization** – Standardizing images for model input  
✅ **Feature Extraction** – CNN-based automated feature learning  

---

## **API Endpoints**
### **1️⃣ Upload Chest X-ray for Prediction**
```sh
POST /predict
```
#### **Request Body (Multipart Form-Data):**
- **File:** `chest_xray.jpg`

#### **Response:**
```json
{
    "prediction": "Pneumonia Detected",
    "confidence": 90.65
}
```

---

## 🤝 **Contributing**
**Want to enhance this project?**  
Fork the repo, make your changes, and submit a pull request!  

```sh
git clone https://github.com/yourusername/Pneumonia-Detection-ViT.git
git checkout -b feature-branch
git commit -m "Added new feature"
git push origin feature-branch
```

---

## 📜 **License**
This project is free and access to all*. Feel free to use and modify it.  

---

## 👨‍💻 **Connect with Me**
📧 Email: [sampathkumarkolichalam@gmail.com]  

🔗 LinkedIn: [https://www.linkedin.com/in/sampath-kumar-kolichalam-18b57b1ab/]

---

