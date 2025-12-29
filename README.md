# 🖼️ Sketch → Photo Retrieval (Streamlit App)

This project allows you to upload a **face sketch** and retrieve the most similar
**face photos** using a trained deep learning model.

The system is already trained — **no training is required**.

---

## 🚀 What This Project Does

- Takes a **sketch image** as input
- Converts it into a feature embedding
- Compares it with a gallery of photo embeddings
- Retrieves the **Top-K most similar photos**
- Runs as a **Streamlit web application**

---

## 📂 Project Structure

Final702/
├── app.py
├── requirements.txt
├── sketch_photo_triplet_model.pth
├── README.md
├── .gitignore
├── data/
│   └── photos/
└── extras/

---

## ⚙️ How to Run

git clone https://github.com/KSPandian7/Final702.git
cd Final702
pip install -r requirements.txt
streamlit run app.py

Open:
http://localhost:8501

---

## 🖼️ How to Use

1. Upload a face sketch image
2. Select Top-K results
3. View retrieved photos

---

## 🧠 Model Information

- Metric Learning with Triplet Loss
- Embedding Dimension: 128
- Distance Metric: Euclidean

---

## 📌 Notes

- sketch_photo_triplet_model.pth must be present in the root folder
- Gallery images must be inside data/photos/
- extras/ contains notebooks and scripts not required to run the app

---

## 👨‍💻 Author

Kulasekarapandian (KSP)

---

## 📜 License

Academic and educational use only
