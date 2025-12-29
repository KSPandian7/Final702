import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
import numpy as np

# --------------------------------------------------
# PAGE CONFIG + THEME
# --------------------------------------------------
st.set_page_config(
    page_title="Sketch → Photo Retrieval",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for colors & UI polish
st.markdown("""
<style>
body {
    background-color: #f6f8fa;
}
.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #1f4fd8;
}
.sub-title {
    font-size: 18px;
    color: #444;
}
.card {
    background-color: white;
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}
.rank-text {
    font-size: 14px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# MODEL DEFINITION
# --------------------------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 31 * 25, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = EmbeddingNet(128).to(device)
    model.load_state_dict(
        torch.load("sketch_photo_triplet_model.pth",
                   map_location=device)
    )
    model.eval()
    return model

model = load_model()

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((250, 200)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --------------------------------------------------
# LOAD PHOTO GALLERY
# --------------------------------------------------
@st.cache_data
def load_gallery():
    photo_dir = "data/photos"
    images = []
    names = []

    for file in sorted(os.listdir(photo_dir)):
        img = Image.open(os.path.join(photo_dir, file)).convert("L")
        images.append(img)
        names.append(file)

    return images, names

gallery_images, gallery_names = load_gallery()

@st.cache_data
def build_gallery_embeddings(images):
    embeds = []
    with torch.no_grad():
        for img in images:
            t = transform(img).unsqueeze(0).to(device)
            embeds.append(model(t).cpu())
    return torch.cat(embeds)

gallery_embeddings = build_gallery_embeddings(gallery_images)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**Sketch → Photo Retrieval System**

• Metric Learning  
• Triplet Loss  
• Euclidean Distance  
• Rank-K Retrieval  

📌 Upload a face sketch  
📌 System retrieves most similar photos  

**Built by:** KSP  
""")

top_k = st.sidebar.slider("Top-K Results", 1, 5, 5)

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.markdown("<div class='main-title'>🖼️ Sketch → Photo Retrieval</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload a face sketch and retrieve the most similar photos</div>", unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader(
    "📤 Upload Sketch Image",
    type=["jpg", "png", "jpeg"]
)

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
if uploaded_file is not None:
    sketch_img = Image.open(uploaded_file).convert("L")

    st.markdown("### 📝 Uploaded Sketch")
    st.image(sketch_img, width=220)

    with torch.no_grad():
        sketch_tensor = transform(sketch_img).unsqueeze(0).to(device)
        sketch_emb = model(sketch_tensor).cpu()

        distances = torch.cdist(sketch_emb, gallery_embeddings)
        top_vals, top_idx = torch.topk(
            distances[0], top_k, largest=False
        )

    st.markdown("### 🔍 Retrieved Photos")

    cols = st.columns(top_k)
    for i, idx in enumerate(top_idx):
        with cols[i]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(
                gallery_images[idx],
                width=160
            )
            st.markdown(
                f"<div class='rank-text'><b>Rank {i+1}</b><br>"
                f"{gallery_names[idx]}<br>"
                f"Distance: {top_vals[i]:.3f}</div>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<center style='color:#777;'>Metric Learning • Triplet Loss • Streamlit Deployment</center>",
    unsafe_allow_html=True
)
