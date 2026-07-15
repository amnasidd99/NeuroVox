# 🧠 NeuroVox — AI-Powered Brain MRI Tumor Analysis

An end-to-end deep learning web app that analyzes brain MRI scans using **PyTorch** and **ResNet-18**. Built to explore computer vision and explainable AI in medical imaging.

The web app lets users:

- Upload a brain MRI image
- Receive an AI-generated tumor classification (**glioma · meningioma · pituitary tumor · no tumor**)
- Visualize model attention using **Grad-CAM** heatmaps
- Adjust the decision threshold
- View model performance metrics
- Download a research report for each analysis

> **Disclaimer:** This application is an educational research prototype demonstrating an end-to-end ML workflow and explainable AI in medical imaging. It is **not** intended for medical diagnosis or clinical decision-making.

---

## 🖼️ Tech stack

Python · PyTorch · Torchvision · Streamlit · ResNet-18 (transfer learning) · Grad-CAM · NumPy · Pillow

---

## 📁 Project structure

```
NeuroVox/
├── app.py                        # Streamlit web app
├── model.py                      # ResNet-18 definition + inference
├── gradcam.py                    # Grad-CAM explainability
├── report.py                     # Per-analysis report generator
├── requirements.txt
├── train_neurovision_colab.ipynb # Colab notebook to train the model
├── assets/metrics.json           # Model metrics (updated after training)
├── weights/best_model.pth        # Trained weights (you add this after training)
└── .streamlit/config.toml        # Theme
```

The app runs **without** trained weights using a random placeholder model (so you can see the UI immediately), and shows a warning. Add real weights for real predictions.

---

## 🚀 Quickstart (run locally)

```bash
git clone https://github.com/<your-username>/NeuroVox.git
cd NeuroVox
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

---

## 🧪 Train the model (free, on Google Colab)

1. Open `train_neurovision_colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set **Runtime → Change runtime type → GPU**.
3. Run all cells. You'll upload your `kaggle.json` to download the
   [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
4. Training takes ~10–15 min and reaches ~97% accuracy.
5. Download `best_model.pth` and `metrics.json`, then place them in:
   - `weights/best_model.pth`
   - `assets/metrics.json`
6. Commit and push. Done.

---

## ☁️ Deploy the live demo (free — Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick your repo, set the main file to `app.py`, and deploy.
4. You'll get a public URL like `https://neurovision-ai.streamlit.app` — use it as your **Live Demo** link.

> **Note on model size:** ResNet-18 weights (~45 MB) can be committed directly, or use [Git LFS](https://git-lfs.com/) / a GitHub Release asset if you prefer. If you exclude the weights, the deployed app still runs in placeholder mode.

---

## 📊 Results (example)

| Metric | Value |
| --- | --- |
| Accuracy | 0.972 |
| Macro F1 | 0.968 |
| ROC-AUC (macro) | 0.995 |

*(These update automatically from `assets/metrics.json` after you train.)*

---

## 📝 What I learned

Transfer learning, computer vision, explainable AI (Grad-CAM), model evaluation (accuracy / F1 / ROC-AUC), Git/GitHub, and deploying machine learning applications with Streamlit.

---

## 📄 License

MIT — for educational and research use only.
