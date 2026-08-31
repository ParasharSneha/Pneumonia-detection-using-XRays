# 🫁 Pneumonia Detection using Chest X-Rays

A deep learning web application that classifies chest X-ray images as **Normal** or **Pneumonia**, built with a Convolutional Neural Network (CNN) and deployed as an interactive **Streamlit** web app with user authentication and patient record-keeping.

> ⚠️ **Disclaimer:** This project is built for educational purposes only. It is **not** a certified medical diagnostic tool and must not be used for real clinical decisions.

---

## 📌 Overview

The app allows a user to:
1. Sign up / log in (with password reset & "forgot password" flows).
2. Fill in a patient's basic details (name, age, mobile number).
3. Upload a chest X-ray image.
4. Get an instant prediction — **Normal** or **Pneumonia** — from a trained CNN model.

The underlying model was trained on the **Chest X-Ray Images (Pneumonia)** dataset (the widely-used Kermany et al. dataset, ~5,800 images across `NORMAL` and `PNEUMONIA` classes).

---

## ✨ Features

- **CNN-based image classifier** — 5 convolutional blocks with batch normalization, dropout, and max-pooling.
- **Data augmentation** (rotation, zoom, shift, horizontal flip) via `ImageDataGenerator` to reduce overfitting.
- **User authentication system** — sign up, login, forgot password, and reset password, backed by a NoSQL database (Deta Base) with `bcrypt`-hashed passwords.
- **Patient record management** — stores patient details and uploaded X-ray images for each session.
- **Email notifications** — sends a newly generated password to the user's email via SMTP.
- **Interactive UI** — built with Streamlit, including Lottie animations for a friendlier interface.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras (CNN) |
| Image processing | OpenCV |
| Web app / UI | Streamlit, Streamlit-Authenticator, Streamlit-Lottie |
| Database | Deta Base (NoSQL, cloud) |
| Auth & security | bcrypt |
| Notifications | smtplib (SMTP email) |
| Experimentation | Jupyter Notebook (Google Colab) |

---

## 🧠 Model Architecture & Training

- **Input:** 200×200 grayscale X-ray images.
- **Architecture:** 5 `Conv2D` blocks (32 → 64 → 64 → 128 → 256 filters) each followed by `BatchNormalization` and `MaxPooling2D`, with `Dropout` for regularization, then a `Flatten` → `Dense(128, relu)` → `Dense(1, sigmoid)` output layer for binary classification.
- **Optimizer:** RMSprop | **Loss:** Binary Crossentropy | **Epochs:** 15
- **Callback:** `ReduceLROnPlateau` to lower the learning rate when validation accuracy plateaus.
- **Data split:** Pre-defined `train` / `val` / `test` folders from the source dataset.

### 📊 Results (on the test set, 624 images)

| Metric | Value |
|---|---|
| Test Accuracy | **90.4%** |
| Precision (Pneumonia) | 0.92 |
| Recall (Pneumonia) | 0.92 |
| F1-score (Pneumonia) | 0.92 |
| Precision (Normal) | 0.87 |
| Recall (Normal) | 0.87 |

**Confusion Matrix**

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | 204 | 30 |
| **Actual Pneumonia** | 30 | 360 |

---

## 📁 Project Structure

```
Pneumonia-detection-using-XRays/
├── main.py                              # Streamlit app entry point (UI + auth flow)
├── dependancies.py                      # Auth, DB, email, and prediction helper functions
├── model.h5                             # Trained CNN model (Keras/TensorFlow)
├── project_Pneumonia_updated_final.ipynb # Model training & evaluation notebook
├── requirements.txt                     # Python dependencies
├── lottie_images/                       # UI animation assets
└── patient_images/                      # Uploaded X-ray images (created at runtime)
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ParasharSneha/Pneumonia-detection-using-XRays.git
   cd Pneumonia-detection-using-XRays
   ```

2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up your own credentials** (do **not** commit real secrets — see Known Issues below):
   - A [Deta](https://deta.space/) project key for the database.
   - SMTP email credentials for the password-reset feature.
   - Store these as environment variables rather than hardcoding them in `dependancies.py`.

4. **Run the app**
   ```bash
   streamlit run main.py
   ```

---

## 🚧 Known Issues & Limitations

- **Secrets are hardcoded as placeholders** in `dependancies.py` (DB key, SMTP email/password) — these must be moved to environment variables or a `.env` file before any real deployment.
- **Deta Base is deprecated** (Deta Space was sunset in 2024); the database layer needs to be migrated to an actively maintained service (Firebase, Supabase, MongoDB Atlas, PostgreSQL, etc.).
- **Very small validation set** (only 16 images in the original dataset's `val` folder), which makes validation accuracy during training noisy and not fully reliable.
- **Class imbalance** in the training data (more Pneumonia than Normal images) is not explicitly corrected with class weights.
- **Fixed 0.5 classification threshold** — not tuned for a medical use case, where minimizing false negatives (missed Pneumonia cases) is usually more important than raw accuracy.
- **No ROC-AUC or precision-recall curve analysis**, which is standard for evaluating medical classifiers, especially under class imbalance.
- Patient data (name, age, mobile number) is stored without explicit anonymization or encryption at rest.

---

## 🔮 Future Improvements

- Move all secrets/config to environment variables (`python-dotenv`).
- Add Grad-CAM visualizations so predictions are explainable (highlighting the lung regions influencing the model's decision).
- Track experiments with MLflow or Weights & Biases instead of only notebook outputs.
- Add unit tests for the validation and preprocessing functions.
- Package the training pipeline as a standalone script (`train.py`) instead of only a notebook, for reproducibility.
- Containerize the app with Docker for consistent deployment.
- Replace Deta Base with a production-grade, actively maintained database.

---

## 📜 License

This project is for educational purposes. Add a license (e.g., MIT) if you intend to share it publicly for reuse.

## 🙋 Author

Sneha Parashar — [GitHub](https://github.com/ParasharSneha)
