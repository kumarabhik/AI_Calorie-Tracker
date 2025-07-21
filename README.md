
# 🍽️ AI Calorie Tracker

**AI Calorie Tracker** is a Streamlit-based web app that lets users upload food images and receive estimated calorie values using computer vision and large language models (LLMs). It combines image captioning, food classification, calorie lookup from a real-world dataset, and LLM fallback (Groq + LLaMA 3).

---

## 🚀 Features

- 📸 **Upload any food image**
- 🧠 **Image captioning** using **BLIP** (via Hugging Face Transformers)
- 🍴 **Food classification** using **ViT** model trained on **Food101**
- 📊 **Calorie estimation** from a real-world dataset: **OpenFoodFacts**
- 🤖 **Fallback with LLM (Groq)** if calorie info is missing in dataset
- 🎨 Responsive **Streamlit UI** with dark/light theme

---

## 🧠 Models & Tools Used

### 🔍 1. Image Captioning
- **Model**: [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
- **Task**: Generates descriptive captions from food images for optional UX feedback.
- **Library**: Hugging Face Transformers

### 🍽️ 2. Food Classification
- **Model**: [`nateraw/food`](https://huggingface.co/nateraw/food)
- **Dataset**: Trained on **Food101**
- **Task**: Predicts top food labels (e.g., `chicken_wings`) with probabilities.
- **Threshold**: Only predictions with confidence ≥ 10% are used.

### 📊 3. Calorie Lookup (Option A)
- **Source**: [Open Food Facts](https://openfoodfacts.org)  
- **Dataset**: `en.openfoodfacts.org.products.csv` (local copy used for speed)
- **Info Used**:
  - `product_name`
  - `energy-kcal_100g`

### 🤖 4. LLM Fallback with Groq (Option B)
- **LLM**: `llama3-8b-8192` via [Groq API](https://console.groq.com/)
- **Use Case**: When food label (e.g., `biryani`) isn’t found in dataset, fallback to Groq for calorie estimate.

---

## 📁 Directory Structure

```
📦 AI-Calorie-Tracker/
├── CalorieAI_Simple.py            # Main Streamlit app
├── en.openfoodfacts.org.products.csv/   # Local CSV folder
│   └── en.openfoodfacts.org.products.csv
├── requirements.txt
└── README.md
```

---

## 🔐 Hugging Face Token

If you're running the app locally and facing model download/auth errors, create a Hugging Face access token:

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Generate a **Read** token
3. Run this command before launching the app:
   ```bash
   huggingface-cli login
   ```

---

## 🔑 Groq API Key

1. Sign up at [https://console.groq.com](https://console.groq.com)
2. Create a new API Key
3. Save your key in `.streamlit/secrets.toml` like this:

```toml
[groq]
api_key = "your_groq_api_key_here"
```

---

## 📦 Installation

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/AI-Calorie-Tracker.git
cd AI-Calorie-Tracker
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run CalorieAI_Simple.py
```

---

## 🧾 Requirements

Here’s what your `requirements.txt` might look like:

```
streamlit
transformers
torch
pillow
pandas
requests
```

---

## 🛠️ How It Works

1. **Upload** an image of food.
2. BLIP **generates a caption** (e.g., "a plate of chicken wings and fries").
3. ViT **classifies top 3 food labels** from the image (e.g., `chicken_wings`, `onion_rings`).
4. App **filters labels ≥ 10% confidence**.
5. App tries to **match labels with OpenFoodFacts dataset** to get `energy-kcal_100g`.
6. If a match is not found:
   - Calls **Groq LLM** (e.g., *"Estimate calories per 100g of biryani"*) to generate fallback value.

---

## 📸 Example

| Uploaded Image | Prediction | Calories |
|----------------|------------|----------|
| 🍗 Chicken Wings | `chicken_wings` (73%) | 211 kcal/100g (dataset) |
| 🍛 Biryani | `biryani` (via LLM) | 276 kcal/100g (Groq) |

---

## 💡 Improvements to Add

- Portion size estimation via object detection
- OCR for text on food packaging
- Nutrient breakdown (fats, carbs, protein)
- Voice-based search

---

## 🧑‍💻 Author

Made by **Abhishek Kumar**  
Inspired by real-world health tracking use cases using AI/LLMs.  
🧠 Powered by Hugging Face, Groq, and Open Food Facts.
