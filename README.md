# 🍽️ AI Calorie Tracker

**AI Calorie Tracker** is a **Streamlit-based web app** that lets users upload food images and receive **estimated calorie values** using computer vision and large language models (LLMs). It combines **image captioning**, **food classification**, **real-world calorie lookup**, and **LLM fallback (Groq + LLaMA 3)**.

---

# 🚀 Features

- 📸 Upload any food image
- 🧠 Image captioning using **BLIP** (via Hugging Face Transformers)
- 🍴 Food classification using **ViT model** trained on Food101
- 📊 Calorie estimation from **OpenFoodFacts**
- 🤖 Fallback to **Groq LLM** if food not found in dataset
- 🗓️ Weekly logging of meals (Breakfast, Lunch, Dinner)
- 📈 Weekly calorie summaries
- 🎨 Responsive **Streamlit UI** with theme support

---

# 🧠 Models & Tools Used

## 🔍 1. Image Captioning
- **Model**: `Salesforce/blip-image-captioning-base`
- **Task**: Generates descriptive captions from food images
- **Library**: Hugging Face Transformers

## 🍽️ 2. Food Classification
- **Model**: `nateraw/food`
- **Dataset**: Food101
- **Task**: Predicts food labels (e.g., `chicken_wings`) with probabilities
- **Threshold**: Only predictions with confidence ≥ **30%** are used

## 📊 3. Calorie Lookup (Option A)
- **Source**: OpenFoodFacts
- **Dataset**: `en.openfoodfacts.org.products.csv` (used **locally** for low latency)
- **Columns Used**:
  - `product_name`
  - `energy-kcal_100g`

## 🤖 4. LLM Fallback with Groq (Option B)
- **LLM**: `llama3-8b-8192` via Groq API
- **Use Case**: If label (e.g., "biryani") isn't in dataset, ask LLM for calorie estimate

---

# 📁 Directory Structure

```
📦 AI-Calorie-Tracker/
├── CalorieAI_Simple.py             # Main Streamlit app
├── en.openfoodfacts.org.products.csv/  # Local calorie dataset
│   └── en.openfoodfacts.org.products.csv
├── requirements.txt
└── README.md
```

---

# 🔐 Hugging Face Token

If running locally and facing model download/auth errors:

1. Go to: https://huggingface.co/settings/tokens
2. Create a token with **Read** access
3. Run this in terminal:
   ```bash
   huggingface-cli login
   ```

---

# 🔑 Groq API Key

1. Sign up at: https://console.groq.com
2. Create an API key
3. Save it in `.streamlit/secrets.toml` like this:

```toml
[groq]
api_key = "your_groq_api_key_here"
```

---

# 📦 Installation

## 1. Clone the Repo

```bash
git clone https://github.com/yourusername/AI-Calorie-Tracker.git
cd AI-Calorie-Tracker
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the App

```bash
streamlit run CalorieAI_Simple.py
```

---

# 🧾 requirements.txt

```txt
streamlit
transformers
torch
pillow
pandas
requests
```

---

# 🛠️ How It Works

1. Upload an image of food
2. BLIP generates a caption:  
   _e.g._ "a plate of chicken wings and fries"
3. ViT classifies top 3 food labels  
   _e.g._ `chicken_wings`, `onion_rings`
4. Only predictions ≥ 30% confidence are used
5. Each label is searched in OpenFoodFacts
6. If found → calorie displayed  
   If not found → Groq LLM queried (e.g. "Estimate calories in biryani")

---

# 📸 Example

| Uploaded Image     | Prediction        | Calories           |
|--------------------|-------------------|--------------------|
| 🍗 Chicken Wings   | `chicken_wings`   | 211 kcal/100g      |
| 🍛 Biryani         | `biryani` (LLM)   | 276 kcal/100g      |

---

# 💡 Future Improvements

- Portion size estimation using object detection
- OCR for calories on food packaging
- Nutrient breakdown (protein, carbs, fats)
- Voice-based food search and logging

---

# 👨‍💻 Author

Made by **Abhishek Kumar**  
Inspired by real-world food tracking and health applications using **AI/LLMs**.

> Powered by **Hugging Face**, **Groq**, and **OpenFoodFacts** 🥗
