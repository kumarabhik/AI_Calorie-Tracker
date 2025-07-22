


# # ------------------------ Imports ------------------------
# import streamlit as st
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import torch
# import pandas as pd
# from datetime import datetime
# import requests

# # ------------------------ Session State: Weekly Logs ------------------------
# def initialize_weekly_logs():
#     return {
#         day: {
#             "Breakfast": [],
#             "Lunch": [],
#             "Dinner": []
#         }
#         for day in [
#             "Monday", "Tuesday", "Wednesday", "Thursday",
#             "Friday", "Saturday", "Sunday"
#         ]
#     }

# if "weekly_logs" not in st.session_state:
#     st.session_state.weekly_logs = initialize_weekly_logs()

# # ------------------------ Page Config ------------------------
# st.set_page_config(
#     page_title="AI Calorie Tracker 🍽️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown(
#     """
#     <style>
#     .main { background-color: #f7f7f7; padding: 1rem 2rem; }
#     .block-container { padding-top: 2rem; }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # ------------------------ Sidebar ------------------------
# st.sidebar.title("🔧 App Controls")
# uploaded_file = st.sidebar.file_uploader("📸 Upload your food image", type=["jpg", "jpeg", "png"])
# st.sidebar.markdown("---")
# st.sidebar.info("Switch theme from ⚙️ → Settings → Theme")

# # ------------------------ Meal Selector ------------------------
# day_of_week = datetime.today().strftime("%A")
# meal_time = st.sidebar.selectbox("🍽️ Select Meal Type", ["Breakfast", "Lunch", "Dinner"])

# # ------------------------ Weekly Meal History Table ------------------------
# def weekly_logs_to_df(weekly_logs):
#     data = []
#     for day, meals in weekly_logs.items():
#         row = {
#             "Day": day,
#             "Breakfast": ", ".join([item["food"] for item in meals["Breakfast"]]) if meals["Breakfast"] else "-",
#             "Lunch": ", ".join([item["food"] for item in meals["Lunch"]]) if meals["Lunch"] else "-",
#             "Dinner": ", ".join([item["food"] for item in meals["Dinner"]]) if meals["Dinner"] else "-"
#         }
#         data.append(row)
#     return pd.DataFrame(data)

# history_df = weekly_logs_to_df(st.session_state.weekly_logs)
# st.sidebar.markdown("### 📅 Weekly Meal History")
# st.sidebar.dataframe(history_df.set_index("Day"), height=310)

# # ------------------------ Title ------------------------
# st.title("🍽️ AI Calorie Tracker")
# st.write("Upload a photo of food and get predicted food name and estimated calories per 100g!")

# # ------------------------ Calorie Dataset ------------------------
# @st.cache_data
# def load_calorie_dataset():
#     chunks = pd.read_csv(
#         "en.openfoodfacts.org.products.csv/en.openfoodfacts.org.products.csv",
#         sep="\t",
#         usecols=["product_name", "energy-kcal_100g"],
#         dtype={"product_name": str, "energy-kcal_100g": str},
#         chunksize=100_000,
#         low_memory=False
#     )
#     filtered_rows = []
#     for chunk in chunks:
#         chunk = chunk.dropna(subset=["product_name", "energy-kcal_100g"])
#         chunk["product_name"] = chunk["product_name"].str.lower()
#         chunk["energy-kcal_100g"] = pd.to_numeric(chunk["energy-kcal_100g"], errors="coerce")
#         filtered_rows.append(chunk.dropna(subset=["energy-kcal_100g"]))
#     return pd.concat(filtered_rows, ignore_index=True)

# # ------------------------ BLIP Model ------------------------
# @st.cache_resource
# def load_blip_model():
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     return processor, model

# # ------------------------ ViT Model ------------------------
# @st.cache_resource
# def load_food_classifier():
#     model_name = "nateraw/food"
#     processor = AutoImageProcessor.from_pretrained(model_name)
#     model = AutoModelForImageClassification.from_pretrained(model_name)
#     return processor, model

# def classify_food(image, processor, model, top_k=3):
#     inputs = processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#     top = torch.topk(probs, k=top_k)
#     labels = [model.config.id2label[i.item()] for i in top.indices[0]]
#     scores = [round(s.item(), 3) for s in top.values[0]]
#     return list(zip(labels, scores))

# # ------------------------ Groq Calorie Estimate ------------------------
# def get_calorie_estimate_from_groq(food_name):
#     prompt = f"""
#     Estimate the average calorie content per 100 grams for the following food item:
#     Food: {food_name}
#     Respond with just the number (no units, no explanation).
#     """
#     api_key = st.secrets["groq"]["api_key"]
#     try:
#         response = requests.post(
#             "https://api.groq.com/openai/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json"
#             },
#             json={
#                 "model": "llama3-8b-8192",
#                 "messages": [{"role": "user", "content": prompt.strip()}],
#                 "temperature": 0.2
#             }
#         )
#         if response.status_code == 200:
#             result = response.json()
#             content = result["choices"][0]["message"]["content"]
#             calorie = "".join([c for c in content if c.isdigit() or c == "."])
#             return float(calorie) if calorie else None
#         else:
#             st.warning(f"Groq API error ({response.status_code}): {response.text}")
#             return None
#     except Exception as e:
#         st.error(f"❌ Error getting calorie from Groq: {e}")
#         return None

# # ------------------------ Main Logic ------------------------
# if uploaded_file is not None:
#     col1, col2 = st.columns([1, 2])

#     with col1:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="📷 Uploaded Image", use_column_width=True)

#     with col2:
#         st.header("🔍 Food Analysis")

#         st.subheader("🧠 Generating Caption (BLIP)...")
#         blip_processor, blip_model = load_blip_model()
#         inputs = blip_processor(image, return_tensors="pt")
#         out = blip_model.generate(**inputs, max_length=50)
#         caption = blip_processor.decode(out[0], skip_special_tokens=True)
#         st.success(f"📋 *{caption}*")

#         st.subheader("🍽️ Classifying Image with Food101 (ViT)")
#         vit_processor, vit_model = load_food_classifier()
#         vit_preds = classify_food(image, vit_processor, vit_model)

#         filtered_preds = [(label, score) for label, score in vit_preds if score >= 0.3]

#         calorie_df = load_calorie_dataset()

#         if filtered_preds:
#             st.success("✅ Predictions:")
#             for label, score in filtered_preds:
#                 st.write(f"🍴 {label.replace('_', ' ').title()} — **{score * 100:.1f}%**")

#                 food_name = label.replace("_", " ").lower()
#                 matches = calorie_df[calorie_df["product_name"].str.contains(food_name, na=False)]

#                 if not matches.empty:
#                     avg_calories = matches["energy-kcal_100g"].mean()
#                     st.write(f"🍽️ {label.replace('_', ' ').title()} → ~**{avg_calories:.0f} kcal/100g** (from dataset)")
#                     st.session_state.weekly_logs[day_of_week][meal_time].append({
#                         "food": label.replace("_", " ").title(),
#                         "calories": round(avg_calories, 2)
#                     })
#                 else:
#                     st.warning(f"⚠️ Not found in dataset → asking Groq for estimate...")
#                     estimate = get_calorie_estimate_from_groq(food_name)
#                     if estimate:
#                         st.write(f"🍽️ {label.replace('_', ' ').title()} → ~**{estimate:.0f} kcal/100g** (from Groq 🤖)")
#                         st.session_state.weekly_logs[day_of_week][meal_time].append({
#                             "food": label.replace("_", " ").title(),
#                             "calories": round(estimate, 2)
#                         })
#                     else:
#                         st.error(f"❌ No calorie estimate available for: {label.title()}")
#         else:
#             st.warning("❗ No high-confidence food predictions found.")

# # ------------------------ Meal Log & History ------------------------
# with st.expander("📅 Weekly Meal History"):
#     total_calories = {}
#     for day, meals in st.session_state.weekly_logs.items():
#         st.markdown(f"### 📅 {day}")
#         for meal, items in meals.items():
#             if items:
#                 items_display = ", ".join([item["food"] for item in items])
#             else:
#                 items_display = "—"
#             st.write(f"🍽️ {meal.title()}: {items_display}")
#             total_calories.setdefault(meal, 0)
#             total_calories[meal] += sum(item.get("calories", 200) for item in items)

#     st.markdown("---")
#     st.markdown("### 📈 Weekly Estimated Calorie Totals")
#     for meal, total in total_calories.items():
#         st.write(f"🔢 {meal.title()}: ~**{total} kcal**")

#     if st.button("🔁 Reset Weekly Logs"):
#         st.session_state.weekly_logs = initialize_weekly_logs()
#         st.success("✅ Meal history cleared!")

# # ------------------------ Current Day Meals ------------------------
# st.markdown("---")
# st.header(f"📅 Today's Logged Meals — {day_of_week}")
# for meal in ["Breakfast", "Lunch", "Dinner"]:
#     st.subheader(meal.title())
#     meal_items = st.session_state.weekly_logs[day_of_week][meal]
#     if meal_items:
#         for item in meal_items:
#             st.markdown(f"- 🍱 **{item['food']}** → {item['calories']} kcal/100g")
#     else:
#         st.markdown("_No items logged._")

# # ------------------------ Footer ------------------------
# st.markdown("---")
# st.markdown("Made with 💡 using BLIP, ViT, and OpenFoodFacts 🥗")


import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import pandas as pd
from datetime import datetime
import requests

# ------------------------ Session State: Weekly Logs ------------------------
def initialize_weekly_logs():
    return {
        day: {
            "Breakfast": [],
            "Lunch": [],
            "Dinner": []
        }
        for day in [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]
    }

if "weekly_logs" not in st.session_state:
    st.session_state.weekly_logs = initialize_weekly_logs()

# ------------------------ Meal Selector & Sidebar ------------------------
day_of_week = datetime.today().strftime("%A")
meal_time = st.sidebar.selectbox("🍽️ Select Meal Type", ["Breakfast", "Lunch", "Dinner"])

st.sidebar.markdown("### 📅 Weekly Meal History")

def weekly_logs_to_df(weekly_logs):
    data = []
    for day, meals in weekly_logs.items():
        row = {
            "Day": day,
            "Breakfast": ", ".join(item["food"] for item in meals["Breakfast"]) if meals["Breakfast"] else "-",
            "Lunch": ", ".join(item["food"] for item in meals["Lunch"]) if meals["Lunch"] else "-",
            "Dinner": ", ".join(item["food"] for item in meals["Dinner"]) if meals["Dinner"] else "-"
        }
        data.append(row)
    return pd.DataFrame(data)

history_df = weekly_logs_to_df(st.session_state.weekly_logs)
st.sidebar.dataframe(history_df.set_index("Day"), height=310)

# ------------------------ Page Config ------------------------
st.set_page_config(
    page_title="AI Calorie Tracker 🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f7f7; padding: 1rem 2rem; }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------ Sidebar ------------------------
st.sidebar.title("🔧 App Controls")
uploaded_file = st.sidebar.file_uploader("📸 Upload your food image", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.info("Switch theme from ⚙️ → Settings → Theme")

st.markdown("---")

# ------------------------ Title ------------------------
st.title("🍽️ AI Calorie Tracker")
st.write("Upload a photo of food and get predicted food name and estimated calories per 100g!")

# ------------------------ Calorie Dataset ------------------------
@st.cache_data
def load_calorie_dataset():
    chunks = pd.read_csv(
        "en.openfoodfacts.org.products.csv/en.openfoodfacts.org.products.csv",
        sep="\t",
        usecols=["product_name", "energy-kcal_100g"],
        dtype={"product_name": str, "energy-kcal_100g": str},
        chunksize=100_000,
        low_memory=False
    )
    filtered_rows = []
    for chunk in chunks:
        chunk = chunk.dropna(subset=["product_name", "energy-kcal_100g"])
        chunk["product_name"] = chunk["product_name"].str.lower()
        chunk["energy-kcal_100g"] = pd.to_numeric(chunk["energy-kcal_100g"], errors="coerce")
        filtered_rows.append(chunk.dropna(subset=["energy-kcal_100g"]))
    return pd.concat(filtered_rows, ignore_index=True)

# ------------------------ BLIP Model ------------------------
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# ------------------------ ViT Model ------------------------
@st.cache_resource
def load_food_classifier():
    model_name = "nateraw/food"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

def classify_food(image, processor, model, top_k=3):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    top = torch.topk(probs, k=top_k)
    labels = [model.config.id2label[i.item()] for i in top.indices[0]]
    scores = [round(s.item(), 3) for s in top.values[0]]
    return list(zip(labels, scores))

# ------------------------ Groq Calorie Estimate ------------------------
def get_calorie_estimate_from_groq(food_name):
    prompt = f"""
    Estimate the average calorie content per 100 grams for the following food item:
    Food: {food_name}
    Respond with just the number (no units, no explanation).
    """
    api_key = st.secrets["groq"]["api_key"]
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt.strip()}],
                "temperature": 0.2
            }
        )
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            calorie = "".join([c for c in content if c.isdigit() or c == "."])
            return float(calorie) if calorie else None
        else:
            st.warning(f"Groq API error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Error getting calorie from Groq: {e}")
        return None

# ------------------------ Main Logic ------------------------
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with col2:
        st.header("🔍 Food Analysis")

        st.subheader("🧠 Generating Caption (BLIP)...")
        blip_processor, blip_model = load_blip_model()
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        st.success(f"📋 *{caption}*")

        st.subheader("🍽️ Classifying Image with Food101 (ViT)")
        vit_processor, vit_model = load_food_classifier()
        vit_preds = classify_food(image, vit_processor, vit_model)

        filtered_preds = [(label, score) for label, score in vit_preds if score >= 0.3]

        if filtered_preds:
            st.success("✅ Predictions:")
            for label, score in filtered_preds:
                st.write(f"🍴 {label.replace('_', ' ').title()} — **{score * 100:.1f}%**")
        else:
            st.warning("❗ No high-confidence food predictions found.")

        # 🔥 Calorie Estimation
        st.subheader("🔥 Calories in Your Food (per 100g)")
        calorie_df = load_calorie_dataset()

        for label, score in filtered_preds:
            food_name = label.replace("_", " ").lower()
            matches = calorie_df[calorie_df["product_name"].str.contains(food_name, na=False)]
            if not matches.empty:
                avg_calories = matches["energy-kcal_100g"].mean()
                st.write(f"🍽️ {label.replace('_', ' ').title()} → ~**{avg_calories:.0f} kcal/100g**")
                st.session_state.weekly_logs[day_of_week][meal_time].append({
                    "food": label.replace("_", " ").title(),
                    "calories": round(avg_calories, 2)
                })
            else:
                st.warning(f"⚠️ Not found in dataset → asking Groq for estimate...")
                estimate = get_calorie_estimate_from_groq(food_name)
                if estimate:
                    st.write(f"🍽️ {label.replace('_', ' ').title()} → ~**{estimate:.0f} kcal/100g**")
                    st.session_state.weekly_logs[day_of_week][meal_time].append({
                        "food": label.replace("_", " ").title(),
                        "calories": round(estimate, 2)
                    })
                else:
                    st.error(f"❌ No calorie estimate available for: {label.title()}")

# ------------------------ Meal Log & History ------------------------
st.markdown("---")
st.header(f"📅 Today's Logged Meals — {day_of_week}")
for meal in ["Breakfast", "Lunch", "Dinner"]:
    st.subheader(meal.title())
    meal_items = st.session_state.weekly_logs[day_of_week][meal]
    if meal_items:
        for item in meal_items:
            st.markdown(f"- 🍱 **{item['food']}** → {item['calories']} kcal/100g")
    else:
        st.markdown("_No items logged._")

with st.expander("📅 Weekly Meal History"):
    total_calories = {}  # Collect calories by meal
    for day, meals in st.session_state.weekly_logs.items():
        st.markdown(f"### 📅 {day}")
        for meal, items in meals.items():
            if items:
                items_display = ", ".join([item["food"] for item in items])
            else:
                items_display = "—"
            st.write(f"🍽️ {meal.title()}: {items_display}")

            total_calories.setdefault(meal, 0)
            total_calories[meal] += sum(item.get("calories", 200) for item in items)

    st.markdown("---")
    st.markdown("### 📈 Weekly Estimated Calorie Totals")
    for meal, total in total_calories.items():
        st.write(f"🔢 {meal.title()}: ~**{total} kcal**")

    if st.button("🔁 Reset Weekly Logs"):
        st.session_state.weekly_logs = initialize_weekly_logs()
        st.success("✅ Meal history cleared!")

# ------------------------ Footer ------------------------
st.markdown("---")
st.markdown("Made with 💡 using BLIP, ViT, and OpenFoodFacts 🥗")
