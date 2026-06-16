#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# fitness_logic.py
import os
import shutil
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
# (Ensure these paths are correct on your machine)
EXERCISE_CSV = r"C:\Users\Admin\Downloads\thesistwocodes\megaGymDataset.csv"
BMI_CSV = r"C:\Users\Admin\Downloads\thesistwocodes\bmiDataset.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH = r"C:\Users\Admin\Downloads\thesistwocodes\sentencetransformers"
DENSE_MODEL_PATH = os.path.join(OUT_PATH, "dense")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"
TOP_K = 5

# --- INITIALIZATION ---
def ensure_dense_model_saved(local_path: str, model_name: str):
    os.makedirs(local_path, exist_ok=True)
    if os.path.exists(os.path.join(local_path, "config.json")):
        return True
    try:
        model = SentenceTransformer(model_name)
        model.save(local_path)
        return True
    except Exception as e:
        print(f"Model save error: {e}")
        return False

# Load Models (Global Variables)
ensure_dense_model_saved(DENSE_MODEL_PATH, EMBED_MODEL)
embedder = SentenceTransformer(DENSE_MODEL_PATH)

# Load Data
if os.path.exists(EXERCISE_CSV) and os.path.exists(BMI_CSV):
    gym_df = pd.read_csv(EXERCISE_CSV).fillna("").astype(str)
    bmi_df = pd.read_csv(BMI_CSV).fillna("").astype(str)
else:
    raise FileNotFoundError("CSV files not found. Check paths in fitness_logic.py")

# Build Indexes
def combine_text(row):
    return f"Exercise: {row.get('Title','')}\nMuscles: {row.get('BodyPart','')}\nDesc: {row.get('Desc','')}\nLevel: {row.get('Level','')}"

gym_texts = gym_df.apply(combine_text, axis=1).tolist()
bmi_texts = [f"Gender: {r['Gender']}, Age: {r['Age']}, BMI: {r['BMI']}, Case: {r['BMIcase']}" for _, r in bmi_df.iterrows()]

gym_emb = embedder.encode(gym_texts, convert_to_numpy=True)
bmi_emb = embedder.encode(bmi_texts, convert_to_numpy=True)
gym_index = faiss.IndexFlatL2(gym_emb.shape[1])
bmi_index = faiss.IndexFlatL2(bmi_emb.shape[1])
gym_index.add(gym_emb)
bmi_index.add(bmi_emb)

# Gemini Setup
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
else:
    gemini_model = None

# --- HELPER FUNCTIONS ---
def compute_bmi(weight_kg, height_cm):
    if weight_kg <= 0 or height_cm <= 0: return None
    return round(weight_kg / ((height_cm / 100.0) ** 2), 1)

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif 18.5 <= bmi < 25: return "Normal"
    elif 25 <= bmi < 30: return "Overweight"
    else: return "Obese"

def retrieve_exercise_context(query, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, idx = gym_index.search(q_emb, top_k)
    return [gym_texts[i] for i in idx[0]]

def find_bmi_case_by_embedding(bmi_val, gender=None, age=None):
    q_text = f"BMI: {bmi_val:.1f}"
    if gender: q_text += f", Gender: {gender}"
    if age: q_text += f", Age: {age}"
    q_emb = embedder.encode([q_text], convert_to_numpy=True).astype("float32")
    _, idx = bmi_index.search(q_emb, 1)
    matched = bmi_df.iloc[idx[0][0]]
    return matched["BMIcase"], f"Gender: {matched['Gender']}, Age: {matched['Age']}, BMI: {matched['BMI']}, Case: {matched['BMIcase']}"

def pick_exercises_for_part(body_part: str, bmi_val: float, n: int = 5):
    candidates = gym_df[gym_df["BodyPart"].str.contains(body_part, case=False, na=False)] if body_part else gym_df
    if candidates.empty: candidates = gym_df
    # Simple logic fallback
    n = min(n, len(candidates))
    sampled = candidates.sample(n=n, replace=False)
    return [f"- {row['Title']}: {row['Desc']}" for _, row in sampled.iterrows()]

# --- MAIN CHAT FUNCTION ---
def chat_with_bmi(question, weight, height, gender, age, chat_history):
    if weight is None or height is None: return "Please provide weight and height."
    bmi_val = compute_bmi(weight, height)
    if bmi_val is None: return "Invalid weight/height."

    bmi_cat = get_bmi_category(bmi_val)
    try:
        bmi_case_name, bmi_case_text = find_bmi_case_by_embedding(bmi_val, gender, age)
    except Exception:
        bmi_case_name, bmi_case_text = bmi_cat, f"Calculated BMI: {bmi_val:.1f}, Category: {bmi_cat}"

    history_text = ""
    # Format history if provided (Telegram might pass a list of tuples or strings)
    if chat_history:
        # Assuming list of tuples (User, AI)
        history_text = "Previous conversation:\n" + "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history[-6:]]) + "\n\n"

    ex_contexts = retrieve_exercise_context(question, top_k=8)

    full_prompt = (
        "🧠 ROLE: You are an expert personal fitness coach and nutrition advisor.\n"
        f"USER PROFILE: BMI {bmi_val} ({bmi_cat}), {gender}, {age} y/o.\n"
        f"BMI DATA CONTEXT: {bmi_case_text}\n\n"
        f"{history_text}"
          "📋 INSTRUCTIONS:\n"
        "1. ANALYZE the 'Internal Exercise Library' below. These are the ONLY exercises you know for this session.\n"
        "2. Treat the exercises as your own expert knowledge.\n"
        "- If the user requests a multi-day plan, produce a structured plan with Day 1 / Day 2 etc., sets & reps, and substitution options.\n"
        "- Avoid repeating the same exercises on consecutive days unless necessary. Provide progressions or regressions when appropriate.\n"
        "- Keep explanations concise and practical.\n"
        "- Retrieved workout contexts include a Level field: Beginner, Intermediate and Expert; use it.\n"
        "3. CUSTOMIZE the plan based on the User Profile:\n"
        "   - High BMI (>30) or Age >50: Strictly prefer 'Beginner' or low-impact exercises.\n"
        "   - Low/Normal BMI & Young: 'Intermediate' or 'Expert' exercises are allowed.\n"
        "4. STRUCTURE: If a plan is requested, use clear headers (Day 1, Day 2).\n"
        "5. NUTRITION: Include brief, goal-aligned dietary advice.\n"
        "6. TONE: Encouraging, professional, and direct. Avoid using any slangs, use clear, explicit language to provide instructions.\n\n"

        "📚 INTERNAL EXERCISE LIBRARY:\n" 
        + "\n".join(ex_contexts) +
        f"\n\n💬 USER REQUEST:\n{question}\n\nRESPONSE:"
    )

    if gemini_model:
        try:
            return gemini_model.generate_content(full_prompt).text.strip()
        except Exception as e:
            return f"Gemini Error: {e}"
    
    return "Gemini Unavailable."

