#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================================
# Exercise RAG Chatbot — Multi-User Profiles + Unified UI
# ==========================================================

import os
import time
import random
import re
import shutil
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv


# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# CONFIG - (Original Paths)
# ---------------------------
EXERCISE_CSV = r"C:\Users\Admin\Downloads\thesistwocodes\megaGymDataset.csv"
BMI_CSV = r"C:\Users\Admin\Downloads\thesistwocodes\bmiDataset.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH = r"C:\Users\Admin\Downloads\thesistwocodes\sentencetransformers"
DENSE_MODEL_PATH = os.path.join(OUT_PATH, "dense")

# FAISS / embedding params
TOP_K = 5
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="🏋️ BMI-aware Fitness Chatbot", layout="wide")

# ---------------------------
# INITIALIZE SESSION STATE FOR PROFILES
# ---------------------------
if "profiles" not in st.session_state:
    # Default profile structure
    st.session_state["profiles"] = {
        "Default User": {
            "weight": 70.0,
            "height": 170.0,
            "gender": "Male",
            "age": 25,
            "history": []
        }
    }

if "current_user" not in st.session_state:
    st.session_state["current_user"] = "Default User"

# ---------------------------
# SIDEBAR: PROFILE MANAGER & INPUTS
# ---------------------------
with st.sidebar:
    st.title("👥 User Profiles")
    
    # 1. Profile Switcher
    profile_names = list(st.session_state["profiles"].keys())
    selected_profile = st.selectbox(
        "Select Profile", 
        options=profile_names, 
        index=profile_names.index(st.session_state["current_user"])
    )
    
    # Update current user if changed
    if selected_profile != st.session_state["current_user"]:
        st.session_state["current_user"] = selected_profile
        st.rerun() # Refresh app to load new user data

    # 2. Add New Profile Section
    with st.expander("➕ Create New Profile"):
        new_name = st.text_input("Name")
        if st.button("Add User"):
            if new_name and new_name not in st.session_state["profiles"]:
                st.session_state["profiles"][new_name] = {
                    "weight": 70.0, "height": 170.0, "gender": "Male", "age": 25, "history": []
                }
                st.session_state["current_user"] = new_name
                st.success(f"Created {new_name}!")
                time.sleep(0.5)
                st.rerun()
            elif new_name in st.session_state["profiles"]:
                st.error("User already exists.")

    st.divider()
    
    # 3. Edit Current Profile Data
    # We load the data from the dictionary, and update it immediately on change
    user_data = st.session_state["profiles"][st.session_state["current_user"]]
    
    st.header(f"👤 {st.session_state['current_user']}'s Stats")
    
    # Note: We use the profile data as 'value', and update the dictionary directly on change
    new_weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=user_data["weight"])
    new_height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=user_data["height"])
    new_gender = st.radio("Gender", ["Male", "Female"], index=0 if user_data["gender"] == "Male" else 1)
    new_age = st.number_input("Age", min_value=10, max_value=100, value=user_data["age"])

    # Update the session state dictionary with current inputs
    st.session_state["profiles"][st.session_state["current_user"]]["weight"] = new_weight
    st.session_state["profiles"][st.session_state["current_user"]]["height"] = new_height
    st.session_state["profiles"][st.session_state["current_user"]]["gender"] = new_gender
    st.session_state["profiles"][st.session_state["current_user"]]["age"] = new_age

    st.divider()

    # Calculate BMI for display
    def quick_bmi(w, h):
        if w > 0 and h > 0:
            return round(w / ((h/100)**2), 1)
        return 0
    
    current_bmi = quick_bmi(new_weight, new_height)
    st.metric("Current BMI", f"{current_bmi}")

    if st.button("🗑️ Clear This Chat History"):
        st.session_state["profiles"][st.session_state["current_user"]]["history"] = []
        st.rerun()

# ---------------------------
# BACKEND LOGIC (Same as before)
# ---------------------------
def ensure_dense_model_saved(local_path: str, model_name: str):
    os.makedirs(local_path, exist_ok=True)
    if os.path.exists(os.path.join(local_path, "config.json")):
        return True, "exists"
    
    files = os.listdir(local_path)
    if files:
        try:
            shutil.rmtree(local_path)
            os.makedirs(local_path, exist_ok=True)
        except Exception:
            pass
    try:
        model = SentenceTransformer(model_name)
        model.save(local_path)
        return True, "saved"
    except Exception as e:
        return False, f"error: {e}"

@st.cache_resource
def load_embedder():
    ok, status = ensure_dense_model_saved(DENSE_MODEL_PATH, EMBED_MODEL)
    if not ok:
        raise RuntimeError(f"Failed: {status}")
    return SentenceTransformer(DENSE_MODEL_PATH)

@st.cache_resource
def build_indexes(gym_texts, bmi_texts):
    embedder = load_embedder()
    gym_emb = embedder.encode(gym_texts, convert_to_numpy=True)
    bmi_emb = embedder.encode(bmi_texts, convert_to_numpy=True)
    gym_index = faiss.IndexFlatL2(gym_emb.shape[1])
    bmi_index = faiss.IndexFlatL2(bmi_emb.shape[1])
    gym_index.add(gym_emb)
    bmi_index.add(bmi_emb)
    return gym_index, bmi_index, embedder

# Gemini Setup
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        gemini_model = None
else:
    gemini_model = None

# Data Loading
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.fillna("").astype(str)
    return df

gym_df = load_csv(EXERCISE_CSV)
bmi_df = load_csv(BMI_CSV)

if gym_df is None or bmi_df is None:
    st.error(f"❌ Critical Error: Files not found at specified paths.")
    st.stop()

# Prepare Contexts
def combine_text(row):
    return f"Exercise: {row.get('Title','')}\nMuscles: {row.get('BodyPart','')}\nDesc: {row.get('Desc','')}\nLevel: {row.get('Level','')}"

gym_texts = gym_df.apply(lambda r: combine_text(r), axis=1).tolist()
bmi_texts = [f"Gender: {r['Gender']}, Age: {r['Age']}, BMI: {r['BMI']}, Case: {r['BMIcase']}" for _, r in bmi_df.iterrows()]

gym_index, bmi_index, embedder = build_indexes(gym_texts, bmi_texts)

# Helper Functions
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
    if bmi_val > 30:
        cand = candidates[candidates["Level"].str.contains("beginner|easy", case=False, na=False)]
        if not cand.empty: candidates = cand
    elif bmi_val < 18.5:
        cand = candidates[candidates["Desc"].str.contains("strength|compound", case=False, na=False)]
        if not cand.empty: candidates = cand
    n = min(n, len(candidates))
    sampled = candidates.sample(n=n, replace=False)
    return [f"- {row['Title']}: {row['Desc']}" for _, row in sampled.iterrows()]

# Chat Logic
def chat_with_bmi(question, weight, height, gender, age, chat_history):
    if weight is None or height is None: return "Please provide weight and height."
    bmi_val = compute_bmi(weight, height)
    if bmi_val is None: return "Invalid weight/height."

    bmi_cat = get_bmi_category(bmi_val)
    try:
        bmi_case_name, bmi_case_text = find_bmi_case_by_embedding(bmi_val, gender, age)
    except Exception:
        bmi_case_name, bmi_case_text = bmi_cat, f"Calculated BMI: {bmi_val:.1f}, Category: {bmi_cat}"

    # Use history specific to this user
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        history_text = "Previous conversation:\n" + "\n".join([f"User: {q}\nAssistant: {a}" for q, a in recent]) + "\n\n"

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
            return f"(Gemini Error: {e}) Showing exercises:\n" + "\n".join(pick_exercises_for_part("full body", bmi_val))
    
    return "(Gemini Unavailable) Exercises:\n" + "\n".join(pick_exercises_for_part("full body", bmi_val))

# ---------------------------
# UI: UNIFIED CHAT INTERFACE
# ---------------------------

st.title("🏋️ BMI-aware Fitness Chatbot")
st.caption(f"Current Profile: {st.session_state['current_user']}")

# Access the CURRENT USER's history
current_history = st.session_state["profiles"][st.session_state["current_user"]]["history"]

# Display History
for q, a in current_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Chat Input
if prompt := st.chat_input("Ask for a plan..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get current user stats for the logic
    c_user = st.session_state["profiles"][st.session_state["current_user"]]
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_bmi(
                prompt,
                c_user["weight"],
                c_user["height"],
                c_user["gender"],
                c_user["age"],
                chat_history=current_history
            )
            st.write(response)
    
    # Save to SPECIFIC USER history
    st.session_state["profiles"][st.session_state["current_user"]]["history"].append((prompt, response))

