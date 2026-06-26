#!/usr/bin/env python
# coding: utf-8

import os
import shutil
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).parent
EXERCISE_CSV = BASE_DIR / "megaGymDataset.csv"
BMI_CSV = BASE_DIR / "bmiDataset.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH = BASE_DIR / "sentencetransformers"
DENSE_MODEL_PATH = OUT_PATH / "dense"

# FAISS / embedding params
TOP_K = 5
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"] 
GEMINI_MODEL_NAME = "gemini-2.5-flash" 

# Streamlit page setup
st.set_page_config(page_title="🏋️ Smart Fitness Plan Generation Chatbot", layout="wide")

if "profiles" not in st.session_state:
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

if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False

# HELPER FUNCTIONS
def compute_bmi(weight_kg, height_cm):
    if weight_kg <= 0 or height_cm <= 0:
        return None
    return round(weight_kg / ((height_cm / 100.0) ** 2), 1)

def get_bmi_category(bmi):
    if bmi is None:
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def validate_profile_name(name):
    if not name or not name.strip():
        return False, "Name cannot be empty."
    if name.strip() in st.session_state["profiles"]:
        return False, "User already exists."
    if len(name.strip()) > 50:
        return False, "Name too long (max 50 characters)."
    return True, ""


# BACKEND LOGIC
def ensure_dense_model_saved(local_path: Path, model_name: str):
    # Convert to absolute path string for compatibility
    local_path_str = str(local_path.resolve())
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists and is valid
    config_file = Path(local_path_str) / "config.json"
    modules_file = Path(local_path_str) / "modules.json"
    
    if config_file.exists() and modules_file.exists():
        try:
            test_model = SentenceTransformer(local_path_str)
            del test_model  
            return True, "exists"
        except Exception:
            # Model files exist but are corrupted, re-download
            st.warning("Model files found but invalid. Re-downloading...")
            try:
                shutil.rmtree(local_path_str)
                local_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
    
    # Download and save if not present or invalid
    try:
        model = SentenceTransformer(model_name)
        model.save(local_path_str)
        return True, "saved"
    except Exception as e:
        return False, f"error: {e}"

@st.cache_resource
def load_embedder():
    ok, status = ensure_dense_model_saved(DENSE_MODEL_PATH, EMBED_MODEL)
    if not ok:
        raise RuntimeError(f"Failed to load embedder: {status}")
    # Use absolute path string for SentenceTransformer
    model_path = str(DENSE_MODEL_PATH.resolve())
    try:
        return SentenceTransformer(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}. Please check the model directory.")

@st.cache_resource
def build_indexes(gym_texts, bmi_texts):
    embedder = load_embedder()
    gym_emb = embedder.encode(gym_texts, convert_to_numpy=True, show_progress_bar=False)
    bmi_emb = embedder.encode(bmi_texts, convert_to_numpy=True, show_progress_bar=False)
    gym_index = faiss.IndexFlatL2(gym_emb.shape[1])
    bmi_index = faiss.IndexFlatL2(bmi_emb.shape[1])
    gym_index.add(gym_emb.astype("float32"))
    bmi_index.add(bmi_emb.astype("float32"))
    return gym_index, bmi_index, embedder

@st.cache_resource
def get_gemini_model():
    if not GOOGLE_API_KEY:
        return None
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        return client
    except Exception as e:
        st.warning(f"⚠️ Gemini API Error: {e}")
        return None
# Data Loading
@st.cache_data
def load_csv(path):
    # Handle both Path objects and strings
    if isinstance(path, Path):
        path_str = str(path.resolve())
        if not path.exists():
            return None
    else:
        path_str = str(path)
        if not os.path.exists(path_str):
            return None
    
    try:
        df = pd.read_csv(path_str)
        
        # Missing data handling for exercise CSV
        if 'BodyPart' in df.columns:
            df['BodyPart'] = df['BodyPart'].fillna('Full Body').astype(str)
        if 'Desc' in df.columns:
            df['Desc'] = df['Desc'].fillna('No description available').astype(str)
        if 'Level' in df.columns:
            df['Level'] = df['Level'].fillna('Intermediate').astype(str)
        if 'Equipment' in df.columns:
            df['Equipment'] = df['Equipment'].fillna('Other').astype(str)
        
        # For BMI CSV - preserve numeric types where possible
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0).astype(str)
        if 'BMI' in df.columns:
            df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce').fillna(0).astype(str)
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].fillna('Unknown').astype(str)
        if 'BMIcase' in df.columns:
            df['BMIcase'] = df['BMIcase'].fillna('Normal').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading {path_str}: {e}")
        return None


# DATA INITIALIZATION
with st.spinner("Loading data and models..."):
    gym_df = load_csv(EXERCISE_CSV)
    bmi_df = load_csv(BMI_CSV)
    
    if gym_df is None or bmi_df is None:
        st.error(f"❌ Critical Error: Files not found. Please check:\n- {EXERCISE_CSV}\n- {BMI_CSV}")
        st.stop()
    
    # Validate data quality
    required_gym_cols = ['Title', 'BodyPart', 'Desc', 'Level']
    missing_gym_cols = set(required_gym_cols) - set(gym_df.columns)
    if missing_gym_cols:
        st.warning(f"⚠️ Missing columns in exercise CSV: {missing_gym_cols}")
    
    required_bmi_cols = ['Gender', 'Age', 'BMI', 'BMIcase']
    missing_bmi_cols = set(required_bmi_cols) - set(bmi_df.columns)
    if missing_bmi_cols:
        st.warning(f"⚠️ Missing columns in BMI CSV: {missing_bmi_cols}")
    
    # Check data quality - count exercises with default/fallback values
    if 'Title' in gym_df.columns:
        gym_default_count = len(gym_df[gym_df['Title'] == 'Unknown Exercise'])
        if gym_default_count > 0:
            st.info(f"ℹ️ {gym_default_count} exercises use default title 'Unknown Exercise' (original data was missing)")
    
    # Prepare Contexts with improved defaults
    def combine_text(row):
        return f"Exercise: {row.get('Title','Unknown Exercise')}\nMuscles: {row.get('BodyPart','Full Body')}\nDesc: {row.get('Desc','No description available')}\nLevel: {row.get('Level','Intermediate')}"
    
    gym_texts = gym_df.apply(lambda r: combine_text(r), axis=1).tolist()
    bmi_texts = [f"Gender: {r.get('Gender','Unknown')}, Age: {r.get('Age','0')}, BMI: {r.get('BMI','0')}, Case: {r.get('BMIcase','Normal')}" 
                 for _, r in bmi_df.iterrows()]
    
    gym_index, bmi_index, embedder = build_indexes(gym_texts, bmi_texts)
    gemini_model = get_gemini_model()

# RETRIEVAL FUNCTIONS
def retrieve_exercise_context(query, top_k=TOP_K, return_details=False):
    """Retrieve relevant exercise contexts using FAISS.
    
    Args:
        query: User's question/query
        top_k: Number of exercises to retrieve
        return_details: If True, also return exercise titles and distances
    
    Returns:
        If return_details=False: List of exercise context strings
        If return_details=True: Tuple of (contexts, titles, distances)
    """
    try:
        q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, idx = gym_index.search(q_emb, top_k)
        
        contexts = []
        titles = []
        distances_list = []
        
        for i, idx_val in enumerate(idx[0]):
            if idx_val < len(gym_texts):
                contexts.append(gym_texts[idx_val])
                # Extract title from the context string
                title = gym_texts[idx_val].split('\n')[0].replace('Exercise: ', '')
                titles.append(title)
                distances_list.append(float(distances[0][i]))
        
        if return_details:
            return contexts, titles, distances_list
        return contexts
    except Exception as e:
        st.error(f"Error retrieving exercises: {e}")
        if return_details:
            return [], [], []
        return []

def find_bmi_case_by_embedding(bmi_val, gender=None, age=None):
    """Find similar BMI case using embedding search."""
    try:
        q_text = f"BMI: {bmi_val:.1f}"
        if gender:
            q_text += f", Gender: {gender}"
        if age:
            q_text += f", Age: {age}"
        q_emb = embedder.encode([q_text], convert_to_numpy=True).astype("float32")
        _, idx = bmi_index.search(q_emb, 1)
        if len(idx[0]) > 0 and idx[0][0] < len(bmi_df):
            matched = bmi_df.iloc[idx[0][0]]
            return (matched["BMIcase"], 
                    f"Gender: {matched['Gender']}, Age: {matched['Age']}, BMI: {matched['BMI']}, Case: {matched['BMIcase']}")
    except Exception as e:
        st.warning(f"Error finding BMI case: {e}")
    
    # Fallback
    bmi_cat = get_bmi_category(bmi_val)
    return bmi_cat, f"Calculated BMI: {bmi_val:.1f}, Category: {bmi_cat}"

def pick_exercises_for_part(body_part: str, bmi_val: float, n: int = 5):
    """Pick exercises based on body part and BMI."""
    try:
        candidates = gym_df[gym_df["BodyPart"].str.contains(body_part, case=False, na=False)] if body_part else gym_df
        if candidates.empty:
            candidates = gym_df
        
        # Filter by BMI category
        if bmi_val > 30:
            cand = candidates[candidates["Level"].str.contains("beginner|easy", case=False, na=False)]
            if not cand.empty:
                candidates = cand
        elif bmi_val < 18.5:
            cand = candidates[candidates["Desc"].str.contains("strength|compound", case=False, na=False)]
            if not cand.empty:
                candidates = cand
        
        n = min(n, len(candidates))
        if n == 0:
            return []
        
        sampled = candidates.sample(n=n, replace=False)
        return [f"- {row['Title']}: {row['Desc']}" for _, row in sampled.iterrows()]
    except Exception as e:
        st.error(f"Error picking exercises: {e}")
        return []

# CHAT LOGIC
def chat_with_bmi(question, weight, height, gender, age, chat_history, return_retrieval_info=False):
    """Main chat function with BMI awareness.
    
    Args:
        return_retrieval_info: If True, returns tuple (response, ex_titles, ex_distances)
    """
    if weight is None or height is None:
        if return_retrieval_info:
            return "Please provide weight and height in your profile.", [], []
        return "Please provide weight and height in your profile."
    
    bmi_val = compute_bmi(weight, height)
    if bmi_val is None:
        if return_retrieval_info:
            return "Invalid weight/height values. Please check your profile.", [], []
        return "Invalid weight/height values. Please check your profile."

    bmi_cat = get_bmi_category(bmi_val)
    bmi_case_name, bmi_case_text = find_bmi_case_by_embedding(bmi_val, gender, age)

    # Use history specific to this user
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # Last 6 exchanges
        history_text = "Previous conversation:\n" + "\n".join([f"User: {q}\nAssistant: {a}" for q, a in recent]) + "\n\n"

    # Always retrieve with details
    ex_contexts, ex_titles, ex_distances = retrieve_exercise_context(question, top_k=8, return_details=True)

    full_prompt = (
        "ROLE: You are an expert personal fitness coach.\n"
        f"USER PROFILE: BMI {bmi_val} ({bmi_cat}), {gender}, {age} y/o.\n"
        f"BMI DATA CONTEXT: {bmi_case_text}\n\n"
        f"{history_text}"
        
        "INSTRUCTIONS:\n"
        "1. Analyze the 'Internal Exercise Library' below. These are the ONLY exercises you know for this session.\n"
        "2. Treat the exercises as your own expert knowledge.\n"
        "- If the user requests a multi-day plan, produce a structured plan with Day 1 / Day 2 etc., sets & reps, and substitution options.\n"
        "- Avoid repeating the same exercises on consecutive days unless necessary. Provide progressions or regressions when appropriate.\n"
        "- Provide brief explanations on how to perform the exercises.\n"
        "- Retrieved workout contexts include a Level field: Beginner, Intermediate and Expert; use it.\n"
        "3. CUSTOMIZE the plan based on the User Profile:\n"
        "   - High BMI (>30) or Age >50: Strictly prefer 'Beginner' or low-impact exercises.\n"
        "   - Low/Normal BMI & Young: 'Intermediate' or 'Expert' exercises are allowed.\n"
        "4. STRUCTURE: If a plan is requested, use clear headers (Day 1, Day 2).\n"
        "5. NUTRITION: Include brief, goal-aligned dietary advice.\n"
        "6. TONE: Encouraging, professional, and direct. Avoid using any slangs, use clear, explicit language to provide instructions.\n\n"
        
        "INTERNAL EXERCISE LIBRARY:\n" 
        + "\n".join(ex_contexts) +
        
        f"\n\n USER REQUEST:\n{question}\n\nRESPONSE:"
    )

    if gemini_model:
    try:
        response = gemini_model.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=full_prompt
        )
        response_text = response.text.strip()
        if return_retrieval_info:
            return response_text, ex_titles, ex_distances
        return response_text
    except Exception as e:
        st.warning(f"Gemini API Error: {e}")
        fallback_exercises = pick_exercises_for_part("full body", bmi_val)
        fallback_text = f"⚠️ AI model temporarily unavailable. Here are some exercises:\n\n" + "\n".join(fallback_exercises)
        if return_retrieval_info:
            return fallback_text, ex_titles, ex_distances
        return fallback_text

# SIDEBAR: PROFILE MANAGER & INPUTS
with st.sidebar:
    st.title("👥 User Profiles")
    
    # 1. Profile Switcher
    profile_names = list(st.session_state["profiles"].keys())
    current_index = 0
    if st.session_state["current_user"] in profile_names:
        current_index = profile_names.index(st.session_state["current_user"])
    
    selected_profile = st.selectbox(
        "Select Profile", 
        options=profile_names, 
        index=current_index,
        key="profile_selector"
    )
    
    if selected_profile != st.session_state["current_user"]:
        st.session_state["current_user"] = selected_profile
        st.rerun()

    # 2. Add New Profile
    with st.expander("➕ Create New Profile"):
        new_name = st.text_input("Name", key="new_profile_name", max_chars=50)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add User", key="add_user_btn"):
                is_valid, error_msg = validate_profile_name(new_name)
                if is_valid:
                    st.session_state["profiles"][new_name.strip()] = {
                        "weight": 70.0, 
                        "height": 170.0, 
                        "gender": "Male", 
                        "age": 25, 
                        "history": []
                    }
                    st.session_state["current_user"] = new_name.strip()
                    st.success(f"Created {new_name.strip()}!")
                    st.rerun()
                else:
                    st.error(error_msg)
    
    # 3. Delete Profile Section 
    if len(profile_names) > 1:  # Don't allow deleting the last profile
        with st.expander("🗑️ Delete Profile", expanded=False):
            profile_to_delete = st.selectbox(
                "Select profile to delete",
                options=[p for p in profile_names if p != st.session_state["current_user"]],
                key="delete_profile_selector"
            )
            if st.button("⚠️ Delete Profile", key="delete_profile_btn", type="secondary"):
                if profile_to_delete in st.session_state["profiles"]:
                    del st.session_state["profiles"][profile_to_delete]
                    if st.session_state["current_user"] == profile_to_delete:
                        st.session_state["current_user"] = list(st.session_state["profiles"].keys())[0]
                    st.success(f"Deleted {profile_to_delete}")
                    st.rerun()

    st.divider()
    
    # 4. Edit Current Profile Data
    user_data = st.session_state["profiles"][st.session_state["current_user"]]
    
    st.header(f"👤 {st.session_state['current_user']}'s Stats")
    
    new_weight = st.number_input(
        "Weight (kg)", 
        min_value=1.0, 
        max_value=300.0, 
        value=user_data["weight"],
        key=f"weight_{st.session_state['current_user']}"
    )
    new_height = st.number_input(
        "Height (cm)", 
        min_value=50.0, 
        max_value=250.0, 
        value=user_data["height"],
        key=f"height_{st.session_state['current_user']}"
    )
    new_gender = st.radio(
        "Gender", 
        ["Male", "Female"], 
        index=0 if user_data["gender"] == "Male" else 1,
        key=f"gender_{st.session_state['current_user']}"
    )
    new_age = st.number_input(
        "Age", 
        min_value=10, 
        max_value=100, 
        value=user_data["age"],
        key=f"age_{st.session_state['current_user']}"
    )

    # Update the session state dictionary with current inputs
    st.session_state["profiles"][st.session_state["current_user"]]["weight"] = new_weight
    st.session_state["profiles"][st.session_state["current_user"]]["height"] = new_height
    st.session_state["profiles"][st.session_state["current_user"]]["gender"] = new_gender
    st.session_state["profiles"][st.session_state["current_user"]]["age"] = new_age

    st.divider()

    # Calculate BMI for display
    current_bmi = compute_bmi(new_weight, new_height)
    bmi_cat = get_bmi_category(current_bmi) if current_bmi else "Invalid"
    st.metric("Current BMI", f"{current_bmi if current_bmi else 'N/A'}", delta=bmi_cat)

    if st.button("🗑️ Clear This Chat History", key="clear_history_btn"):
        st.session_state["profiles"][st.session_state["current_user"]]["history"] = []
        st.success("Chat history cleared!")
        st.rerun()
    
    st.divider()
    
    # Debug Mode Toggle
    st.session_state["debug_mode"] = st.checkbox(
        "🔍 Debug Mode (Show Retrieved Exercises)", 
        value=st.session_state["debug_mode"],
        key="debug_mode_checkbox"
    )


#UI
st.title("🏋️ BMI-aware Fitness Chatbot")
st.caption(f"Current Profile: **{st.session_state['current_user']}** | "
          f"BMI: {compute_bmi(st.session_state['profiles'][st.session_state['current_user']]['weight'], 
                              st.session_state['profiles'][st.session_state['current_user']]['height']) or 'N/A'}")

# Access the CURRENT USER's history
current_history = st.session_state["profiles"][st.session_state["current_user"]]["history"]

# Display History
for q, a in current_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Chat Input
if prompt := st.chat_input("Ask for a workout plan, nutrition advice, or exercise recommendations..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get current user stats for the logic
    c_user = st.session_state["profiles"][st.session_state["current_user"]]
    
    with st.chat_message("assistant"):
        with st.spinner("💭 Analyzing your request..."):
            # Get response with optional retrieval info for debug mode
            debug_mode = st.session_state.get("debug_mode", False)
            if debug_mode:
                response, ex_titles, ex_distances = chat_with_bmi(
                    prompt,
                    c_user["weight"],
                    c_user["height"],
                    c_user["gender"],
                    c_user["age"],
                    chat_history=current_history,
                    return_retrieval_info=True
                )
            else:
                response = chat_with_bmi(
                    prompt,
                    c_user["weight"],
                    c_user["height"],
                    c_user["gender"],
                    c_user["age"],
                    chat_history=current_history,
                    return_retrieval_info=False
                )
                ex_titles, ex_distances = [], []
            
            st.write(response)
            
            # Show retrieved exercises in debug mode
            if debug_mode and len(ex_titles) > 0:
                with st.expander(f"🔍 Debug: Retrieved {len(ex_titles)} Exercises", expanded=False):
                    st.write("**Exercises retrieved from database:**")
                    for i, (title, distance) in enumerate(zip(ex_titles, ex_distances), 1):
                        # Format distance as similarity (lower is better for L2 distance)
                        st.write(f"{i}. **{title}** (distance: {distance:.4f})")
                    st.caption(f"💡 Lower distance = more similar to your query: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
                    st.info("These exercises were used as context for the AI response above.")
    
    # Save to SPECIFIC USER history
    st.session_state["profiles"][st.session_state["current_user"]]["history"].append((prompt, response))

