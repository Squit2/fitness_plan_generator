#!/usr/bin/env python
# coding: utf-8

import os
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    MessageHandler, 
    filters, 
    ContextTypes, 
    ConversationHandler
)
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# ---------------------------
# CONFIG - Using relative paths for portability
# ---------------------------
BASE_DIR = Path(__file__).parent
EXERCISE_CSV = BASE_DIR / "megaGymDataset.csv"
BMI_CSV = BASE_DIR / "bmiDataset.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH = BASE_DIR / "sentencetransformers"
DENSE_MODEL_PATH = OUT_PATH / "dense"
TOP_K = 5

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def compute_bmi(weight_kg, height_cm):
    """Calculate BMI from weight and height."""
    if weight_kg <= 0 or height_cm <= 0:
        return None
    return round(weight_kg / ((height_cm / 100.0) ** 2), 1)

def get_bmi_category(bmi):
    """Get BMI category string."""
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

# ---------------------------
# BACKEND LOGIC - Model Loading
# ---------------------------
def ensure_dense_model_saved(local_path: Path, model_name: str):
    """Ensure the embedding model is saved locally."""
    local_path_str = str(local_path.resolve())
    local_path.mkdir(parents=True, exist_ok=True)
    
    config_file = Path(local_path_str) / "config.json"
    modules_file = Path(local_path_str) / "modules.json"
    
    if config_file.exists() and modules_file.exists():
        try:
            test_model = SentenceTransformer(local_path_str)
            del test_model
            return True, "exists"
        except Exception:
            logging.warning("Model files found but invalid. Re-downloading...")
            try:
                shutil.rmtree(local_path_str)
                local_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
    
    try:
        model = SentenceTransformer(model_name)
        model.save(local_path_str)
        return True, "saved"
    except Exception as e:
        return False, f"error: {e}"

def load_embedder():
    """Load the sentence transformer embedder."""
    ok, status = ensure_dense_model_saved(DENSE_MODEL_PATH, EMBED_MODEL)
    if not ok:
        raise RuntimeError(f"Failed to load embedder: {status}")
    model_path = str(DENSE_MODEL_PATH.resolve())
    try:
        return SentenceTransformer(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def build_indexes(gym_texts, bmi_texts):
    """Build FAISS indexes for exercise and BMI data."""
    embedder = load_embedder()
    gym_emb = embedder.encode(gym_texts, convert_to_numpy=True, show_progress_bar=False)
    bmi_emb = embedder.encode(bmi_texts, convert_to_numpy=True, show_progress_bar=False)
    gym_index = faiss.IndexFlatL2(gym_emb.shape[1])
    bmi_index = faiss.IndexFlatL2(bmi_emb.shape[1])
    gym_index.add(gym_emb.astype("float32"))
    bmi_index.add(bmi_emb.astype("float32"))
    return gym_index, bmi_index, embedder

def get_gemini_model():
    """Get Gemini model instance with proper error handling."""
    if not GOOGLE_API_KEY:
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        logging.warning(f"⚠️ Gemini API Error: {e}")
        return None

def load_csv(path):
    """Load CSV file with error handling."""
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
        df = df.fillna("").astype(str)
        return df
    except Exception as e:
        logging.error(f"Error loading {path_str}: {e}")
        return None

# ---------------------------
# DATA INITIALIZATION (Global)
# ---------------------------
logging.info("Loading data and models...")
gym_df = load_csv(EXERCISE_CSV)
bmi_df = load_csv(BMI_CSV)

if gym_df is None or bmi_df is None:
    raise RuntimeError(f"Critical Error: Files not found. Please check:\n- {EXERCISE_CSV}\n- {BMI_CSV}")

# Prepare Contexts
def combine_text(row):
    return f"Exercise: {row.get('Title','')}\nMuscles: {row.get('BodyPart','')}\nDesc: {row.get('Desc','')}\nLevel: {row.get('Level','')}"

gym_texts = gym_df.apply(lambda r: combine_text(r), axis=1).tolist()
bmi_texts = [f"Gender: {r['Gender']}, Age: {r['Age']}, BMI: {r['BMI']}, Case: {r['BMIcase']}" 
             for _, r in bmi_df.iterrows()]

gym_index, bmi_index, embedder = build_indexes(gym_texts, bmi_texts)
gemini_model = get_gemini_model()
logging.info("Data and models loaded successfully!")

# ---------------------------
# RETRIEVAL FUNCTIONS
# ---------------------------
def retrieve_exercise_context(query, top_k=TOP_K):
    """Retrieve relevant exercise contexts using FAISS."""
    try:
        q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
        _, idx = gym_index.search(q_emb, top_k)
        return [gym_texts[i] for i in idx[0] if i < len(gym_texts)]
    except Exception as e:
        logging.error(f"Error retrieving exercises: {e}")
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
        logging.warning(f"Error finding BMI case: {e}")
    
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
        logging.error(f"Error picking exercises: {e}")
        return []

# ---------------------------
# CHAT LOGIC
# ---------------------------
def chat_with_bmi(question, weight, height, gender, age, chat_history):
    """Main chat function with BMI awareness."""
    if weight is None or height is None:
        return "Please provide weight and height in your profile."
    
    bmi_val = compute_bmi(weight, height)
    if bmi_val is None:
        return "Invalid weight/height values. Please check your profile."

    bmi_cat = get_bmi_category(bmi_val)
    bmi_case_name, bmi_case_text = find_bmi_case_by_embedding(bmi_val, gender, age)

    # Use history specific to this user
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # Last 6 exchanges
        history_text = "Previous conversation:\n" + "\n".join([f"User: {q}\nAssistant: {a}" for q, a in recent]) + "\n\n"

    # Retrieve exercises
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
        "- Provide brief explanations on how to perform the exercises.\n"
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
            response = gemini_model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            logging.warning(f"Gemini API Error: {e}")
            fallback_exercises = pick_exercises_for_part("full body", bmi_val)
            return f"⚠️ AI model temporarily unavailable. Here are some exercises:\n\n" + "\n".join(fallback_exercises)
    
    # Fallback if Gemini unavailable
    fallback_exercises = pick_exercises_for_part("full body", bmi_val)
    return "⚠️ AI model unavailable. Here are some exercises:\n\n" + "\n".join(fallback_exercises)

# Define conversation states
WEIGHT, HEIGHT, GENDER, AGE = range(4)

# --- CONVERSATION HANDLERS (The Setup Wizard) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the conversation and asks for weight."""
    await update.message.reply_text(
        "👋 Hello! I'm your fitness assistant.\n\n"
        "To give you the best advice, I need to set up your profile.\n"
        "First, please enter your **Weight in kg** (e.g., 70):",
        parse_mode="Markdown"
    )
    return WEIGHT

async def receive_weight(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stores weight and asks for height."""
    try:
        weight = float(update.message.text)
        context.user_data['weight'] = weight
        await update.message.reply_text("Great. Now, enter your **Height in cm** (e.g., 175):", parse_mode="Markdown")
        return HEIGHT
    except ValueError:
        await update.message.reply_text("⚠️ Please enter a valid number for weight (e.g., 70.5). Try again:")
        return WEIGHT

async def receive_height(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stores height and asks for gender."""
    try:
        height = float(update.message.text)
        context.user_data['height'] = height
        
        # Create buttons for gender selection
        reply_keyboard = [["Male", "Female"]]
        await update.message.reply_text(
            "Got it. What is your **Gender**?", 
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True),
            parse_mode="Markdown"
        )
        return GENDER
    except ValueError:
        await update.message.reply_text("⚠️ Please enter a valid number for height (e.g., 170). Try again:")
        return HEIGHT

async def receive_gender(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stores gender and asks for age."""
    gender = update.message.text
    if gender not in ["Male", "Female"]:
        await update.message.reply_text("Please tap one of the buttons (Male or Female).")
        return GENDER
    
    context.user_data['gender'] = gender
    await update.message.reply_text("Almost done! Enter your **Age**:", reply_markup=ReplyKeyboardRemove(), parse_mode="Markdown")
    return AGE

async def receive_age(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stores age and ends the setup."""
    try:
        age = int(update.message.text)
        context.user_data['age'] = age
        
        # Summary
        w = context.user_data['weight']
        h = context.user_data['height']
        g = context.user_data['gender']
        
        await update.message.reply_text(
            f"✅ **Profile Saved!**\n\nStats: {g}, {age}y/o, {w}kg, {h}cm.\n\n"
            "You can now ask me anything about workouts or nutrition!",
            parse_mode="Markdown"
        )
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text("⚠️ Please enter a valid whole number for age (e.g., 25). Try again:")
        return AGE

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancels and ends the conversation."""
    await update.message.reply_text("Setup canceled. Type /start to try again.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

# --- CHAT LOGIC (The Actual Bot) ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles normal chat messages after profile is set."""
    user_text = update.message.text

    # Check if user has completed the setup
    if 'weight' not in context.user_data:
        await update.message.reply_text("⚠️ I don't know your stats yet. Please tap /start to set up your profile.")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Retrieve stored data
    weight = context.user_data['weight']
    height = context.user_data['height']
    gender = context.user_data['gender']
    age = context.user_data['age']

    # Get history (Optional: You can implement simple history storage here if needed)
    chat_history = [] 

    response = chat_with_bmi(
        question=user_text,
        weight=weight,
        height=height,
        gender=gender,
        age=age,
        chat_history=chat_history
    )

    try:
        if len(response) > 4096:
            for x in range(0, len(response), 4096):
                await update.message.reply_text(response[x:x+4096])
        else:
            await update.message.reply_text(response)
    except Exception as e:
        print(f"❌ FAILED TO SEND MESSAGE: {e}")
        await update.message.reply_text("Error sending response. Try a simpler question.")

# --- MAIN ---

def main():
    if TELEGRAM_BOT_TOKEN is None:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing! Check your .env file.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Create the conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_weight)],
            HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_height)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_gender)],
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_age)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # Add handlers
    app.add_handler(conv_handler) # Handles /start and the setup flow
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)) # Handles regular chat

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()

