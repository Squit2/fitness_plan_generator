# Quick Reference: Critical Fixes

## 🔧 Most Important Fixes to Apply

### 1. Fix Hardcoded Paths (Lines 34-38)
**Before:**
```python
EXERCISE_CSV = r"C:\Users\Admin\Downloads\thesistwocodes\megaGymDataset.csv"
BMI_CSV = r"C:\Users\Admin\Downloads\thesistwocodes\bmiDataset.csv"
OUT_PATH = r"C:\Users\Admin\Downloads\thesistwocodes\sentencetransformers"
DENSE_MODEL_PATH = os.path.join(OUT_PATH, "dense")
```

**After:**
```python
BASE_DIR = Path(__file__).parent
EXERCISE_CSV = BASE_DIR / "megaGymDataset.csv"
BMI_CSV = BASE_DIR / "bmiDataset.csv"
OUT_PATH = BASE_DIR / "sentencetransformers"
DENSE_MODEL_PATH = OUT_PATH / "dense"
```

### 2. Fix Profile Selection IndexError (Line 79)
**Before:**
```python
selected_profile = st.selectbox(
    "Select Profile", 
    options=profile_names, 
    index=profile_names.index(st.session_state["current_user"])
)
```

**After:**
```python
current_index = 0
if st.session_state["current_user"] in profile_names:
    current_index = profile_names.index(st.session_state["current_user"])

selected_profile = st.selectbox(
    "Select Profile", 
    options=profile_names, 
    index=current_index,
    key="profile_selector"
)
```

### 3. Fix Model Directory Handling (Lines 140-157)
**Before:**
```python
def ensure_dense_model_saved(local_path: str, model_name: str):
    os.makedirs(local_path, exist_ok=True)
    if os.path.exists(os.path.join(local_path, "config.json")):
        return True, "exists"
    
    files = os.listdir(local_path)
    if files:
        try:
            shutil.rmtree(local_path)  # ❌ Unnecessary deletion
            os.makedirs(local_path, exist_ok=True)
        except Exception:
            pass
```

**After:**
```python
def ensure_dense_model_saved(local_path: Path, model_name: str):
    local_path.mkdir(parents=True, exist_ok=True)
    config_file = local_path / "config.json"
    if config_file.exists():
        return True, "exists"
    # ✅ No unnecessary deletion
```

### 4. Fix Gemini Model Initialization (Lines 177-185)
**Before:**
```python
# Gemini Setup
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        gemini_model = None
else:
    gemini_model = None
```

**After:**
```python
@st.cache_resource
def get_gemini_model():
    """Get Gemini model instance with proper error handling."""
    if not GOOGLE_API_KEY:
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        st.warning(f"⚠️ Gemini API Error: {e}")
        return None

# Then call: gemini_model = get_gemini_model()
```

### 5. Add Profile Name Validation
**Before:**
```python
if new_name and new_name not in st.session_state["profiles"]:
    # Create profile
elif new_name in st.session_state["profiles"]:
    st.error("User already exists.")
```

**After:**
```python
def validate_profile_name(name):
    """Validate profile name - no empty, no duplicates."""
    if not name or not name.strip():
        return False, "Name cannot be empty."
    if name.strip() in st.session_state["profiles"]:
        return False, "User already exists."
    if len(name.strip()) > 50:
        return False, "Name too long (max 50 characters)."
    return True, ""

# Usage:
is_valid, error_msg = validate_profile_name(new_name)
if is_valid:
    # Create profile
else:
    st.error(error_msg)
```

### 6. Add Unique Keys to Input Widgets
**Before:**
```python
new_weight = st.number_input("Weight (kg)", ..., value=user_data["weight"])
```

**After:**
```python
new_weight = st.number_input(
    "Weight (kg)", 
    ..., 
    value=user_data["weight"],
    key=f"weight_{st.session_state['current_user']}"  # ✅ Prevents state conflicts
)
```

### 7. Improve Error Handling in Retrieval
**Before:**
```python
def retrieve_exercise_context(query, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, idx = gym_index.search(q_emb, top_k)
    return [gym_texts[i] for i in idx[0]]  # ❌ No bounds checking
```

**After:**
```python
def retrieve_exercise_context(query, top_k=TOP_K):
    try:
        q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
        _, idx = gym_index.search(q_emb, top_k)
        return [gym_texts[i] for i in idx[0] if i < len(gym_texts)]  # ✅ Bounds check
    except Exception as e:
        st.error(f"Error retrieving exercises: {e}")
        return []
```

## 🎯 Priority Order

1. **Fix #1** (Paths) - Critical for portability
2. **Fix #2** (IndexError) - Prevents crashes
3. **Fix #3** (Model handling) - Performance
4. **Fix #4** (Gemini init) - Better caching
5. **Fix #5-7** - Quality improvements






