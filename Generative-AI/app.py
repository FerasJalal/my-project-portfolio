import os
import csv
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq
from dotenv import load_dotenv


GROQ_API_KEY = ""  # paste your Groq key here


# Create file ".env" and add:

load_dotenv()
if not GROQ_API_KEY:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# CONFIG
USERS_FILE = "users_data_enriched.csv"
LOG_FILE = "interaction_logs.csv"
MODEL_NAME = "llama-3.3-70b-versatile"


# HELPERS
def ensure_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_id", "user_input", "ai_response", "category", "language"])


def detect_language(text: str) -> str:
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            return "ar"
    return "en"


def categorize_question(text: str) -> str:
    """
    Recognizes:
    - profile retrieval
    - workouts
    - nutrition (calories/macros/mealplan)
    """
    t = text.lower()

    # Profile / user info retrieval
    if any(w in t for w in [
        "my info", "my profile", "my data", "my details", "show my info",
        "what is my weight", "what is my height", "what is my age",
        "profile", "user info",
        "معلوماتي", "بياناتي", "ملفي", "بروفايلي",
        "كم وزني", "وزني كم", "كم طولي", "طولي كم", "كم عمري", "عمري كم",
        "اعطيني معلوماتي", "اعرض معلوماتي"
    ]):
        return "profile"

    # Calories / intake / weight change
    if any(w in t for w in [
        "calorie", "calories", "tdee", "bmr", "cut", "bulk", "deficit", "surplus",
        "lose weight", "fat loss", "gain weight", "weight gain",
        "سعرات", "تنشيف", "زيادة", "عجز", "خسارة وزن", "زيادة وزن"
    ]):
        return "nutrition_calories"

    # Macros / protein
    if any(w in t for w in [
        "protein", "carb", "fat", "macros", "micros", "nutrients",
        "بروتين", "كارب", "دهون", "ماكروز", "مغذيات"
    ]):
        return "nutrition_macros"

    # Meal plan / diet / food
    if any(w in t for w in [
        "meal plan", "meal", "meals", "diet", "food", "nutrition", "bulking",
        "increase muscle", "muscle mass", "mass gain", "clean bulk",
        "خطة غذائية", "دايت", "وجبات", "وجبة", "غذاء", "تغذية", "زيادة عضل", "كتلة عضلية"
    ]):
        return "nutrition_mealplan"

    # Workouts
    if any(w in t for w in [
        "split", "workout", "push", "pull", "legs", "gym", "exercise", "training",
        "تمارين", "جيم", "نادي", "تقسيمة", "صدر", "ظهر", "أرجل", "كتف"
    ]):
        return "workout"

    return "other"


#  (retrieval from CSV)
def build_profile_reply(row: pd.Series, lang: str) -> str:
    if lang == "ar":
        return (
            "معلومات حسابك:\n"
            f"- رقم المستخدم (user_id): {int(row['user_id'])}\n"
            f"- العمر: {int(row['age'])}\n"
            f"- الجنس: {row['gender']}\n"
            f"- الوزن: {float(row['weight_kg'])} كغ\n"
            f"- الطول: {float(row['height_cm'])} سم\n"
            f"- مستوى النشاط: {row['activity_level']}\n"
            f"- الهدف: {row['goal']}\n"
        )
    else:
        return (
            "Your Profile Info:\n"
            f"- user_id: {int(row['user_id'])}\n"
            f"- Age: {int(row['age'])}\n"
            f"- Gender: {row['gender']}\n"
            f"- Weight: {float(row['weight_kg'])} kg\n"
            f"- Height: {float(row['height_cm'])} cm\n"
            f"- Activity Level: {row['activity_level']}\n"
            f"- Goal: {row['goal']}\n"
        )


def extract_requested_profile_fields(text: str) -> list:
    """
    Returns a list of requested fields from the user's message.
    If empty list => user wants full profile.
    """
    t = text.lower()
    fields = []

    # English keywords
    if any(k in t for k in ["user id", "userid", "user_id", "my id", "id number"]):
        fields.append("user_id")
    if any(k in t for k in ["age", "my age", "how old"]):
        fields.append("age")
    if any(k in t for k in ["gender", "my gender"]):
        fields.append("gender")
    if any(k in t for k in ["weight", "my weight"]):
        fields.append("weight_kg")
    if any(k in t for k in ["height", "my height"]):
        fields.append("height_cm")
    if any(k in t for k in ["activity", "activity level"]):
        fields.append("activity_level")
    if any(k in t for k in ["goal", "my goal"]):
        fields.append("goal")

    # Arabic keywords
    if any(k in t for k in ["رقم المستخدم", "يوزر", "معرف", "id"]):
        fields.append("user_id")
    if any(k in t for k in ["العمر", "عمري"]):
        fields.append("age")
    if any(k in t for k in ["الجنس"]):
        fields.append("gender")
    if any(k in t for k in ["الوزن", "وزني"]):
        fields.append("weight_kg")
    if any(k in t for k in ["الطول", "طولي"]):
        fields.append("height_cm")
    if any(k in t for k in ["النشاط", "مستوى النشاط"]):
        fields.append("activity_level")
    if any(k in t for k in ["الهدف", "هدفي"]):
        fields.append("goal")

    # remove duplicates while keeping order
    seen = set()
    final_fields = []
    for f in fields:
        if f not in seen:
            final_fields.append(f)
            seen.add(f)

    return final_fields


def build_profile_reply_smart(row: pd.Series, lang: str, requested_fields: list) -> str:
    """
    If requested_fields is empty => show full profile
    Else => show only requested fields
    """
    labels = {
        "user_id": ("user_id", "رقم المستخدم (user_id)"),
        "age": ("Age", "العمر"),
        "gender": ("Gender", "الجنس"),
        "weight_kg": ("Weight (kg)", "الوزن (كغ)"),
        "height_cm": ("Height (cm)", "الطول (سم)"),
        "activity_level": ("Activity Level", "مستوى النشاط"),
        "goal": ("Goal", "الهدف"),
    }

    # If user asked for general info or no specific fields detected -> full profile
    if len(requested_fields) == 0:
        requested_fields = ["user_id", "age", "gender", "weight_kg", "height_cm", "activity_level", "goal"]

    if lang == "ar":
        lines = ["معلومات حسابك:"]
        for f in requested_fields:
            ar_label = labels[f][1]
            value = row[f]
            lines.append(f"- {ar_label}: {value}")
        return "\n".join(lines)

    else:
        lines = ["Your Profile Info:"]
        for f in requested_fields:
            en_label = labels[f][0]
            value = row[f]
            lines.append(f"- {en_label}: {value}")
        return "\n".join(lines)


def needs_followup_question(user_text: str, category: str, lang: str):
    """
    Returns (True, followup_question) if info is missing.
    Returns (False, None) if no follow-up needed.
    """
    t = user_text.lower()

    if category == "workout":
        has_days = any(k in t for k in [
            "day", "days", "week", "times", "x", "2", "3", "4", "5",
            "يوم", "ايام", "أيام", "بالأسبوع", "مرات"
        ])
        has_goal = any(k in t for k in [
            "fat loss", "muscle gain", "strength", "bulking", "cutting",
            "تنشيف", "زيادة عضل", "قوة", "ضخامة"
        ])

        if not has_days:
            q = "How many days per week can you train?" if lang == "en" else "كم يوم بالأسبوع تقدر تتمرن؟"
            return True, q

        if not has_goal:
            q = "What is your main goal (fat loss / muscle gain / strength)?" if lang == "en" else "شو هدفك الرئيسي؟ (تنشيف / زيادة عضل / قوة)"
            return True, q

        return False, None

    if category.startswith("nutrition"):
        has_goal = any(k in t for k in [
            "fat loss", "muscle gain", "recomposition",
            "تنشيف", "زيادة عضل", "ريكمب"
        ])
        has_preferences = any(k in t for k in [
            "chicken", "rice", "eggs", "vegetarian", "vegan",
            "دجاج", "رز", "بيض", "نباتي"
        ])

        if not has_goal:
            q = "What is your goal (fat loss / muscle gain / recomposition)?" if lang == "en" else "شو هدفك؟ (تنشيف / زيادة عضل / recomposition)"
            return True, q

        if category == "nutrition_mealplan" and not has_preferences:
            q = "Any food preferences or restrictions (e.g., vegetarian, allergies)?" if lang == "en" else "عندك تفضيلات أو حساسية من أكلات معينة؟"
            return True, q

        return False, None

    return False, None


# Detect aggressive / crash diet requests
def is_aggressive_diet_request(text: str) -> bool:
    t = text.lower()

    aggressive_keywords = [
        # English
        "aggressive diet", "crash diet", "extreme diet", "starve", "starving",
        "lose weight fast", "rapid weight loss", "in 1 week", "in a week",
        "water fast", "only water", "detox", "no food",
        "zero carbs", "no carbs at all",
        "800 calories", "900 calories", "1000 calories", "1100 calories",

        # Arabic
        "دايت قاسي", "رجيم قاسي", "دايت سريع", "تنحيف سريع", "حرمان",
        "سعرات قليلة جدا", "سعرات قليلة جداً",
        "800 سعرة", "900 سعرة", "1000 سعرة", "1100 سعرة",
        "بدون اكل", "بدون أكل", "ماء فقط", "ديتوكس", "صيام لعدة أيام",
        "بدون كربوهيدرات", "صفر كارب"
    ]

    if any(k in t for k in aggressive_keywords):
        return True

    for num in ["400", "500", "600", "700"]:
        if f"{num} calories" in t or f"{num} سعرة" in t:
            return True

    if "lose" in t and ("kg" in t or "كيلو" in t) and ("day" in t or "week" in t or "أسبوع" in t or "يوم" in t):
        return True

    return False


def aggressive_diet_warning(lang: str) -> str:
    if lang == "ar":
        return (
            "ما بقدر أوصف دايت/رجيم قاسي أو (Crash Diet).\n\n"
            "ليش؟ لأنه ممكن يسبب آثار جانبية مثل:\n"
            "- دوخة وتعب وخمول\n"
            "- فقدان عضل بدل الدهون\n"
            "- تباطؤ الحرق (الأيض)\n"
            "- نقص فيتامينات ومعادن\n"
            "- جوع شديد + زيادة وزن بسرعة بعده\n\n"
            "بديل آمن: عجز سعرات معتدل + بروتين عالي + تمارين مقاومة.\n"
            "إذا بدك، احكيلي هدفك وبعطيك خطة صحية."
        )
    else:
        return (
            "I can’t recommend an aggressive/crash diet.\n\n"
            "Possible side effects:\n"
            "- dizziness, fatigue, low energy\n"
            "- muscle loss instead of fat loss\n"
            "- slower metabolism over time\n"
            "- nutrient deficiencies\n"
            "- rebound hunger and fast weight regain\n\n"
            "Safer alternative: a moderate calorie deficit + high protein + strength training.\n"
            "If you want, tell me your goal and I’ll suggest a safe plan."
        )


def log_interaction(user_id, user_input, ai_response, category, language):
    ensure_log_file()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_id,
            user_input,
            ai_response,
            category,
            language
        ])


def build_system_prompt(language: str) -> str:
    if language == "ar":
        return (
            "أنت مدرب شخصي ذكي (AI Gym PT).\n\n"
            " قواعد صارمة جداً:\n"
            "1) أجب فقط على سؤال المستخدم الحالي ولا تضف معلومات إضافية غير مطلوبة.\n"
            "2) إذا كان السؤال عن التمارين فقط أعطِ التمارين فقط.\n"
            "3) إذا كان السؤال عن التغذية/الوجبات/السعرات/البروتين أعطِ التغذية فقط.\n"
            "4) إذا كان السؤال خارج الجيم/التمارين/التغذية ارفض بأدب.\n"
            "5) لا تقدم نصيحة طبية.\n"
            "6) اجعل الإجابة منظمة بعناوين ونقاط.\n"
            "7) إذا طلب المستخدم دايت قاسي/سعرات قليلة جداً، ارفض وقدّم تحذير وآثار جانبية وبديلاً آمناً.\n"
        )
    else:
        return (
            "You are an AI Gym Personal Trainer (PT).\n\n"
            " Strict rules:\n"
            "1) Answer ONLY the user's current question. Do NOT add extra info.\n"
            "2) If the question is workout-only, answer workouts only.\n"
            "3) If the question is nutrition/meal plan/calories/protein, answer nutrition only.\n"
            "4) If asked about anything outside gym/workouts/nutrition, refuse politely.\n"
            "5) No medical advice.\n"
            "6) Keep the answer structured (headings + bullet points).\n"
            "7) If the user asks for an aggressive/crash diet, refuse and explain side effects + provide a safer alternative.\n"
        )


def minimal_profile_context(row: pd.Series, language: str) -> str:
    if language == "ar":
        return (
            f"معلومات بسيطة:\n"
            f"- العمر: {int(row['age'])}\n"
            f"- الجنس: {row['gender']}\n"
            f"- الوزن: {float(row['weight_kg'])} كغ\n"
            f"- الهدف: {row['goal']}\n"
        )
    else:
        return (
            f"Minimal user info:\n"
            f"- Age: {int(row['age'])}\n"
            f"- Gender: {row['gender']}\n"
            f"- Weight: {float(row['weight_kg'])} kg\n"
            f"- Goal: {row['goal']}\n"
        )


def nutrition_context(row: pd.Series, language: str) -> str:
    if language == "ar":
        return (
            f"بيانات التغذية:\n"
            f"- TDEE: {round(float(row['TDEE']), 2)}\n"
            f"- السعرات المستهدفة/اليوم: {round(float(row['target_calories']), 2)}\n"
            f"- البروتين (غ/اليوم): {int(row['protein_g'])}\n"
        )
    else:
        return (
            f"Nutrition calculations:\n"
            f"- TDEE: {round(float(row['TDEE']), 2)}\n"
            f"- Target Calories/day: {round(float(row['target_calories']), 2)}\n"
            f"- Protein (g/day): {int(row['protein_g'])}\n"
        )


def call_groq(messages):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Paste it in code or put it in .env")

    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.4,
        max_tokens=900,
    )
    return completion.choices[0].message.content


# Fitness calculations for NEW USER
activity_map = {"low": 1.2, "moderate": 1.55, "high": 1.725}


def calc_bmr(age, gender, weight_kg, height_cm):
    if gender == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def calc_tdee(bmr, activity_level):
    return bmr * activity_map[activity_level]


def calc_target_calories(tdee, goal):
    if goal == "fat_loss":
        return tdee - 500
    if goal == "muscle_gain":
        return tdee + 300
    return tdee - 200


def calc_protein(weight_kg, goal):
    if goal == "muscle_gain":
        return round(2.2 * weight_kg)
    return round(2.0 * weight_kg)


def add_new_user_to_csv(new_user_row: dict):
    df_existing = pd.read_csv(USERS_FILE)
    df_new = pd.DataFrame([new_user_row])
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_csv(USERS_FILE, index=False)


# STREAMLIT UI
st.set_page_config(page_title="AI Gym PT Agent", page_icon="🏋️")
st.title(" AI Gym PT Agent (Groq)")

if not os.path.exists(USERS_FILE):
    st.error(f" File not found: {USERS_FILE}")
    st.stop()

df = pd.read_csv(USERS_FILE)

# Clean values (recommended safety)
df["gender"] = df["gender"].astype(str).str.lower().str.strip()
df["activity_level"] = df["activity_level"].astype(str).str.lower().str.strip()
df["goal"] = df["goal"].astype(str).str.lower().str.strip()

# Validation checks are applied using allowed values:
allowed_gender = {"male", "female"}
allowed_activity = {"low", "moderate", "high"}
allowed_goal = {"fat_loss", "muscle_gain", "recomposition"}

invalid_gender = ~df["gender"].isin(allowed_gender)
invalid_activity = ~df["activity_level"].isin(allowed_activity)
invalid_goal = ~df["goal"].isin(allowed_goal)

if invalid_gender.any() or invalid_activity.any() or invalid_goal.any():
    st.error("Dataset has invalid values in gender/activity_level/goal. Please fix the CSV before running.")
    st.stop()

# Session states
if "chat" not in st.session_state:
    st.session_state.chat = []

if "show_create_user" not in st.session_state:
    st.session_state.show_create_user = False

if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None

# Sidebar
with st.sidebar:
    st.header("Settings")

    language_choice = st.selectbox("Language / اللغة", ["Auto", "English", "Arabic"])

    user_id_input = st.number_input(
        "Enter your user_id",
        min_value=1,
        max_value=999999,
        value=1,
        step=1
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Chat"):
            st.session_state.chat = []
            st.success("Chat cleared")

    with col2:
        if st.button("Add New User"):
            st.session_state.show_create_user = True

# User ID
user_id = int(user_id_input)
user_exists = user_id in df["user_id"].values

# Create new user form
if st.session_state.show_create_user or not user_exists:
    st.subheader("Create New User Profile")

    with st.form("create_user_form"):
        new_user_id = st.number_input(
            "New user_id",
            min_value=1,
            max_value=999999,
            value=int(df["user_id"].max() + 1)
        )
        age = st.number_input("Age", min_value=16, max_value=70, value=22)
        gender = st.selectbox("Gender", ["male", "female"])
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
        activity_level = st.selectbox("Activity Level", ["low", "moderate", "high"])
        goal = st.selectbox("Goal", ["fat_loss", "muscle_gain", "recomposition"])

        submitted = st.form_submit_button(" Create User")

    if submitted:
        bmr = calc_bmr(age, gender, weight_kg, height_cm)
        tdee = calc_tdee(bmr, activity_level)
        target_cal = calc_target_calories(tdee, goal)
        protein = calc_protein(weight_kg, goal)

        new_user_row = {
            "user_id": int(new_user_id),
            "age": age,
            "gender": gender,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "activity_level": activity_level,
            "goal": goal,
            "BMR": bmr,
            "TDEE": tdee,
            "target_calories": target_cal,
            "protein_g": protein
        }

        add_new_user_to_csv(new_user_row)

        st.success(" New user added successfully!")
        st.session_state.show_create_user = False
        st.rerun()

    st.stop()

# Get user row
user_row = df[df["user_id"] == user_id].iloc[0]

st.divider()
st.subheader("Chat")

# Show chat bubbles
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_question = st.chat_input("Type your message...")

if user_question:
    st.session_state.chat.append({"role": "user", "content": user_question})

    detected_lang = detect_language(user_question)
    if language_choice == "English":
        lang = "en"
    elif language_choice == "Arabic":
        lang = "ar"
    else:
        lang = detected_lang

    cat = categorize_question(user_question)

    # Follow-up answering mode
    if st.session_state.pending_followup is not None:
        original = st.session_state.pending_followup["original_question"]
        pending_cat = st.session_state.pending_followup["category"]
        pending_lang = st.session_state.pending_followup["lang"]

        combined_question = original + "\nAdditional details: " + user_question

        st.session_state.pending_followup = None

        user_question = combined_question
        cat = pending_cat
        lang = pending_lang

    # Profile retrieval
    if cat == "profile":
        requested_fields = extract_requested_profile_fields(user_question)
        reply = build_profile_reply_smart(user_row, lang, requested_fields)

        st.session_state.chat.append({"role": "assistant", "content": reply})
        log_interaction(user_id, user_question, reply, cat, lang)

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.stop()

    # Out of domain
    if cat == "other":
        reply = (
            "عذراً، أنا مدرب جيم فقط ولا أستطيع الإجابة عن أسئلة خارج التمارين والتغذية."
            if lang == "ar"
            else
            "Sorry, I can only answer questions about workouts and nutrition."
        )

        st.session_state.chat.append({"role": "assistant", "content": reply})
        log_interaction(user_id, user_question, reply, cat, lang)

        with st.chat_message("assistant"):
            st.markdown(reply)

    else:
        # Aggressive diet refusal
        if cat.startswith("nutrition") and is_aggressive_diet_request(user_question):
            reply = aggressive_diet_warning(lang)

            st.session_state.chat.append({"role": "assistant", "content": reply})
            log_interaction(user_id, user_question, reply, "nutrition_aggressive_refused", lang)

            with st.chat_message("assistant"):
                st.markdown(reply)

            st.stop()

        # Follow-up question mode
        need_follow, follow_q = needs_followup_question(user_question, cat, lang)
        if need_follow:
            st.session_state.pending_followup = {
                "original_question": user_question,
                "category": cat,
                "lang": lang
            }

            st.session_state.chat.append({"role": "assistant", "content": follow_q})
            log_interaction(user_id, user_question, follow_q, "followup_question", lang)

            with st.chat_message("assistant"):
                st.markdown(follow_q)

            st.stop()

        system_prompt = build_system_prompt(lang)

        # Correct routing
        if cat == "workout":
            context = minimal_profile_context(user_row, lang)
        else:
            context = nutrition_context(user_row, lang)

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "system", "content": f"Context:\n{context}"})

        for m in st.session_state.chat:
            messages.append(m)

        with st.spinner("Thinking..."):
            try:
                reply = call_groq(messages)
            except Exception as e:
                reply = f"Error: {e}"

        st.session_state.chat.append({"role": "assistant", "content": reply})
        log_interaction(user_id, user_question, reply, cat, lang)

        with st.chat_message("assistant"):
            st.markdown(reply)

st.divider()
st.caption(" Meal plan + Add new user + Focused answers + Logging + Crash diet safety + Profile retrieval")
