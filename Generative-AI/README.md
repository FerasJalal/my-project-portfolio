# AI Gym PT Agent — Streamlit + Groq + CSV (Demo)

A small Generative AI backend project:
- Backend: **Python + Streamlit**
- Data storage: **CSV (Pandas)**
- Chatbot: server-side call to **Groq Chat Completions API**
- Supports: **English + Arabic**, multi-turn chat, logging, user profile retrieval, safety filters

---

1) Project purpose and features

 Project purpose
This system is a Generative AI Personal Trainer (PT) assistant designed to support gym users by providing:
- workout guidance (splits, exercises, routines)
- nutrition guidance (calories, protein, meal plans)
- profile retrieval from stored user data (CSV)
- safe and controlled responses (refusal of extreme diet requests)

Main features
- Multi-turn chat (conversation continues back-and-forth)
- English + Arabic support (Auto detection + manual selection)
- Domain filtering:
  - answers only workout and nutrition questions
  - refuses unrelated topics politely
- Profile retrieval from CSV:
  - user can ask for personal details (weight, height, age, etc.)
  - returns only the requested info (example: “my weight” → weight only)
- Safety filter:
  - refuses aggressive/crash diet requests
  - provides short warning about side effects and safer alternative
- Add new user:
  - create a new profile if user_id is not found
  - calculates BMR, TDEE, target calories, and protein target
- Logging:
  - logs all interactions to `interaction_logs.csv` for evaluation

---

2) System architecture

High-level architecture (simple flow):

1. User interacts with the system through the Streamlit chat UI  
2. Streamlit backend receives the user message  
3. The system detects:
   - language (Arabic/English)
   - question category (workout/nutrition/profile/other)
4. If user requests profile info:
   - data is retrieved from `users_data_enriched.csv`
5. If workout/nutrition question:
   - the message is sent to Groq Chat Completions API
   - Groq returns an AI-generated response
6. The conversation is stored in Streamlit session state (multi-turn memory)
7. Each interaction is saved to `interaction_logs.csv` for evaluation

Components:
- Frontend: Streamlit UI (chat interface)
- Backend: Python logic (categorization, safety checks, retrieval)
- LLM: Groq API (llama-3.3-70b-versatile)
- Data source: CSV file (`users_data_enriched.csv`)
- Logging storage: CSV file (`interaction_logs.csv`)

---

3) Requirements
- Python 3.10+
- Groq API key
- Install Python libraries:
  - `streamlit`
  - `pandas`
  - `python-dotenv`
  - `groq`
  - `matplotlib`

---

 4) Setup instructions and dependencies

 4.1 Project files
Make sure your folder contains:
- `app.py`
- `users_data_enriched.csv`
- `README.md`

The app will generate:
- `interaction_logs.csv` (created automatically after you start chatting)

 4.2 Install dependencies
From the project directory:
```powershell
pip install streamlit pandas python-dotenv groq matplotlib
4.3 Add Groq API key (two options)
Option A (Recommended) — .env file

Create a file named .env inside the project folder

Add your key:

GROQ_API_KEY=your_key_here
Option B (Quick testing only) — paste in code
Inside app.py:

GROQ_API_KEY = "PUT_YOUR_GROQ_KEY_HERE"
Important:

Never upload your real key to GitHub

5) How to run the demo
From the project directory:

streamlit run app.py
Then open the URL shown in the terminal (example):

http://localhost:8501

6) How to use the system
Inside the app sidebar:

Select language: Auto / English / Arabic

Enter your user_id

Clear Chat button

Add New User button

Supported chat topics:

Workouts: “best shoulder workout”, “push pull legs split”

Nutrition: “how many calories should I eat”, “protein for muscle gain”

Meal plans: “meal plan to increase muscle mass”

Profile retrieval (CSV retrieval):

“my weight” → returns only weight

“my height and age” → returns only height and age

“show my info” → returns full profile information

7) Data files
7.1 Users dataset
File: users_data_enriched.csv

Must contain these columns:

user_id

age

gender

weight_kg

height_cm

activity_level (low / moderate / high)

goal (fat_loss / muscle_gain / recomposition)

BMR

TDEE

target_calories

protein_g

7.2 Interaction logs
File: interaction_logs.csv (auto-generated)

Columns:

timestamp

user_id

user_input

ai_response

category

language

8) Known limitations
The assistant depends on the LLM response quality

The system stores profiles in a local CSV file (not a database)

No login/authentication is implemented (demo/educational version)

The assistant provides general fitness guidance only (not medical advice)