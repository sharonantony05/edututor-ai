#merged with css
!pip install transformers torch gradio huggingface_hub PyPDF2 -q

import gradio as gr
import PyPDF2
import pandas as pd
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------
# Load IBM Granite Model
# --------------------
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# --------------------
# Local Database (CSV)
# --------------------
DATA_FILE = "user_data.csv"
try:
    pd.read_csv(DATA_FILE)
except:
    df = pd.DataFrame(columns=["timestamp", "username", "action", "details"])
    df.to_csv(DATA_FILE, index=False)

def save_to_history(username, action, details):
    df = pd.read_csv(DATA_FILE)
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "action": action,
        "details": details
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# --------------------
# Core Functions
# --------------------
def edu_tutor_ai(prompt, username):
    answer = generate_response(prompt, max_length=400)
    save_to_history(username, "chat", f"Q: {prompt} | A: {answer[:100]}...")
    return answer

def extract_text_from_file(file):
    text = ""
    try:
        if file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif file.name.endswith(".txt"):
            text = file.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="ignore")
        else:
            return "❌ Unsupported file type. Please upload PDF or TXT."
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"
    return text[:3000] if text else "❌ No readable text found in file."

def summarize_notes(file, username):
    text = extract_text_from_file(file)
    if text.startswith("❌"):
        return text
    prompt = f"Summarize the following study notes:\n\n{text}"
    summary = generate_response(prompt, max_length=600)
    save_to_history(username, "summary", summary[:200])
    return summary

def generate_quiz_from_file(file, username):
    text = extract_text_from_file(file)
    if text.startswith("❌"):
        return text
    prompt = f"Create 5 quiz questions (with answers) from the following notes:\n\n{text}"
    quiz = generate_response(prompt, max_length=700)
    save_to_history(username, "quiz_file", quiz[:200])
    return quiz

def quiz_generator(concept, username):
    prompt = f"Generate 5 quiz questions about {concept} with different question types. At the end, provide an ANSWERS section:"
    quiz = generate_response(prompt, max_length=700)
    save_to_history(username, "quiz_topic", quiz[:200])
    return quiz

def view_history():
    df = pd.read_csv(DATA_FILE)
    return df.tail(15).to_string(index=False)

# --------------------
# Login System
# --------------------
USER_CREDENTIALS = {"student": "password123", "teacher": "admin123"}
current_user = {"name": ""}

def check_login(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        current_user["name"] = username
        save_to_history(username, "login", "User logged in successfully")
        return gr.update(visible=False), gr.update(visible=True), f"✅ Welcome {username}!"
    else:
        return gr.update(visible=True), gr.update(visible=False), "❌ Invalid login. Try again."

# --------------------
# Custom CSS (Colorful UI)
# --------------------
custom_css = """
body {
    background: linear-gradient(135deg, #89f7fe, #66a6ff);
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    background: white;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
}
h1 {
    text-align: center; 
    color: #2c3e50; 
    font-size: 32px; 
    margin-bottom: 5px;
}
h2 {
    text-align: center; 
    color: #34495e; 
    font-size: 18px; 
    margin-bottom: 25px;
}
button {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white !important;
    border: none !important;
    font-weight: bold;
    border-radius: 10px !important;
    transition: 0.3s;
}
button:hover {
    background: linear-gradient(90deg, #ff758c, #ff7eb3);
    transform: scale(1.05);
}
input, textarea {
    border-radius: 10px !important;
    border: 1px solid #dcdcdc !important;
    padding: 8px;
}
"""

# --------------------
# Gradio UI
# --------------------
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>🌟 EduTutor AI</h1>")
    gr.Markdown("<h2>Your Colorful Learning Companion</h2>")

    # Login
    with gr.Group(visible=True) as login_box:
        username = gr.Textbox(label="👤 Username")
        password = gr.Textbox(label="🔑 Password", type="password")
        login_btn = gr.Button("Login")
        login_msg = gr.Label("")

    # App (hidden until login)
    with gr.Group(visible=False) as app_box:
        with gr.Tab("💬 Ask AI"):
            user_input = gr.Textbox(label="💡 Ask EduTutor AI", placeholder="Type your question here...", lines=3)
            submit_btn = gr.Button("✨ Ask Now")
            output = gr.Textbox(label="🤖 AI Response", lines=10)
            submit_btn.click(lambda q: edu_tutor_ai(q, current_user["name"]), inputs=user_input, outputs=output)

        with gr.Tab("📄 Notes Upload"):
            file_input = gr.File(label="Upload Notes (PDF/TXT)")
            summarize_btn = gr.Button("📌 Summarize Notes")
            quiz_btn = gr.Button("📝 Generate Quiz")
            summary_output = gr.Textbox(label="📌 Summary", lines=8)
            quiz_output = gr.Textbox(label="📝 Quiz Questions", lines=12)
            summarize_btn.click(lambda f: summarize_notes(f, current_user["name"]), inputs=file_input, outputs=summary_output)
            quiz_btn.click(lambda f: generate_quiz_from_file(f, current_user["name"]), inputs=file_input, outputs=quiz_output)

        with gr.Tab("📝 Topic Quiz"):
            quiz_input = gr.Textbox(label="Enter a topic", placeholder="e.g., Physics")
            quiz_btn2 = gr.Button("Generate Quiz")
            quiz_output2 = gr.Textbox(label="Quiz Questions", lines=12)
            quiz_btn2.click(lambda c: quiz_generator(c, current_user["name"]), inputs=quiz_input, outputs=quiz_output2)

        with gr.Tab("📜 History"):
            history_btn = gr.Button("📂 View Recent Activity")
            history_output = gr.Textbox(label="📜 Last 15 Actions", lines=15)
            history_btn.click(view_history, outputs=history_output)

    login_btn.click(check_login, inputs=[username, password], outputs=[login_box, app_box, login_msg])

demo.launch(share=True)
