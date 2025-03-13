from flask import Flask, render_template, request,redirect, jsonify, send_file, url_for
from flask import session
from flask import send_file
import subprocess
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyrebase
import PyPDF2
import os
import numpy as np
from PIL import Image, ImageOps
import io
from fpdf import FPDF
from reportlab.pdfgen import canvas
#from io import BytesIO
from flask import send_from_directory
import re 
import time
from datetime import datetime
import random
from io import BytesIO 
import firebase_admin
from firebase_admin import credentials, storage, db
import base64
import json

def save_report_as_pdf(report_text, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("ArialUnicode", "", "arialuni.ttf", uni=True)
    pdf.set_font("ArialUnicode", "", 12)

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

    # ✅ Step 2: Convert text to Latin-1 compatible format
    report_text = remove_emojis(report_text)  
    report_text = report_text.encode('latin-1', 'replace').decode('latin-1') 

    pdf.multi_cell(0, 10, report_text)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'F')
    pdf_output.seek(0)
    
    return pdf_output

app = Flask(__name__)

@app.route('/static/css/<path:filename>')
def custom_static(filename):
    return send_from_directory('static/css', filename)


# Firebase Configuration
firebaseConfig = {
    "apiKey": "AIzaSyBUc09vVr13lufna3PcRKIvkScJMmL84F8",
    "authDomain": "diabetic-retinopathy-d9ad0.firebaseapp.com",
    "databaseURL": "https://diabetic-retinopathy-d9ad0-default-rtdb.firebaseio.com",
    "projectId": "diabetic-retinopathy-d9ad0",
    "storageBucket": "diabetic-retinopathy-d9ad0.appspot.com",
    "messagingSenderId": "784479554899",
    "appId": "1:784479554899:web:741bb70f664687c789e5a6",
    "measurementId": "G-T9E5GHPJ65"
}
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()
# Define the model path

# Initialize Firebase Admin SDK
cred = credentials.Certificate("C:\\Users\\rithi\\Downloads\\diabetic-retinopathy-d9ad0-firebase-adminsdk-fbsvc-a1194e67ca.json")  
# ✅ Initialize Firebase Admin SDK with Storage
firebase_admin.initialize_app(cred, {
    'storageBucket': 'diabetic-retinopathy-d9ad0.firebasestorage.app'  # ✅ Replace with your actual bucket name
})
# ✅ Get Firebase Storage Bucket (🔥 FIX HERE 🔥)
bucket = firebase_admin.storage.bucket()


tips = [
    "Stay hydrated and take deep breaths!",
    "You’re doing great! Consistency is key.",
    "Remember to relax your shoulders and straighten your spine.",
    "Keep your mind clear, and focus on your breathing.",
    "Little progress each day adds up to big results. Keep going!",
    "Your body is your temple. Take care of it.",
    "Breathe deeply and release any tension. You got this!",
    "Every time you practice, you get stronger physically and mentally."
]
# ✅ Function to trigger Text-to-Speech
@app.route('/speak', methods=['POST'])
def speak():
    tip = request.form.get('tip')
    return f'<script>let msg = new SpeechSynthesisUtterance("{tip}"); window.speechSynthesis.speak(msg);</script>'

MODEL_PATH = "C:\\Users\\rithi\\Downloads\\Diabetic_Retinopathy_Project\\Diabetic_Retinopathy_Project\\dr_weights.h5"

# Load the model
try:
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loading model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("✅ Model Loaded Successfully!")
    else:
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"🚨 Model Loading Error: {e}")
    model = None

def import_and_predict(image_data):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    return model.predict(img_reshape)

# Remove Emoji

def remove_emojis(text):
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"u"]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    name = request.form['name']
    image = Image.open(file)

    prediction = import_and_predict(image)
    class_names = ["NO DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    dr = class_names[np.argmax(prediction)]
    confidence_score = float(np.max(prediction) * 100)
    diagnosis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save to Firebase
    db.child("Patients Diabetic Retinopathy prediction").child(name).update({
        "diabetic_retinopathy": dr,
        "confidence": confidence_score,
        "diagnosis_date": diagnosis_date
    })
    return jsonify({"name": name, "prediction": dr, "confidence": confidence_score})

@app.route("/generate_pdf")
def download_pdf():
    name = request.args.get("name", "Unknown")
    dr = request.args.get("dr", "Not Available")
    confidence_score = request.args.get("confidence_score", "0%")

    pdf_file = generate_pdf(name, dr, confidence_score)

    if pdf_file:
        return send_file(
            pdf_file,
            as_attachment=True,
            download_name=f"{name}_Report.pdf",
            mimetype="application/pdf"
        )
    else:
        return "⚠️ Error generating PDF", 500

@app.route('/summarise')
def summarise_form():
    return render_template("summarise.html")

@app.route('/summarise', methods=['POST'])
def summarise_text():
    if request.method == 'POST':
        text = request.form.get("text") 
        summary = " ".join(text.split()[:10]) + "..."  
        save_summary(text, summary) 
        return render_template("summarise.html", name=summary)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")  
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER  

@app.route("/summarize", methods=["POST"])
def summarizePDF():
    if "file" not in request.files:
        return "No file part", 400
    
    f = request.files["file"]
    
    if f.filename == "":
        return "No selected file", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(file_path)

    return f"File successfully uploaded to {file_path}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template("login.html")  # Ensure 'login.html' exists in 'templates/' folder


app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)  # Create folder if it doesn't exist

# --------------------------------------------------------------
# 🔹 **1. Upload PDF to Firebase Storage**
# --------------------------------------------------------------
def upload_pdf(file_path, file_name):
    try:
        blob = bucket.blob(f"reports/{file_name}")  # Store in 'reports/' folder
        blob.upload_from_filename(file_path)
        blob.make_public()  # Make it publicly accessible

        file_url = blob.public_url  # Get public URL
        return file_url
    except Exception as e:
        print(f"🔥 Error Uploading PDF: {e}")
        return None


# --------------------------------------------------------------
# 🔹 **2. Route to Handle PDF Upload**
# --------------------------------------------------------------
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)  # Save locally before upload

    # ✅ Upload to Firebase Storage
    pdf_url = upload_pdf(file_path, file.filename)

    if pdf_url:
        # ✅ Store details in Firebase Database
        db.child("summaries").push({
            "original_Document": file.filename,
            "file_url": pdf_url
        })
        return jsonify({"success": True, "url": pdf_url})
    else:
        return jsonify({"error": "Failed to upload PDF"}), 500


# --------------------------------------------------------------
# 🔹 **3. Doctor Dashboard to View & Download PDFs**
# --------------------------------------------------------------

@app.route('/doctor_dashboard')
def doctor_dashboard():
    documents = db.child("summaries").get().val()  # Get summaries from Firebase

    pdf_list = []
    if documents:
        for doc_id, doc_data in documents.items():
            pdf_list.append({
                "filename": doc_data.get("original_Document", "Unknown"),
                "url": doc_data.get("file_url", "#")  # Ensure the file_url exists
            })

    return render_template('doctor_dashboard.html', pdf_list=pdf_list)

# --------------------------------------------------------------
# 🔹 **4. Patient View - Retrieve Their Reports**
# --------------------------------------------------------------
@app.route('/patient_reports/<name>')
def patient_reports(name):
    reports = db.child("Documents").get().val()

    patient_reports = []
    if reports:
        for doc_id, doc_data in reports.items():
            if name in doc_data.get("original_Document", ""):
                patient_reports.append({
                    "filename": doc_data.get("original_Document", "Unknown"),
                    "url": doc_data.get("file_url", "#")
                })

    return render_template('patient_reports.html', name=name, reports=patient_reports)

@app.route('/posebot')
def posebot():
    return render_template('posebot.html')

@app.route('/posebots') 
def posebots():
    return render_template("posebots.html")

@app.route("/patient_dadhboard")
def patient_dashboard():
    return render_template("patient_dashboard.html")

@app.route("/patient_main")
def patient_main():
    return render_template("patient_main.html")

@app.route("/save-session", methods=["POST"])
def save_session():
    from flask import request, jsonify
    data = request.json
    session_data = {
        "user": data.get("user", "unknown"),
        "duration": data.get("duration", "00:00:00"),
        "timestamp": data.get("timestamp", "Not Provided")
    }
    db.child("sessions").push(session_data)
    return jsonify({"status": "success", "message": "Session saved!"}), 200

@app.route('/palming',methods=['POST','GET'])
def palming():
    name = request.args.get('name')
    random_tip = random.choice(tips)
    url="https://teachablemachine.withgoogle.com/models/vKItIykyq/"
    return render_template('posebot.html', url=url, name=name, tip=random_tip)

@app.route('/trataka',methods=['POST','GET'])
def trataka():
    name = request.args.get('name')
    random_tip = random.choice(tips)
    url="https://teachablemachine.withgoogle.com/models/YBbEKwztQ/"
    return render_template('posebot.html', url=url, name=name, tip=random_tip)

@app.route('/bhramamudra',methods=['POST','GET'])
def bhramamudra():
    name = request.args.get('name')
    random_tip = random.choice(tips)
    url="https://teachablemachine.withgoogle.com/models/znvpEncRJ/"
    return render_template('posebot.html', url=url, name=name, tip=random_tip)

@app.route('/pranayama',methods=['POST','GET'])
def pranayama():
    name = request.args.get('name')
    random_tip = random.choice(tips)
    url="https://teachablemachine.withgoogle.com/models/ZW47sepej/"
    return render_template('posebot.html', url=url, name=name, tip=random_tip)

@app.route('/parvatasana',methods=['POST','GET'])
def parvatasana():
    name = request.args.get('name')
    random_tip = random.choice(tips)
    url="https://teachablemachine.withgoogle.com/models/JnfkxNVh3/"
    return render_template('posebot.html', url=url, name=name, tip=random_tip)

@app.route('/login/patient',methods=['POST','GET'])
def login_patient():
    flag=0
    flag2=0
    int_features=[x for x in request.form.values()]
    print(int_features)
    name=int_features[0]
    email=int_features[1]
    password=int_features[2]
    users = db.child("Patient Signup").child(name).get()
    if(users.each()!=None):
        flag2=1
        for user in users.each():
            if(user.val()==email):
                print('found')
                flag=1
    if(flag==1 and flag2==1):
        try:
            auth.sign_in_with_email_and_password(email, password)
            print("Patient Succesfully SignedIn")
            return render_template('patient.html', name=name) 
        except:
            print("Invalid User Or Password")
    return render_template('login.html')

@app.route('/login/doctor',methods=['POST','GET'])
def login_doctor():
    flag=0
    flag2=0
    int_features=[x for x in request.form.values()]
    print(int_features)
    name=int_features[0]
    email=int_features[1]
    password=int_features[2]
    users = db.child("Doctors Signup").child(name).get()
    if(users.each()!=None):
        flag2=1
        for user in users.each():
            if(user.val()==email):
                print('found')
                flag=1
    if(flag==1 and flag2==1):
        try:
            auth.sign_in_with_email_and_password(email, password)
            print("Doctor Succesfully SignedIn")
            return render_template('doctor.html', name=name) 
        except:
            print("Invalid User Or Password")
    return render_template('login.html')


@app.route('/signup/doctor',methods=['POST','GET'])
def signup_doctor():
    int_features=[x for x in request.form.values()]
    print(int_features)
    name=int_features[0]
    email=int_features[1]
    password=int_features[2]
    confirmpasswd=int_features[3]
    designation=int_features[4]
    hospital=int_features[5]
    if(password==confirmpasswd):
        try:
            auth.create_user_with_email_and_password(email,password)
            print("Doctor Created")
            data={"name":name, "email": email, "designation":designation, "hospital":hospital }
            print(email)
            db.child("Doctors Signup").child(name).set(data)
            return render_template('doctor.html', name=name)
        except:
            print("Email already Exists")
    return render_template('signup.html')

@app.route('/signup/patient',methods=['POST','GET'])
def signup_patient():
    int_features=[x for x in request.form.values()]
    print(int_features)
    name=int_features[0]
    email=int_features[1]
    password=int_features[2]
    confirmpassword=int_features[3]
    height=int_features[4]
    weight=int_features[5]
    age=int_features[6]
    if(password==confirmpassword):
        try:
            auth.create_user_with_email_and_password(email,password)
            print("Patient Created")
            data={"name":name, "email": email, "age":age, "height":height, "weight":weight}
            print(email)
            db.child("Patient Signup").child(name).set(data)
            return render_template('patient.html', name=name)
        except:
            print("Email already Exists")
    return render_template('signup.html')

@app.route('/view_patients', methods=['POST', 'GET'])
def view_patients():
    patients = []
    
    # ✅ Fetch patient details
    users = db.child("Patient Signup").get()
    
    # ✅ Fetch DR predictions (Debugging)
    dr_predictions = db.child("Patients Diabetic Retinopathy prediction").get()
    
    print("Patient Signup Data:", users.val())  # Debug: Check data structure
    print("DR Predictions Data:", dr_predictions.val())  # Debug: Check DR predictions
    
    if users.each():
        for user in users.each():
            patient_info = user.val()
            name = user.key()  # Assuming 'name' is the key
            
            # ✅ Fetch DR status separately
            dr_data = db.child("Patients Diabetic Retinopathy prediction").child(name).get().val()
            print(f"DR data for {name}: ", dr_data)  # Debug: Print each patient's DR status
            
            dr_status = dr_data.get('diabetic_retinopathy', 'Not Predicted Yet') if dr_data else 'Not Predicted Yet'
            
            # ✅ Attach DR status to the patient's data
            patient_info["DR"] = dr_status  
            patients.append(patient_info)
    
    print("Final Patients List:", patients)  # Debugging - Check if DR values are attached correctly
    
    return render_template('patientlist.html', patients=patients)


@app.route('/report', methods=['POST','GET'])
def report():
    patients=[]
    users = db.child("Patient Signup").get()
    for user in users.each():
        patients.append(user.val())
    int_features=[x for x in request.form.values()]
    print(int_features)
    name=int_features[0]
    report=int_features[1]
    db.child("Patient Signup").child(name).update({"report":report})
    return redirect('/view_patients')

@app.route('/patientreport', methods=['POST', 'GET'])
def patientreport():
    if request.method == 'POST':
        name = request.form.get('name')  # Ensure 'name' is retrieved correctly
    else:
        name = request.args.get('name')

    # ✅ Fetch patient details
    patient_data = db.child("Patient Signup").child(name).get().val()

    if not patient_data:
        return "Error: Patient not found in Firebase", 404

    # ✅ Fetch DR value separately
    dr_data = db.child("Patients Diabetic Retinopathy prediction").child(name).get().val()
    dr = dr_data.get('diabetic_retinopathy', 'Not Predicted Yet') if dr_data else 'Not Predicted Yet'

    return render_template('report.html', name=name, patient=patient_data, dr=dr)

@app.route('/summarise-section')
def view_summarise_section():
    return render_template('summarise-section.html')  # Ensure the HTML file exists in the templates folder

@app.route('/summaries')
def view_summaries():
    summaries = get_summaries()
    return render_template("summaries.html", summaries=summaries)

# Function to Save Summary
def save_summary(original_text, summary):
    try:
        db.child("summaries").push({"original_text": original_text, "summary": summary})
    except Exception as e:
        print(f"Error saving summary: {e}")

# Function to Get All Summaries
def get_summaries():
    try:
        summaries = db.child("summaries").get()
        return summaries.val().values() if summaries.val() else []
    except Exception as e:
        print(f"Error fetching summaries: {e}")
        return []

@app.route('/summarisePDF', methods=['POST'])
def summarisePDF():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]

    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file locally
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(file_path)

    try:
        # Read PDF and extract text
        text = ""
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + " "

        if not text.strip():
            return jsonify({"error": "PDF text extraction failed. Try another file."}), 400

        # Generate a summary (Basic summarization)
        summary = " ".join(text.split()[:50]) + "..."

        # Store the summary in Firebase
        db.child("summaries").push({
            "original_Document": f.filename,
            "summary": summary
        })

        return render_template("summarise-section.html", name=summary)

    except Exception as e:
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'img', 'icon'), 'favicon.png', mimetype='image/png')

@app.route('/recommend', methods=['POST','GET'])
def recommend():
    if request.method == 'POST':
        name = request.form.get('name')  # Capture the name from the form
    else:
        name = request.args.get('name')

    # ✅ Fetch the latest DR prediction from correct path
    user_data = db.child("Patients Diabetic Retinopathy prediction").child(name).get().val()

    # ✅ Prevent TypeError by setting default if None
    if user_data is None:
        user_data = {}

    # ✅ Always check for latest DR prediction
    dr = user_data.get('diabetic_retinopathy', 'Not Predicted Yet')
    return render_template('posebots.html', name=name, dr=dr)


import subprocess
import threading

# ✅ Automatically Run Streamlit
@app.route('/app1')
def app1():
    streamlit_path = os.path.join(os.getcwd(), 'app1.py')
    subprocess.Popen(['streamlit', 'run', streamlit_path, '--server.headless', 'true', '--server.enableCORS', 'false', '--server.enableXsrfProtection', 'false', '--browser.gatherUsageStats', 'false'], shell=True)

    # ✅ Automatically fetch the latest DR prediction from Streamlit
    name = request.args.get('name')

    # ✅ Fetch latest prediction from Firebase
    user_data = db.child("Patients Diabetic Retinopathy prediction").child(name).get().val()

    # ✅ Check if 'dr' exists and update in Firebase
    if user_data and 'app1' in user_data:
        dr = user_data['app1']
    else:
        dr = 'Not Predicted Yet'

    # ✅ Store the prediction in Firebase (if new prediction is found)
    db.child("Patients Diabetic Retinopathy prediction").child(name).update({"app1": dr})



    streamlit_path = os.path.join(os.getcwd(), 'app1.py')
    subprocess.Popen(['streamlit', 'run', streamlit_path], shell=True)
    return redirect('http://localhost:8501')
 

@app.route('/back_to_dashboard')
def back_to_dashboard():
    name = request.args.get('name')
    return redirect(url_for('patient', name=name))

@app.route('/latest_dr', methods=['GET'])
def latest_dr():
    name = request.args.get('name')
    if not name:
        return jsonify({'dr': 'Name not provided'})

    # Correct Firebase Path Access
    user_data = db.child("Patients Diabetic Retinopathy prediction").child(name).get().val()
    
    if user_data is None:
        return jsonify({'dr': 'Not Predicted Yet'})

    # Ensures 'diabetic_retinopathy' is retrieved
    dr = user_data.get('diabetic_retinopathy', 'Not Predicted Yet')
    return jsonify({'dr': dr})


if __name__ == '__main__':
    app.run(debug=True)
