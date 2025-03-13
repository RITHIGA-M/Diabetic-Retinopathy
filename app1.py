import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyrebase
import numpy as np
from PIL import Image, ImageOps
import io
from fpdf import FPDF
from io import BytesIO

# ✅ Firebase Configuration
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
db = firebase.database()

# ✅ Load Model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model("dr_weights.h5")
    return model

model = load_trained_model()
st.success("✅ Model Loaded Successfully")

st.title("Diabetic Retinopathy Diagnosis")

# ✅ Frontend Design
html_temp = """
    <div style="background:linear-gradient(to bottom, #66ccff 0%, #ff99cc 100%);padding:10px">
    <h1 style="color:white;text-align:center;"><em>EyeDR</em></h1>
    </div>
    <br></br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# ✅ Input Fields
name = st.text_input("Enter Patient Name")
file = st.file_uploader("Upload an Eye Retinal Image", type=["jpg", "png", "jpeg"])

# ✅ Function for Image Prediction
def import_and_predict(image_data, model):
    try:
        size = (224, 224)
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image) / 255.0

        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        img_reshape = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"🚨 Error Processing Image: {e}")
        return None

# ✅ Handle File Upload
if file is None:
    st.warning("⚠️ Please upload an image")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    if model is not None:
        prediction = import_and_predict(image, model)
        
        if prediction is not None:
            class_names = ["NO DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            dr = class_names[np.argmax(prediction)]
            confidence_score = float(np.max(prediction) * 100)

            st.write(f"🎯 **Prediction:** {dr}")
            st.write(f"📊 **Confidence Score:** {confidence_score:.2f}%")

            # ✅ Save to Firebase
            if name.strip():
                db.child("Patients Diabetic Retinopathy prediction").child(name).update({
                    "diabetic_retinopathy": dr,
                    "confidence": confidence_score
                })
                st.success(f"✅ Diagnosis Saved in Firebase for {name}")
            else:
                st.warning("⚠️ Please enter a patient name.")

            # ✅ Function to Generate PDF with Image
            def generate_pdf(name, dr, confidence_score, image):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"Diabetic Retinopathy Report for {name}", ln=True, align="C")
                    pdf.ln(10)  # Add space
                    pdf.cell(200, 10, txt=f"Diabetic Retinopathy Diagnosis: {dr}", ln=True, align="L")
                    pdf.cell(200, 10, txt=f"Confidence Score: {confidence_score:.2f}%", ln=True, align="L")

                    # ✅ Convert Image to Bytes & Save
                    img_bytes = BytesIO()
                    image = image.convert("RGB")  # Ensure RGB format
                    image.save(img_bytes, format="JPEG")
                    img_bytes.seek(0)  

                    # ✅ Save image to a temporary file
                    img_filename = "temp_eye_image.jpg"
                    with open(img_filename, "wb") as f:
                        f.write(img_bytes.read())

                    # ✅ Insert Image into PDF
                    pdf.ln(10)  # Add space before image
                    pdf.image(img_filename, x=50, y=None, w=100)  # Center image

                    # ✅ Output PDF as Bytes
                    pdf_output = pdf.output(dest="S").encode("latin1")
                    return BytesIO(pdf_output)
                
                except Exception as e:
                    st.error(f"🚨 Error generating PDF: {e}")
                    return None

            # ✅ Generate & Download PDF
            if st.button("📥 Generate PDF Report"):
                pdf_bytes = generate_pdf(name, dr, confidence_score, image)  # ✅ Pass image
                
                if pdf_bytes:  # ✅ Ensure valid PDF content
                    st.download_button(
                        label="📄 Download Diagnosis Report",
                        data=pdf_bytes,
                        file_name=f"{name}_EyeDR_Report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("⚠️ PDF generation failed. Please try again.")
