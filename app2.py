
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import joblib
#st.set_option('deprecation.showPyplotGlobalUse', False)

coxnetUT=joblib.load("C:\\Users\\rithi\\Downloads\\HackerCamp-Visara-main\\HackerCamp-Visara-main\\coxnetUT.pkl")
coxnetTR=joblib.load("C:\\Users\\rithi\\Downloads\\HackerCamp-Visara-main\\HackerCamp-Visara-main\\coxnetTR.pkl")

def funcUT(age,typee,UT_group,eye_type):
  type1=0
  type2=0

  eye1=0
  eye2=0

  laser_type1=0
  laser_type2=0

  ug6=1
  ug8=0
  ug9=0
  ug10=0
  ug11=0
  ug12=0
  
  if(eye_type=='left'):
    eye1=1
    eye2=0
  if(eye_type=='right'):
    eye1=0
    eye2=1
  if(typee==1):
    type1=1
    type2=0
  if(typee==2):
    type1=0
    type2=1
  if(UT_group==6):
    ug6=1
    ug8=0
    ug9=0
    ug10=0
    ug11=0
    ug12=0
  if(UT_group==8):
    ug6=0
    ug8=1
    ug9=0
    ug10=0
    ug11=0
    ug12=0
  if(UT_group==9):
    ug6=0
    ug8=0
    ug9=1
    ug10=0
    ug11=0
    ug12=0
  if(UT_group==10):
    ug6=0
    ug8=0
    ug9=0
    ug10=1
    ug11=0
    ug12=0
  if(UT_group==11):
    ug6=0
    ug8=0
    ug9=0
    ug10=0
    ug11=1
    ug12=0
  if(UT_group==12):
    ug6=0
    ug8=0
    ug9=0
    ug10=0
    ug11=0
    ug12=1
  
  return [[age,type1,type2,ug6,ug8,ug9,ug10,ug11,ug12,eye1,eye2]]

def funcTR(age,laser_type,typee,TR_group,eye_type):
  type1=0
  type2=0

  eye1=0
  eye2=0

  laser_type1=0
  laser_type2=0

  ug6=1
  ug8=0
  ug9=0
  ug10=0
  ug11=0
  ug12=0  

  if(eye_type=='left'):
    eye1=1
    eye2=0
  if(eye_type=='right'):
    eye1=0
    eye2=1
  if(laser_type=='Xenon'):
    laser_type1=1
    laser_type2=0
  if(laser_type=='Argon'):
    laser_type1=0
    laser_type2=1
  if(typee==1):
    type1=1
    type2=0
  if(typee==2):
    type1=0
    type2=1
  if(TR_group==6):
    ug6=1
    ug8=0
    ug9=0
    ug10=0
    ug11=0
    ug12=0
  if(TR_group==8):
    ug6=0
    ug8=1
    ug9=0
    ug10=0
    ug11=0
    ug12=0
  if(TR_group==9):
    ug6=0
    ug8=0
    ug9=1
    ug10=0
    ug11=0
    ug12=0
  if(TR_group==10):
    ug6=0
    ug8=0
    ug9=0
    ug10=1
    ug11=0
    ug12=0
  if(TR_group==11):
    ug6=0
    ug8=0
    ug9=0
    ug10=0
    ug11=1
    ug12=0
  if(TR_group==12):
    ug6=0
    ug8=0
    ug9=0
    ug10=0
    ug11=0
    ug12=1
  
  
  return [[age,laser_type1,laser_type2,type1,type2,ug6,ug8,ug9,ug10,ug11,ug12,eye1,eye2]]

def welcome():
    return "Welcome All"

def eye(age, laser_type,typee,treated_group,untreated_group,eyee):
     # Create figures
    fig, ax = plt.subplots()
    
    # Prediction for untreated group
    surv_funcs = {alpha: coxnetUT.predict_survival_function(funcUT(age, typee, untreated_group, eyee), alpha=alpha) for alpha in coxnetUT.alphas_[:5]}
    for alpha, surv_alpha in surv_funcs.items():
        for fn in surv_alpha:
            ax.step(fn.x, fn(fn.x), where="post", label=f"Untreated α={alpha:.4f}")  # Added label

   # Prediction for treated group
    surv_funcs2 = {alpha: coxnetTR.predict_survival_function(funcTR(age, laser_type, treated_group, untreated_group, eyee), alpha=alpha) for alpha in coxnetTR.alphas_[:5]}
    for alpha, surv_alpha in surv_funcs2.items():
        for fn in surv_alpha:
            ax.step(fn.x, fn(fn.x), where="post", label=f"Treated α={alpha:.4f}")  # Added label
    
    for alpha, surv_alpha in surv_funcs.items():
        for fn in surv_alpha:
            ax.step(fn.x, fn(fn.x), where="post")

    for alpha, surv_alpha in surv_funcs2.items():
        for fn in surv_alpha:
            ax.step(fn.x, fn(fn.x), where="post")

    ax.set_ylim(0, 1)
    ax.legend()
  
    st.pyplot(fig)   # Pass the figure to Streamlit

def main():
  st.title("Diabetic Retinopathy")
  
  html_temp = """
    <div style="background:linear-gradient(to bottom, #66ccff 0%, #ff99cc 100%);padding:10px">
    <h1 style="color:white;text-align:center;"><em>Prognosis</em> </h1>
    </div>
    <br></br>
    """
  
  st.markdown(html_temp,unsafe_allow_html=True)
  st.sidebar.header("Blindness Prognosis")
  st.sidebar.write(''' 
  *This graph predicts the probability of you turning blind in the span of 70 months. The upper curve represents the probability if the groups remain untreated and the lower curve is when the groups are treated
  
  ''')
  age= st.slider("AGE",10,100)
  offer=('Xenon','Argon')
  laser_type=list(range(len(offer)))
  laser_type = st.selectbox("Laser_Type",laser_type,format_func=lambda x:offer[x])
  offer1=(1,2)
  typee= list(range(len(offer1)))
  typee = st.selectbox("Type",typee,format_func=lambda x:offer1[x])
  offer3=(6,7,8,10,11,12)
  treated_group= list(range(len(offer3)))
  treated_group = st.selectbox("Treated Group",treated_group,format_func=lambda x:offer3[x])
  untreated_group= list(range(len(offer3)))
  untreated_group = st.selectbox("untreated_group",untreated_group,format_func=lambda x:offer3[x])
  offer4=('Left','Right')
  eyee=list(range(len(offer4)))
  eyee = st.selectbox("Eye",eyee,format_func=lambda x:offer4[x])
  if st.button("Predict"):
          st.write(eye(age,laser_type,typee,treated_group,untreated_group,eyee))
          st.success("hello")

if __name__=='__main__':
    main()
