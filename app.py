import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Hospital information
hospitals = [
    {
        "name": "Mayo Clinic",
        "address": "200 First St SW, Rochester, MN 55905, USA",
        "link": "https://www.mayoclinic.org"
    },
    {
        "name": "Cleveland Clinic",
        "address": "9500 Euclid Ave, Cleveland, OH 44195, USA",
        "link": "https://my.clevelandclinic.org"
    },
    {
        "name": "Boston Children's Hospital",
        "address": "300 Longwood Ave, Boston, MA 02115, USA",
        "link": "https://www.childrenshospital.org"
    },
    {
        "name": "Johns Hopkins Hospital",
        "address": "1800 Orleans St, Baltimore, MD 21287, USA",
        "link": "https://www.hopkinsmedicine.org"
    },
    {
        "name": "Stanford Children's Health",
        "address": "725 Welch Rd, Palo Alto, CA 94304, USA",
        "link": "https://www.stanfordchildrens.org"
    }
]

# Streamlit app
st.title("Autism Spectrum Disorder (ASD) Prediction Tool")

# Sidebar for user input features
st.sidebar.header("User Input Features")

# Replace A1-A10 with clear questions
st.sidebar.subheader("Behavioral Traits")
q1 = st.sidebar.selectbox("1. I prefer to do things the same way over and over again.", ["No", "Yes"])
q2 = st.sidebar.selectbox("2. I find it hard to make small talk.", ["No", "Yes"])
q3 = st.sidebar.selectbox("3. I notice small sounds that others do not.", ["No", "Yes"])
q4 = st.sidebar.selectbox("4. I often focus on details rather than the bigger picture.", ["No", "Yes"])
q5 = st.sidebar.selectbox("5. I find it easy to understand how others are feeling.", ["No", "Yes"])
q6 = st.sidebar.selectbox("6. I enjoy social chit-chat.", ["No", "Yes"])
q7 = st.sidebar.selectbox("7. I am good at remembering phone numbers or license plates.", ["No", "Yes"])
q8 = st.sidebar.selectbox("8. I find it hard to imagine what it's like to be someone else.", ["No", "Yes"])
q9 = st.sidebar.selectbox("9. I enjoy doing things spontaneously.", ["No", "Yes"])
q10 = st.sidebar.selectbox("10. I get upset if my daily routine is disrupted.", ["No", "Yes"])

# Other input fields
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", ["f", "m"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["White-European", "Latino", "Others", "Black", "Asian", "Middle Eastern", "Pasifika", "South Asian", "Hispanic", "Turkish", "others"])
jundice = st.sidebar.selectbox("Jundice", ["no", "yes"])
austim = st.sidebar.selectbox("Austim", ["no", "yes"])
contry_of_res = st.sidebar.selectbox("Country of Residence", ["United States", "Brazil", "Spain", "Egypt", "New Zealand", "Bahamas", "Burundi", "Austria", "Argentina", "Jordan", "Ireland", "United Arab Emirates", "Afghanistan", "Lebanon", "United Kingdom", "South Africa", "Italy", "Pakistan", "Bangladesh", "Chile", "France", "China", "Australia", "Canada", "Saudi Arabia", "Netherlands", "Romania", "Sweden", "Tonga", "Oman", "India", "Philippines", "Sri Lanka", "Sierra Leone", "Ethiopia", "Viet Nam", "Iran", "Costa Rica", "Germany", "Mexico", "Russia", "Armenia", "Iceland", "Nicaragua", "Hong Kong", "Japan", "Ukraine", "Kazakhstan", "AmericanSamoa", "Uruguay", "Serbia", "Portugal", "Malaysia", "Ecuador", "Niger", "Belgium", "Bolivia", "Aruba", "Finland", "Turkey", "Nepal", "Indonesia", "Angola", "Azerbaijan", "Iraq", "Czech Republic", "Cyprus"])
used_app_before = st.sidebar.selectbox("Used App Before", ["no", "yes"])
result = st.sidebar.number_input("Result", min_value=0, max_value=10, value=5)
age_desc = st.sidebar.selectbox("Age Description", ["18 and more"])
relation = st.sidebar.selectbox("Relation", ["Self", "Parent", "Health care professional", "Relative", "Others"])

# Create a DataFrame for user input
input_data = pd.DataFrame({
    'A1_Score': [1 if q1 == "Yes" else 0],
    'A2_Score': [1 if q2 == "Yes" else 0],
    'A3_Score': [1 if q3 == "Yes" else 0],
    'A4_Score': [1 if q4 == "Yes" else 0],
    'A5_Score': [1 if q5 == "Yes" else 0],
    'A6_Score': [1 if q6 == "Yes" else 0],
    'A7_Score': [1 if q7 == "Yes" else 0],
    'A8_Score': [1 if q8 == "Yes" else 0],
    'A9_Score': [1 if q9 == "Yes" else 0],
    'A10_Score': [1 if q10 == "Yes" else 0],
    'age': [age],
    'gender': [gender],
    'ethnicity': [ethnicity],
    'jundice': [jundice],
    'austim': [austim],
    'contry_of_res': [contry_of_res],
    'used_app_before': [used_app_before],
    'result': [result],
    'age_desc': [age_desc],
    'relation': [relation]
})

# Display user input
st.subheader("User Input Features")
st.write(input_data)

# Encode categorical variables
label_encoder = LabelEncoder()
for col in ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation']:
    input_data[col] = label_encoder.fit_transform(input_data[col])

# Normalize numerical features using the saved scaler
input_data = scaler.transform(input_data)

# Predict ASD
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The individual is predicted to have ASD.")
        
        # Display precautions
        st.subheader("Precautions for Individuals with ASD")
        st.write("""
        1. **Early Intervention**: Seek early diagnosis and intervention to improve developmental outcomes.
        2. **Structured Environment**: Create a predictable and structured environment to reduce anxiety.
        3. **Communication Support**: Use clear, simple language and visual aids to enhance communication.
        4. **Sensory Sensitivities**: Identify and minimize sensory triggers (e.g., loud noises, bright lights).
        5. **Social Skills Training**: Encourage social interactions in a supportive and controlled environment.
        6. **Parent and Caregiver Support**: Educate yourself about ASD and connect with support groups.
        """)

        # Display top hospitals
        st.subheader("Top Hospitals for ASD Treatment")
        for hospital in hospitals:
            st.write(f"**{hospital['name']}**")
            st.write(f"Address: {hospital['address']}")
            st.write(f"Website: [Visit Website]({hospital['link']})")
            st.write("---")
    else:
        st.success("The individual is predicted to not have ASD.")