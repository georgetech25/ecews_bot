import streamlit as st
import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set page config and global styles
st.set_page_config(page_title="Treatment Failure Prediction", page_icon="💊", layout="wide")

# Background color styling
page_bg_color = """
<style>
body {
    background-color: #e6f7ff;
}
header, footer {
    visibility: hidden;
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

# Database querying function
def query_data():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="treatment",
            user="postgres",
            password="lamis"
        )
        query = """
            SELECT 
            facility, hospital_number, age, gender, education_level, marital_status, art_duration, 
            changed_regimen, side_effects, adherence, missed_doses, base_line_viral_load, 
            current_viral_load, most_recent_viral_load, first_cd4, current_cd4, smoking, alcohol, 
            recreational_drugs, experience, clinic_appointments, barriers
            FROM treatment_data_new
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error querying the database: {e}")
        return pd.DataFrame()

# Preprocessing function for clustering
def preprocess_data_for_clustering(df):
    categorical_columns = ['facility', 'gender', 'education_level', 'marital_status', 'changed_regimen',
                           'side_effects', 'adherence', 'base_line_viral_load', 'current_viral_load',
                           'most_recent_viral_load', 'first_cd4', 'current_cd4', 'smoking', 'alcohol',
                           'recreational_drugs', 'experience', 'clinic_appointments', 'barriers', 'missed_doses']
    
    missed_doses_mapping = {'0': 0, '1-2': 1, '3-5': 2, '>5': 3}
    df['missed_doses'] = df['missed_doses'].map(missed_doses_mapping).fillna(-1)
    
    encoders = {}
    for column in categorical_columns:
        if column in df.columns:
            encoder = LabelEncoder()
            df[column] = df[column].astype(str)
            df[column] = encoder.fit_transform(df[column])
            encoders[column] = encoder
    
    features = df.drop(columns=['hospital_number'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return df, scaled_features, encoders, scaler

# Function to explain clusters
def explain_clusters(cluster_id):
    explanations = {
        0: "Cluster 0: Patients with high adherence, low viral load, and stable CD4 count.",
        1: "Cluster 1: Patients who may have fluctuating viral load and experience side effects.",
        2: "Cluster 2: Patients with poor adherence, high viral load, and potentially more barriers to treatment.",
        3: "Cluster 3: Patients with a high chance of interrupting treatment due to poor adherence and frequent missed doses."
    }
    return explanations.get(cluster_id, "No explanation available")

# Streamlit app with pages
st.sidebar.title("Navigation")
pages = ["Questionnaire", "Model Prediction"]
selected_page = st.sidebar.radio("Go to", pages)

if selected_page == "Questionnaire":
    st.title("Treatment Questionnaire")
    with st.form(key="questionnaire_form"):
        st.subheader("Patient Information")
        facility = st.selectbox("Select Facility", ['Akamkpa General Hospital'])
        hospital_number = st.text_input("Hospital Number")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.selectbox("Gender", ['MALE', 'FEMALE'])
        education_level = st.selectbox("Education Level", ['NO EDUCATION', 'PRIMARY', 'SECONDARY', 'HIGHER EDUCATION'])
        marital_status = st.selectbox("Marital Status", ['SINGLE', 'MARRIED', 'DIVORCE', 'WIDOWED'])
        art_duration = st.number_input("Duration on ART (years)", min_value=0, step=1)
        changed_regimen = st.selectbox("Changed ART Regimen?", ['NO', 'YES'])
        side_effects = st.selectbox("Experienced Side Effects?", ['NO', 'YES'])
        adherence = st.selectbox("Adherence to Medication?", ['RARELY', 'SOMETIMES', 'ALWAYS'])
        missed_doses = st.selectbox("Missed Doses in a Month", ['0', '1-2', '3-5', '>5'])
        base_line_viral_load = st.selectbox("Baseline Viral Load", ["I DON'T KNOW", '<1000', '>1000'])
        current_viral_load = st.selectbox("Current Viral Load", ["I DON'T KNOW", '<1000', '>1000'])
        most_recent_viral_load = st.selectbox("Most Recent Viral Load", ["I DON'T KNOW", '<1000', '>1000'])
        first_cd4 = st.selectbox("First CD4 Count", ["<200", ">200"])
        current_cd4 = st.selectbox("Current CD4 Count", ["<200", ">200"])
        smoking = st.selectbox("Do you smoke?", ['NO', 'YES'])
        alcohol = st.selectbox("Do you consume alcohol?", ['NO', 'YES'])
        recreational_drugs = st.selectbox("Do you use recreational drugs?", ['NO', 'YES'])
        experience = st.selectbox("Psychosocial Experience", ['Depression', 'Anxiety', 'Stress related to stigma or discrimination', 'None'])
        clinic_appointments = st.selectbox("Clinic Appointments", ['Regularly', 'Occasionally', 'Rarely'])
        barriers = st.selectbox("Barriers to Healthcare", ['NO', 'YES'])
        submit = st.form_submit_button("Submit")

        if submit:
            st.success("Questionnaire submitted successfully!")

elif selected_page == "Model Prediction":
    st.title("Model Prediction")
    st.subheader("Load and Predict Data")
    
    if st.button("Load and Predict"):
        data = query_data()
        if data.empty:
            st.warning("No data available!")
        else:
            data, scaled_features, _, _ = preprocess_data_for_clustering(data)
            kmeans = KMeans(n_clusters=4, random_state=42)
            kmeans.fit(scaled_features)
            data['Cluster'] = kmeans.predict(scaled_features)
            data['Cluster Explanation'] = data['Cluster'].apply(explain_clusters)
            
            st.write("Prediction Results")
            st.dataframe(data[['hospital_number', 'facility', 'Cluster', 'Cluster Explanation']])
            
            # Enable CSV download
            st.download_button(
                label="Download Results as CSV",
                data=data.to_csv(index=False),
                file_name="clustered_results.csv",
                mime="text/csv"
            )
