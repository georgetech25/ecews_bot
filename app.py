import streamlit as st
import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to query data from the database
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
            facility,	
            hospital_number,
            age,	
            gender,
            education_level,	
            marital_status,
            art_duration,	
            changed_regimen,	
            side_effects,	
            adherence,	
            missed_doses,	
            base_line_viral_load,	
            current_viral_load,	
            most_recent_viral_load,
            first_cd4,	
            current_cd4,	
            smoking,	
            alcohol,	
            recreational_drugs,	
            experience,	
            clinic_appointments,	
            barriers
            FROM treatment_data_new
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error querying the database: {e}")
        return pd.DataFrame()

# Function to explain clusters
def explain_clusters(cluster_id):
    explanations = {
        0: "Cluster 0: Likely patients with high adherence, low viral load, and stable CD4 count.",
        1: "Cluster 1: Patients who may have fluctuating viral load and experience side effects.",
        2: "Cluster 2: Patients with poor adherence, high viral load, and potentially more barriers to treatment."
    }
    return explanations.get(cluster_id, "No explanation available")

# Function to handle categorical data for clustering
def preprocess_data_for_clustering(df):
    categorical_columns = ['facility', 'gender', 'education_level', 'marital_status', 'changed_regimen', 
                           'side_effects', 'adherence', 'base_line_viral_load', 'current_viral_load', 
                           'most_recent_viral_load', 'first_cd4', 'current_cd4', 'smoking', 'alcohol', 
                           'recreational_drugs', 'experience', 'clinic_appointments', 'barriers', 'missed_doses']
    
    # Handling 'missed_doses' column with categories like '1-2'
    missed_doses_mapping = {'0': 0, '1-2': 1, '3-5': 2, '>5': 3}
    df['missed_doses'] = df['missed_doses'].map(missed_doses_mapping).fillna(-1)  # -1 for any missing values
    
    # Encoding categorical columns
    encoders = {}
    for column in categorical_columns:
        if column in df.columns:
            encoder = LabelEncoder()
            df[column] = df[column].astype(str)
            df[column] = encoder.fit_transform(df[column])
            encoders[column] = encoder
    
    # Standardizing the data
    features = df.drop(columns=['hospital_number'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return df, scaled_features, encoders, scaler

# Streamlit app with sidebar
st.title("HIV/AIDS Treatment Failure Clustering with KMeans")

# Sidebar for user input (data for individual submission)
with st.sidebar:
    st.header("Input Data for Individual Submission")
    facility = st.selectbox("Select Facility", ['Akamkpa General Hospital'], key="facility_input")
    hospital_number = st.text_input("Hospital Number", key="hospital_number_input")
    age = st.number_input("Age", min_value=0, max_value=120, step=1, key="age_input")
    gender = st.selectbox("Gender", ['MALE', 'FEMALE'], key="gender_input")
    education_level = st.selectbox("Education Level", ['NO EDUCATION', 'PRIMARY', 'SECONDARY', 'HIGHER EDUCATION'], key="education_input")
    marital_status = st.selectbox("Marital Status", ['SINGLE', 'MARRIED', 'DIVORCE', 'WIDOWED'], key="marital_input")
    art_duration = st.number_input("Duration on ART (years)", min_value=0, step=1, key="art_duration_input")
    changed_regimen = st.selectbox("Changed ART Regimen?", ['NO', 'YES'], key="changed_regimen_input")
    side_effects = st.selectbox("Experienced Side Effects?", ['NO', 'YES'], key="side_effects_input")
    adherence = st.selectbox("Adherence to Medication?", ['RARELY', 'SOMETIMES', 'ALWAYS'], key="adherence_input")
    missed_doses = st.selectbox("Missed Doses in a Month", ['0', '1-2', '3-5', '>5'], key="missed_doses_input")
    base_line_viral_load = st.selectbox("Baseline Viral Load", ["I DON'T KNOW", '<1000', '>1000'], key="baseline_viral_load_input")
    current_viral_load = st.selectbox("Current Viral Load", ["I DON'T KNOW", '<1000', '>1000'], key="current_viral_load_input")
    most_recent_viral_load = st.selectbox("Most Recent Viral Load", ["I DON'T KNOW", '<1000', '>1000'], key="recent_viral_load_input")
    first_cd4 = st.selectbox("First CD4 Count", ["<200", ">200"], key="first_cd4_input")
    current_cd4 = st.selectbox("Current CD4 Count", ["<200", ">200"], key="current_cd4_input")
    smoking = st.selectbox("Do you smoke?", ['NO', 'YES'], key="smoking_input")
    alcohol = st.selectbox("Do you consume alcohol?", ['NO', 'YES'], key="alcohol_input")
    recreational_drugs = st.selectbox("Do you use recreational drugs?", ['NO', 'YES'], key="recreational_drugs_input")
    experience = st.selectbox("Psychosocial Experience", ['Depression', 'Anxiety', 'Stress related to stigma or discrimination', 'None'], key="experience_input")
    clinic_appointments = st.selectbox("Clinic Appointments", ['Regularly', 'Occasionally', 'Rarely'], key="appointments_input")
    barriers = st.selectbox("Barriers to Healthcare", ['NO', 'YES'], key="barriers_input")
    
    # Submit button for individual data entry
    submit_button_individual = st.button("Submit Individual Data", key="submit_individual_button")

# Button to load data and make predictions
if st.button("Load and Cluster Data"):
    data = query_data()

    if data.empty:
        st.warning("No data available or failed to fetch data from the database.")
    else:
        st.success("Data loaded successfully!")

        # Data Preprocessing and Clustering
        data, scaled_features, encoders, scaler = preprocess_data_for_clustering(data)

        # Train the KMeans model
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(scaled_features)

        # Make predictions and add explanation
        data['Cluster'] = kmeans.predict(scaled_features)
        data['Cluster_Explanation'] = data['Cluster'].apply(explain_clusters)

        # Replace cluster number with the corresponding 'facility' name
        data['Cluster_Facility'] = data.apply(lambda row: row['facility'] if row['Cluster'] == 0 else
                                               ('Cluster 1' if row['Cluster'] == 1 else 'Cluster 2'), axis=1)

        # Display results with 'facility' added
        st.write("Clustering Results with Explanation:")
        st.dataframe(data[['hospital_number', 'facility', 'Cluster_Facility', 'Cluster_Explanation']])

        # Enable CSV download of the clustered data
        st.download_button(
            label="Download Clustered Data as CSV",
            data=data.to_csv(index=False),
            file_name="clustered_data.csv",
            mime="text/csv"
        )

# Submit individual data to the database (no prediction here)
if submit_button_individual:
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="treatment",
            user="postgres",
            password="lamis"
        )
        cursor = conn.cursor()

        # Insert the individual data into the database
        sql = """
        INSERT INTO treatment_data_new (
            facility, hospital_number, age, gender, education_level, marital_status, art_duration, changed_regimen, 
            side_effects, adherence, missed_doses, base_line_viral_load, current_viral_load, most_recent_viral_load, 
            first_cd4, current_cd4, smoking, alcohol, recreational_drugs, experience, clinic_appointments, barriers
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
        """
        
        # Collecting the data from the form inputs
        data = (
            facility, hospital_number, age, gender, education_level, marital_status, art_duration, 
            changed_regimen, side_effects, adherence, missed_doses, base_line_viral_load, current_viral_load, 
            most_recent_viral_load, first_cd4, current_cd4, smoking, alcohol, recreational_drugs, experience, 
            clinic_appointments, barriers
        )

        # Execute the SQL query
        cursor.execute(sql, data)
        conn.commit()
        st.success("Data submitted successfully to the database!")
    except Exception as e:
        st.error(f"Error submitting data: {e}")
    finally:
        cursor.close()
        conn.close()
