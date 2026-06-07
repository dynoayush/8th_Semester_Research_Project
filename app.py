import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv

# Set page configuration FIRST before any other st commands
st.set_page_config(page_title="Medi-Link", layout="wide", page_icon="⚕️")

# RAG imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; font-family: 'Inter', sans-serif; font-weight: 700;}
    h2, h3 {color: #34495e; font-family: 'Inter', sans-serif;}
    .stButton>button {
        background-color: #3498db; color: white; border-radius: 8px; font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {background-color: #2980b9;}
    .card {
        background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚕️ Medi-Link : Intelligent Healthcare Consultation System")

# Cache data loading
@st.cache_data
def load_csv_data():
    desc_df = pd.read_csv('Data_Required/description.csv')
    med_df = pd.read_csv('Data_Required/medications.csv')
    diet_df = pd.read_csv('Data_Required/diets.csv')
    prec_df = pd.read_csv('Data_Required/precautions.csv')
    workout_df = pd.read_csv('Data_Required/workout.csv')
    return desc_df, med_df, diet_df, prec_df, workout_df

desc_df, med_df, diet_df, prec_df, workout_df = load_csv_data()

# Cache RAG setup
@st.cache_resource
def setup_rag():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64}
        )
        vectorstore = FAISS.load_local("medquad_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20}
        )
        
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
        
        prompt = PromptTemplate.from_template("""You are a friendly and easy-to-understand health assistant.
Use ONLY the information provided in the context below to answer. Do NOT use outside knowledge.

IMPORTANT RULES:
- Always use the simple, everyday common name of the disease (e.g., say 'High Blood Sugar' instead of 'Type 2 Diabetes Mellitus', 'High Blood Pressure' instead of 'Hypertension', 'Weak Heart' instead of 'Congestive Heart Failure').
- Focus on what condition the person likely HAS RIGHT NOW based on their symptoms — NOT what they might develop or die from in the future.
- Write as if you are explaining to a friend with no medical background. Use short, simple sentences.
- Do NOT use scary or death-related language.
- Do NOT use medical abbreviations without first explaining them in plain words.

Answer strictly in this format:

**Most Likely Condition:**
<Write the simple, everyday common name. Example: 'High Blood Sugar (Diabetes)' or 'Chest Infection (Pneumonia)'>

**What This Means (In Simple Words):**
<In 2-3 easy sentences, explain what this condition is and how the person's symptoms match it. Use simple language that a school student can understand.>

**Go See a Doctor If:**
- <Warning sign 1>
- <Warning sign 2>
- <Warning sign 3>

**What You Can Do Next:**
Please visit the 'Medicine Recommender' section of this app for personalized medicine suggestions, diet tips, and safety precautions for your condition.

If the context does not have enough information to answer, say exactly:
"I'm not sure based on the available information. Please see a doctor for a proper check-up."

Context:
{context}

Question:
{question}

Answer:""")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        qa_chain = (
            {"context": lambda x: format_docs(retriever.invoke(x)), "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )
        return qa_chain
    except Exception as e:
        return None

# Load XGBoost setup
@st.cache_resource
def load_ml_model():
    if os.path.exists('xgb_model.pkl') and os.path.exists('label_encoder.pkl') and os.path.exists('symptoms_list.pkl'):
        model = joblib.load('xgb_model.pkl')
        le = joblib.load('label_encoder.pkl')
        symptoms = joblib.load('symptoms_list.pkl')
        return model, le, symptoms
    return None, None, None

tab1, tab2, tab3 = st.tabs(["RAG Chatbot", "ML Symptom Predictor", "Medicine Recommender"])

with tab1:
    st.markdown("### Describe your symptoms")
    user_query = st.text_area("How are you feeling today?", height=150, placeholder="E.g., I have a severe headache, fever, and nausea...")
    if st.button("Ask"):
        if not user_query:
            st.warning("Please enter your symptoms.")
        else:
            qa_chain = setup_rag()
            if qa_chain is None:
                st.error("RAG pipeline is not properly configured. Ensure the FAISS index and API keys are present.")
            else:
                with st.spinner("Analyzing symptoms..."):
                    try:
                        response = qa_chain.invoke(user_query)
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown(response)
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

with tab2:
    st.markdown("### Predict Disease based on exact Symptoms (XGBoost Model)")
    model, le, symptoms = load_ml_model()
    if model is None:
        st.warning("XGBoost model not found. Please run the model training script first.")
    else:
        selected_symptoms = st.multiselect("Select your symptoms", options=symptoms)
        if st.button("Predict Disease"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                input_data = [0] * len(symptoms)
                for sym in selected_symptoms:
                    input_data[symptoms.index(sym)] = 1
                
                prediction = model.predict([input_data])
                disease = le.inverse_transform(prediction)[0]
                st.success(f"### Predicted Condition: **{disease}**")
                st.info("💡 You can now search for this condition in the **Medicine Recommender** tab!")

with tab3:
    st.markdown("### 💊 Medicine, Diet, and Precautions Lookup")
    search_disease = st.text_input("Enter the Disease Name (e.g., from Chatbot or ML Predictor)")
    
    if st.button("Get Recommendations"):
        if not search_disease:
            st.warning("Please enter a disease name.")
        else:
            # Case-insensitive partial match
            search_term = search_disease.lower().strip()
            
            # Helper function to find match
            def get_data(df, col_name="Disease"):
                match = df[df[col_name].str.lower().str.contains(search_term, na=False)]
                return match
            
            desc_match = get_data(desc_df)
            med_match = get_data(med_df)
            diet_match = get_data(diet_df)
            prec_match = get_data(prec_df)
            workout_match = get_data(workout_df)
            
            if med_match.empty and diet_match.empty and prec_match.empty:
                st.error("⚠️ I don't have enough information about this condition. Please consult a doctor immediately.")
            else:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                if not desc_match.empty:
                    st.subheader("📝 Description")
                    st.write(desc_match.iloc[0]['Description'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💊 Medications")
                    if not med_match.empty:
                        meds = med_match.iloc[0]['Medication']
                        try:
                            # It's saved as string representation of list
                            import ast
                            meds_list = ast.literal_eval(meds)
                            for m in meds_list: st.write(f"- {m}")
                        except:
                            st.write(meds)
                    else:
                        st.write("No medications found.")
                        
                    st.subheader("🏋️ Workouts / Activities")
                    if not workout_match.empty:
                        try:
                            w_list = ast.literal_eval(workout_match.iloc[0]['Workouts'])
                            for w in w_list: st.write(f"- {w}")
                        except:
                            st.write(workout_match.iloc[0]['Workouts'])
                    else:
                        st.write("No specific workouts found.")
                
                with col2:
                    st.subheader("🥗 Recommended Diet")
                    if not diet_match.empty:
                        try:
                            diet_list = ast.literal_eval(diet_match.iloc[0]['Diet'])
                            for d in diet_list: st.write(f"- {d}")
                        except:
                            st.write(diet_match.iloc[0]['Diet'])
                    else:
                        st.write("No specific diet found.")
                        
                    st.subheader("🛡️ Precautions")
                    if not prec_match.empty:
                        p_row = prec_match.iloc[0]
                        # Precautions are in Precaution_1, Precaution_2, Precaution_3, Precaution_4
                        for i in range(1, 5):
                            col = f'Precaution_{i}'
                            if col in p_row and pd.notna(p_row[col]):
                                st.write(f"- {p_row[col]}")
                    else:
                        st.write("No specific precautions found.")
                        
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption("Disclaimer: This is an AI-powered system and should not replace professional medical advice.")
