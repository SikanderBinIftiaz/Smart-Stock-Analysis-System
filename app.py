import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation Tools
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Smart Stock Analysis System",
    page_icon="📈",
    layout="wide"
)

# --- SYSTEM STYLING ---
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 16px; text-align: center; color: #6B7280; margin-bottom: 30px; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- DATA PREPARATION ENGINE ---
@st.cache_data
def get_historical_data():
    # Your project's base mock dataset closing prices
    raw_prices = [150.2, 152.1, 151.5, 153.8, 155.0, 154.2, 156.5, 159.1, 158.0, 162.4, 161.0, 163.5, 165.2, 164.0, 166.5]
    df = pd.DataFrame({'Close': raw_prices})
    
    # Feature engineering (Lag Returns)
    df['Return_Lag1'] = df['Close'].pct_change(1)
    df['Return_Lag2'] = df['Close'].shift(1).pct_change(1)
    
    # Label mapping (1 for UP, 0 for DOWN)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    # Extract features for immediate next prediction horizon
    latest_lag1 = (raw_prices[-1] - raw_prices[-2]) / raw_prices[-2]
    latest_lag2 = (raw_prices[-2] - raw_prices[-3]) / raw_prices[-3]
    next_features = np.array([[latest_lag1, latest_lag2]])
    
    return df, next_features, raw_prices[-1]

# Load structural dataset elements
df_clean, X_predict_next, last_close_price = get_historical_data()
X = df_clean[['Return_Lag1', 'Return_Lag2']].values
y = df_clean['Target'].values

# --- APPLICATION HEADER ---
st.markdown("<div class='main-title'>Smart Stock Analysis System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Department of Robotics & Artificial Intelligence — Final Project Interface</div>", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Model Control Center")

model_choice = st.sidebar.selectbox(
    "Choose Classifier Algorithm:",
    ["Decision Tree", "Logistic Regression", "Support Vector Classifier (SVC)", "Random Forest", "K-Neighbors Classifier"]
)

# Instantiate selected option mapping
models_factory = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Classifier (SVC)": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Neighbors Classifier": KNeighborsClassifier(n_neighbors=3)
}

selected_model_obj = models_factory[model_choice]

# --- MAIN INTERFACE SPLIT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Training Dataset View")
    st.write("Engineered input metrics extracted from historical closing sequences:")
    st.dataframe(df_clean, use_container_width=True)
    
    # Interactive Live Training Trigger Button
    train_button = st.button("🚀 Train & Save Selected Model", type="primary", use_container_width=True)

with col2:
    st.subheader("🧠 Model Evaluation & Real-Time Analytics")
    
    if train_button:
        # Fit algorithm to parameters safely
        selected_model_obj.fit(X, y)
        predictions = selected_model_obj.predict(X)
        
        # Calculate standard target indicators
        acc_score = accuracy_score(y, predictions)
        cv_scores = cross_val_score(selected_model_obj, X, y, cv=2)
        conf_mat = confusion_matrix(y, predictions)
        class_rep = classification_report(y, predictions, target_names=["DOWN", "UP"], zero_division=0, output_dict=True)
        
        # Display Metric Badges
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Training Accuracy Score", f"{acc_score * 100:.2f}%")
        m_col2.metric("Cross-Validation Score (CV:2)", f"{cv_scores.mean() * 100:.2f}%")
        
        # Display Matrix Information Breakdown
        st.markdown("#### Confusion Matrix Grid Layout")
        cm_df = pd.DataFrame(conf_mat, index=["Actual DOWN", "Actual UP"], columns=["Predicted DOWN", "Predicted UP"])
        st.dataframe(cm_df, use_container_width=True)
        
        # Classification report layout table tracking
        st.markdown("#### Detailed Precision & Recall Report")
        rep_df = pd.DataFrame(class_rep).transpose()
        st.dataframe(rep_df.style.format(precision=2), use_container_width=True)
        
        # --- PREDICTION RUN TIME ---
        next_prediction = selected_model_obj.predict(X_predict_next)[0]
        
        st.markdown("---")
        st.subheader("🔮 Predictive Direction Target Outcome")
        
        if next_prediction == 1:
            st.success(f"🟩 **BUY SIGNAL Generated** — The target trend class is predicted to go **UP (1)** from the last close of ${last_close_price:.2f}.")
        else:
            st.error(f"🟥 **SELL SIGNAL Generated** — The target trend class is predicted to go **DOWN (0)** from the last close of ${last_close_price:.2f}.")
            
        # --- PICKLE SAVING OPERATIONS ---
        pickle_filename = "Smart_Stock_Analysis_System.pkl"
        try:
            with open(pickle_filename, 'wb') as file:
                pickle.dump(selected_model_obj, file)
            st.toast(f"💾 Model pickled successfully as '{pickle_filename}'!", icon="✅")
        except Exception as e:
            st.error(f"Error executing binary pickle save parameters: {e}")
            
    else:
        st.info("💡 Adjust your system properties on the left sidebar panel, then click the **'Train & Save Selected Model'** button to evaluate outcomes.")

# --- PERSISTENT PICKLE FILE CHECKS ---
st.markdown("---")
st.subheader("📦 Model File Status Check")
if os.path.exists("Smart_Stock_Analysis_System.pkl"):
    st.write("🟢 A valid binary model instance named `Smart_Stock_Analysis_System.pkl` is currently initialized in your active working directory and ready for client systems interface use.")
else:
    st.write("⚪ No static saved file has been generated yet. Run the workspace execution pipeline training flow above to export your compiled data structures.")