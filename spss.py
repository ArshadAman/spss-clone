import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyCQZeCg0szNGG3mRW-k2kzSqGSCkArlYJI")
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_gemini_analysis(prompt, data=None):
    """Generate insights using Gemini AI."""
    try:
        if data:
            response = model.generate_content([prompt, str(data)])
        else:
            response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini Analysis Error: {e}")
        return "Analysis not available."

class SPSSClone:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.data = self.data.drop(columns=[col for col in self.data.columns if 'id' in col.lower()], errors='ignore')
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype('category')
    
    def summary(self):
        return self.data.describe(include='all')
    
    def correlation_matrix(self):
        return self.data.select_dtypes(include=['number']).corr()
    
    def visualize_distribution(self, column):
        fig, ax = plt.subplots()
        sns.histplot(self.data[column], kde=True, ax=ax)
        st.pyplot(fig)
        
        gemini_response = generate_gemini_analysis(
            f"Analyze the distribution of {column} and suggest actions based on the data trends.", 
            self.data[column].describe().to_dict()
        )
        st.write("### Gemini Insights:")
        st.write(gemini_response)
    
    def visualize_correlation(self):
        fig, ax = plt.subplots()
        sns.heatmap(self.data.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    def visualize_boxplots(self):
        fig, ax = plt.subplots()
        self.data.select_dtypes(include=['number']).boxplot(rot=90, ax=ax)
        st.pyplot(fig)
    
    def visualize_pairplot(self):
        st.write("Generating pairplot, this may take some time...")
        pairplot_fig = sns.pairplot(self.data.select_dtypes(include=['number']))
        st.pyplot(pairplot_fig)
    
    def chi_square_test(self, col1, col2):
        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
        chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
        return {"Chi-Square": chi2, "p-value": p, "Degrees of Freedom": dof}
    
    def linear_regression(self, x_col, y_col):
        X = self.data[[x_col]].dropna()
        y = self.data[y_col].loc[X.index]
        model = LinearRegression().fit(X, y)
        return {"Coefficient": model.coef_[0], "Intercept": model.intercept_}
    
    def random_forest_classification(self, target_col):
        X = self.data.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore').dropna()
        y = self.data[target_col].loc[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        feature_importance_fig, ax = plt.subplots()
        pd.Series(model.feature_importances_, index=X.columns).sort_values().plot(kind='barh', ax=ax)
        st.pyplot(feature_importance_fig)
        
        gemini_response = generate_gemini_analysis(
            "Suggest treatments or interventions based on the classification model results.",
            report
        )
        
        return {"Accuracy": accuracy, "Classification Report": report, "Gemini Insights": gemini_response}
    
# Streamlit UI
st.title("SPSS Clone in Python 🧠📊")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    spss = SPSSClone(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.write(spss.data.head())
    
    st.subheader("📊 Summary Statistics")
    st.write(spss.summary())
    
    st.subheader("📈 Correlation Matrix")
    st.write(spss.correlation_matrix())
    
    st.subheader("🎨 Visualized Correlation Matrix")
    spss.visualize_correlation()
    
    st.subheader("📦 Boxplots for Numeric Columns")
    spss.visualize_boxplots()
    
    st.subheader("📊 Pairplot of Numerical Variables")
    spss.visualize_pairplot()
    
    column = st.selectbox("Select a column for distribution analysis", spss.data.select_dtypes(include=[np.number]).columns)
    spss.visualize_distribution(column)
    
    st.subheader("🔬 Chi-Square Test")
    chi_col1 = st.selectbox("First Categorical Column", spss.data.select_dtypes(include=['category']).columns)
    chi_col2 = st.selectbox("Second Categorical Column", spss.data.select_dtypes(include=['category']).columns)
    st.write(spss.chi_square_test(chi_col1, chi_col2))
    
    st.subheader("📊 Linear Regression")
    x_col = st.selectbox("Select X variable", spss.data.select_dtypes(include=['number']).columns)
    y_col = st.selectbox("Select Y variable", spss.data.select_dtypes(include=['number']).columns)
    st.write(spss.linear_regression(x_col, y_col))
    
    st.subheader("🤖 Machine Learning - Random Forest Classification")
    target_col = st.selectbox("Select Target Column for Classification", spss.data.select_dtypes(include=['category']).columns)
    st.write(spss.random_forest_classification(target_col))
    
    st.subheader("📢 Overall Gemini Insights")
    overall_insights = generate_gemini_analysis("Provide an overall analysis of the dataset for efficiency, resource allocation, and planning based on the patterns and trends observed.", spss.data.describe(include='all').to_dict())
    st.write(overall_insights)