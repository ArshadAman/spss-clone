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
import pyarrow as pa

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
        try:
            self.data = pd.read_csv(file)
            self.data = self.data.drop(columns=[col for col in self.data.columns if 'id' in col.lower()], errors='ignore')
            
            # Convert categorical/object columns to category type
            for col in self.data.select_dtypes(include=['object']).columns:
                self.data[col] = self.data[col].astype('category')
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.data = pd.DataFrame()  # Empty DataFrame in case of failure
    
    def summary(self):
        try:
            return self.data.describe(include='all')
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            return pd.DataFrame()
    
    def correlation_matrix(self):
        try:
            return self.data.select_dtypes(include=['number']).corr()
        except Exception as e:
            st.error(f"Error computing correlation matrix: {e}")
            return pd.DataFrame()
    
    def visualize_distribution(self, column):
        try:
            fig, ax = plt.subplots()
            sns.histplot(self.data[column], kde=True, ax=ax)
            st.pyplot(fig)

            gemini_response = generate_gemini_analysis(
                f"Analyze the distribution of {column} and suggest actions based on the data trends. Do not give any code and make sure it is easily understood to the user", 
                self.data[column].describe().to_dict()
            )
            st.write("### Data Insights:")
            st.write(gemini_response)
        except Exception as e:
            st.error(f"Error visualizing distribution: {e}")

    def visualize_correlation(self):
        try:
            fig, ax = plt.subplots()
            sns.heatmap(self.data.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error visualizing correlation: {e}")

    def visualize_boxplots(self):
        try:
            fig, ax = plt.subplots()
            self.data.select_dtypes(include=['number']).boxplot(rot=90, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error visualizing boxplots: {e}")

    def visualize_pairplot(self):
        try:
            st.write("Generating pairplot, this may take some time...")
            pairplot_fig = sns.pairplot(self.data.select_dtypes(include=['number']))
            st.pyplot(pairplot_fig)
        except Exception as e:
            st.error(f"Error generating pairplot: {e}")

    def chi_square_test(self, col1, col2):
        try:
            contingency_table = pd.crosstab(self.data[col1], self.data[col2])
            chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
            return {"Chi-Square": chi2, "p-value": p, "Degrees of Freedom": dof}
        except Exception as e:
            st.error(f"Error performing Chi-Square test: {e}")
            return {}

    def linear_regression(self, x_col, y_col):
        try:
            X = self.data[[x_col]].dropna()
            y = self.data[y_col].loc[X.index]
            model = LinearRegression().fit(X, y)
            return {"Coefficient": model.coef_[0], "Intercept": model.intercept_}
        except Exception as e:
            st.error(f"Error performing linear regression: {e}")
            return {}

    def random_forest_classification(self, target_col):
        try:
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

            return {"Accuracy": accuracy, "Classification Report": report, "Data Insights": gemini_response}
        except Exception as e:
            st.error(f"Error in Random Forest Classification: {e}")
            return {}

# Streamlit UI
st.title("Data Dig 🧠📊")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    spss = SPSSClone(uploaded_file)

    st.subheader("📋 Dataset Preview")
    st.write(spss.data.head() if not spss.data.empty else "Error loading dataset.")

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

    column = st.selectbox("Select a column for distribution analysis", spss.data.select_dtypes(include=[np.number]).columns, index=0)
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
    
    st.subheader("🧠 AI Data Insights")

    gemini_prompt = """
    Analyze the given dataset and provide key insights. Identify patterns, trends, and correlations. 
    For example, if weight increases, does BP also increase? Do any categorical variables have strong associations? 
    Summarize relationships between numeric columns and suggest meaningful insights.
    """

    try:
        dataset_summary = spss.data.describe(include='all').to_dict()
        gemini_overview = generate_gemini_analysis(gemini_prompt, dataset_summary)
        st.write(gemini_overview)
    except Exception as e:
        st.error(f"Error generating AI insights: {e}")