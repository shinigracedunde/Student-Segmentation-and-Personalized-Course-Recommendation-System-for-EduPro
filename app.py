
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="EduPro Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("EduPro - Student Segmentation and Course Recommendation System")
st.write("Personalized learning recommendations using K-Means clustering.")

@st.cache_data
def load_data():
    profiles = pd.read_csv("EduPro_Learner_Profiles.csv")
    recommendations = pd.read_csv("EduPro_Personalized_Recommendations.csv")
    courses = pd.read_csv("EduPro_Courses.csv")
    evaluation = pd.read_csv("EduPro_Model_Evaluation.csv")
    return profiles, recommendations, courses, evaluation

try:
    profiles, recommendations, courses, evaluation = load_data()

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Learners", len(profiles))
    col2.metric("Total Courses", len(courses))
    col3.metric("Total Clusters", profiles["Cluster"].nunique())

    st.subheader("Learner Distribution")

    cluster_counts = profiles["Cluster"].value_counts().sort_index()

    fig, ax = plt.subplots()
    cluster_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Learners")
    ax.set_title("Learners in Each Cluster")
    st.pyplot(fig)

    st.subheader("Learner Profiles")
    st.dataframe(profiles.head(20), use_container_width=True)

    st.subheader("Personalized Course Recommendations")

    user_id = st.selectbox(
        "Select User ID",
        profiles["UserID"].dropna().unique()
    )

    user_recommendations = recommendations[
        recommendations["UserID"] == user_id
    ]

    st.dataframe(user_recommendations, use_container_width=True)

    st.subheader("Model Evaluation")
    st.dataframe(evaluation, use_container_width=True)

except FileNotFoundError as e:
    st.error(f"Required project file not found: {e}")

except Exception as e:
    st.error(f"Dashboard error: {e}")
