import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from vector_db import get_vector_db
from llm_summarizer import get_llm_response
from data_loader import load_data
from ml_model import predict_risk

# Load dataset
df = load_data(
    "D:/Course/Brainworks/Python_Practice/Projects/Final_project/datasets/edu_mentor_dataset_final(5000).csv")

# Define mappings for categorical features
LEARNING_STYLE_MAPPING = {
    'Visual': 0,
    'Auditory': 1,
    'Kinesthetic': 2,
    'Read/Write': 3
}

CONTENT_TYPE_MAPPING = {
    'Text': 0,
    'Video': 1,
    'Interactive': 2,
    'Audio': 3
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'is_at_risk' not in st.session_state:
    st.session_state.is_at_risk = False
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

# Title
st.title("Student Academic Performance Dashboard")

# Input: Student ID and Name
student_id = st.text_input("Enter Student ID")
student_name = st.text_input("Enter Student Name")

if student_id and student_name:
    student_record = df[(df['student_id'].astype(str) == str(student_id)) &
                        (df['student_name'].str.lower() == student_name.lower())]

    if not student_record.empty:
        st.success("Student found. Please choose a performance category.")
        student = student_record.squeeze()

        # Calculate risk score once per session
        if st.session_state.risk_score is None:
            # Preprocess categorical features
            learning_style_num = LEARNING_STYLE_MAPPING.get(student['learning_style'], 0)
            content_type_num = CONTENT_TYPE_MAPPING.get(student['content_type_preference'], 0)

            # Convert teacher comments to sentiment score
            try:
                if pd.isna(student['teacher_comments_summary']) or student['teacher_comments_summary'].strip() == "":
                    comments_sentiment = 0.0
                else:
                    blob = TextBlob(str(student['teacher_comments_summary']))
                    comments_sentiment = blob.sentiment.polarity
            except:
                comments_sentiment = 0.0

            # Prepare input features with converted values
            input_features = [
                student['std'],
                student['math_grade'],
                student['english_grade'],
                student['science_grade'],
                student['history_grade'],
                student['overall_grade'],
                student['assignment_completion'],
                student['engagement_score'],
                student['math_lec_present'],
                student['science_lec_present'],
                student['history_lec_present'],
                student['english_lec_present'],
                student['attendance_ratio'],
                student['login_frequency_per_week'],
                student['average_session_duration_minutes'],
                learning_style_num,
                content_type_num,
                student['completed_lessons'],
                student['practice_tests_taken'],
                student['lms_test_scores'],
                comments_sentiment
            ]

            # Convert to numpy array and reshape
            input_array = np.array(input_features).reshape(1, -1)

            # Display input features for debugging
            st.subheader("Model Input Features")
            feature_names = [
                'std', 'math_grade', 'english_grade', 'science_grade', 'history_grade',
                'overall_grade', 'assignment_completion', 'engagement_score',
                'math_lec_present', 'science_lec_present', 'history_lec_present',
                'english_lec_present', 'attendance_ratio', 'login_frequency_per_week',
                'average_session_duration', 'learning_style', 'content_type_preference',
                'completed_lessons', 'practice_tests_taken', 'lms_test_scores',
                'comments_sentiment'
            ]
            st.write(pd.DataFrame([input_features], columns=feature_names))

            # Predict risk
            st.session_state.risk_score, st.session_state.is_at_risk = predict_risk(input_array)

            # Display prediction result
            st.subheader("Risk Prediction Result")
            st.write(f"Risk Score: {st.session_state.risk_score:.2f}")
            st.write(f"At Risk: {'Yes' if st.session_state.is_at_risk else 'No'}")

        # --- DASHBOARD SECTION ---
        # Dropdown to select performance category
        option = st.selectbox("Select performance category:",
                              ("Academic Performance (Grades)",
                               "Attendance & Participation",
                               "Learning Behavior & Engagement",
                               "Learning Preferences",
                               "Qualitative Feedback",
                               "All"))

        if option in ["Academic Performance (Grades)", "All"]:
            st.subheader("Academic Performance (Grades)")
            cols = ['math_grade', 'english_grade', 'science_grade', 'history_grade', 'overall_grade']
            st.write(student[cols])

            for subject in cols[:-1]:
                if student[subject] < 40:
                    st.warning(
                        f"{subject.replace('_', ' ').title()}: Score below 40. Please consult respective teacher.")
            if student['overall_grade'] < 60:
                st.warning("Overall grade is below 60. Consult all subject teachers and work harder.")
            elif 60 <= student['overall_grade'] <= 85:
                st.info("Performance is average. Keep pushing to improve further.")
            elif student['overall_grade'] > 85:
                st.success("Congratulations! Great academic performance.")

        if option in ["Attendance & Participation", "All"]:
            st.subheader("Attendance & Participation")
            cols = ['math_lec_present', 'science_lec_present', 'history_lec_present', 'english_lec_present',
                    'attendance_ratio', 'login_frequency_per_week']
            st.write(student[cols])

            for lec in cols[:-2]:
                if student[lec] < 10:
                    st.warning(f"{lec.replace('_', ' ').title()}: Less than 10 classes attended. Minimum 15 required.")
            if student['attendance_ratio'] < 0.5:
                st.warning("Attendance ratio below 0.5. Please consult with the class teacher.")
            if student['login_frequency_per_week'] < 4:
                st.warning("Login frequency less than 4 per week. Please consult with the class teacher.")

        if option in ["Learning Behavior & Engagement", "All"]:
            st.subheader("Learning Behavior & Engagement")
            cols = ['average_session_duration_minutes', 'engagement_score', 'completed_lessons',
                    'practice_tests_taken', 'lms_test_scores', 'assignment_completion']
            st.write(student[cols])

            if student['average_session_duration_minutes'] < 60:
                st.warning("Average session duration is low (< 60 minutes). Increase your study time.")
            if student['engagement_score'] < 0.8:
                st.warning("Engagement score is below 0.8. Stay focused and consistent.")
            if student['completed_lessons'] < 10:
                st.warning("Fewer than 10 lessons completed. Please complete more lessons.")
            if student['practice_tests_taken'] < 5:
                st.warning("Practice tests taken < 5. Practice more.")
            if student['lms_test_scores'] < 40:
                st.warning("LMS test scores are low (< 40). Focus on understanding topics.")
            if student['assignment_completion'] < 60:
                st.warning("Assignment completion rate is low (< 60%). Submit assignments on time.")

        if option in ["Learning Preferences", "All"]:
            st.subheader("Learning Preferences")
            cols = ['learning_style', 'content_type_preference']
            st.write(student[cols])

        if option in ["Qualitative Feedback", "All"]:
            st.subheader("Qualitative Feedback")
            st.write({
                "Risk Score": f"{st.session_state.risk_score:.2f}",
                "Is At Risk": "Yes" if st.session_state.is_at_risk else "No",
                "Teacher Comments": student['teacher_comments_summary']
            })

            if st.session_state.is_at_risk:
                st.warning(
                    "Student is at risk. Immediate intervention required. Focus on feedback and improve performance.")
            else:
                st.success("Student is performing well. Keep up the good work!")

        # --- CHATBOT SECTION ---
        st.divider()
        st.header("EduMentor AI Assistant")

        # Initialize vector DB only once per session
        if st.session_state.vector_db is None:
            with st.spinner("Initializing knowledge base..."):
                st.session_state.vector_db = get_vector_db(df)

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Ask about your performance"):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.spinner("Analyzing your performance..."):
                ai_response = get_llm_response(
                    prompt=prompt,
                    student_id=student_id,
                    vector_db=st.session_state.vector_db,
                    student_data=student,
                    risk_score=st.session_state.risk_score,
                    is_at_risk=st.session_state.is_at_risk
                )

            # Add AI response to history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

            with st.chat_message("assistant"):
                st.markdown(ai_response)
    else:
        st.error("Student not found. Please check your input or contact the staff.")
else:
    st.info("Please enter your Student ID and Name to access your dashboard")