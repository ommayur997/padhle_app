# train_scaler.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv(
    "D:/Course/Brainworks/Python_Practice/Projects/Final_project/datasets/edu_mentor_dataset_final(5000).csv")

# Define mappings
LEARNING_STYLE_MAPPING = {'Visual': 0, 'Auditory': 1, 'Kinesthetic': 2, 'Read/Write': 3}
CONTENT_TYPE_MAPPING = {'Text': 0, 'Video': 1, 'Interactive': 2, 'Audio': 3}

# Preprocess features
features = []
for _, row in df.iterrows():
    learning_style_num = LEARNING_STYLE_MAPPING.get(row['learning_style'], 0)
    content_type_num = CONTENT_TYPE_MAPPING.get(row['content_type_preference'], 0)

    # Create feature vector (same order as in app.py)
    feature_vector = [
        row['std'],
        row['math_grade'],
        row['english_grade'],
        row['science_grade'],
        row['history_grade'],
        row['overall_grade'],
        row['assignment_completion'],
        row['engagement_score'],
        row['math_lec_present'],
        row['science_lec_present'],
        row['history_lec_present'],
        row['english_lec_present'],
        row['attendance_ratio'],
        row['login_frequency_per_week'],
        row['average_session_duration_minutes'],
        learning_style_num,
        content_type_num,
        row['completed_lessons'],
        row['practice_tests_taken'],
        row['lms_test_scores'],
        0.0  # Placeholder for comments_sentiment
    ]
    features.append(feature_vector)

# Create and fit scaler
scaler = StandardScaler()
scaler.fit(np.array(features))

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler created and saved!")