import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Define mappings for categorical features
LEARNING_STYLE_MAPPING = {
    'Visual': 0, 'Auditory': 1, 'Kinesthetic': 2, 'Read/Write': 3
}

CONTENT_TYPE_MAPPING = {
    'Text': 0, 'Video': 1, 'Interactive': 2, 'Audio': 3
}


def preprocess_data(df):
    """Preprocess data exactly as in app.py"""
    features = []
    targets = []

    for _, row in df.iterrows():
        # Preprocess categorical features
        learning_style_num = LEARNING_STYLE_MAPPING.get(row['learning_style'], 0)
        content_type_num = CONTENT_TYPE_MAPPING.get(row['content_type_preference'], 0)

        # Convert teacher comments to sentiment score
        try:
            if pd.isna(row['teacher_comments_summary']) or row['teacher_comments_summary'].strip() == "":
                comments_sentiment = 0.0
            else:
                blob = TextBlob(str(row['teacher_comments_summary']))
                comments_sentiment = blob.sentiment.polarity
        except:
            comments_sentiment = 0.0

        # Prepare input features (same order as in app.py)
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
            comments_sentiment
        ]

        features.append(feature_vector)
        targets.append(row['risk_score'])

    return np.array(features), np.array(targets)


def main():
    # Load dataset
    df = pd.read_csv("edu_mentor_dataset_final(5000).csv")

    # Preprocess data
    X, y = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    print("\nModel Performance:")
    print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
    print(f"Test R²: {r2_score(y_test, test_pred):.4f}")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.2f}")

    # Save model and scaler
    joblib.dump(model, 'trained_model.pkl')
    joblib.dump(scaler, 'trained_scaler.pkl')
    print("\nModel and scaler saved:")
    print("- trained_model.pkl")
    print("- trained_scaler.pkl")

    # Feature importance
    print("\nTop 10 Features:")
    feature_names = [
        'std', 'math_grade', 'english_grade', 'science_grade', 'history_grade',
        'overall_grade', 'assignment_completion', 'engagement_score',
        'math_lec_present', 'science_lec_present', 'history_lec_present',
        'english_lec_present', 'attendance_ratio', 'login_frequency_per_week',
        'average_session_duration', 'learning_style', 'content_type_preference',
        'completed_lessons', 'practice_tests_taken', 'lms_test_scores',
        'comments_sentiment'
    ]

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    for i in sorted_idx[:10]:
        print(f"{feature_names[i]}: {importances[i]:.4f}")


if __name__ == "__main__":
    main()