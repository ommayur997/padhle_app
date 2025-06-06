import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

def get_vector_db(df):
    client = chromadb.PersistentClient(path="./vector_db")   #takes a pandas DataFrame as input and returns a ChromaDB collection
    #PersistentClient is a class which used to create a persistent vector database meaning the data is stored on disk, not just in memory
    #path parameter tells Chroma where to save the vector database files on your system
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    #embedding_functions converts text to vectors and Handles sentence similarity
    collection = client.get_or_create_collection(name="student_data", embedding_function=embedding_func)

    if collection.count() == 0:
        documents = []
        metadatas = []
        ids = []

        for _, row in df.iterrows():
            doc = f"""
            Student ID: {row['student_id']}
            Name: {row['student_name']}
            Math Grade: {row['math_grade']}
            English Grade: {row['english_grade']}
            Science Grade: {row['science_grade']}
            History Grade: {row['history_grade']}
            Overall Grade: {row['overall_grade']}
            Attendance Ratio: {row['attendance_ratio']}
            Engagement Score: {row['engagement_score']}
            Risk Score: {row['risk_score']}
            Teacher Comments: {row['teacher_comments_summary']}
            Learning Style: {row['learning_style']}
            Completed Lessons: {row['completed_lessons']}
            Practice Tests Taken: {row['practice_tests_taken']}
            """
            documents.append(doc)
            metadatas.append({"student_id": str(row['student_id'])})
            ids.append(str(row['student_id']))

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    return collection