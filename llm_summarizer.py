import os
from groq import Groq

def get_llm_response(prompt, student_id, vector_db, student_data, risk_score, is_at_risk):
    # Retrieve student context
    try:
        context = vector_db.get(ids=[str(student_id)], include=["documents"])["documents"][0]
    except:
        context = "No additional context available."

    # Initialize Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Prepare system prompt with risk information
    system_prompt = f"""
    [Student Risk Assessment]
    - Current Risk Score: {risk_score:.2f}
    - At Risk Status: {'Yes' if is_at_risk else 'No'}

    [Student Context]
    {context}

    You are an AI academic advisor. Provide:
    1. Specific analysis of the student's query
    2. Actionable recommendations
    3. References to their actual metrics
    4. Encouragement and support

    Guidelines:
    - Use markdown formatting for clear presentation
    - Include specific numbers from the student's data
    - Provide at least 3 actionable recommendations
    - Explain concepts in simple terms
    - Be supportive and encouraging
    """

    # Generate response
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.3,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content