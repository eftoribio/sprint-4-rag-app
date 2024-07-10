import streamlit as st
from openai import OpenAI
import random

api_key = st.secrets["api_key"]
client = OpenAI(api_key=api_key)

# Function to generate response from OpenAI
def generate_response(prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system',
             'content':
             "You are a go-to finance bot with responses oriented to providing information for users curious about finance. Answer the question as if you are talking to someone who has zero knowledge on finance. Make the explanations as simple and as concise as possible. Make it short and straightforward."},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message.content

# FAQ list
faq_list = [
    "Understanding basic financial terms",
    "Explaining different types of investments",
    "Budgeting tips and strategies",
    "Credit score and how to improve it",
    "Saving money and setting financial goals",
    "Retirement planning basics",
    "Managing debt and loans",
    "General tips for financial well-being"
]

# Streamlit app layout
st.set_page_config(layout="wide")
st.title("Finance Bot")
st.sidebar.success("What do you want to learn today?")

st.write("How can I help you? Ask finance questions!")

# Dropdown for FAQ selection
selected_faq = st.selectbox("Select a topic or ask your own question:", [""] + faq_list)

# User input
if selected_faq:
    user_question = st.text_input("Your finance question:", value=selected_faq)
else:
    user_question = st.text_input("Enter your finance question:")
# Generate and display response
if user_question:
    response = generate_response(user_question)
    st.write("## Response:")
    st.write(response)

    