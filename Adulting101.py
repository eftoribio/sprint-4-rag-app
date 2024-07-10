import streamlit as st
import pandas as pd
from openai import OpenAI
import random

# Load the CSV file

def return_best_article(user_input, collection, n_results=1):
    query_result = collection.query(query_texts=[user_input], n_results=n_results)
    
    if not query_result['ids'] or not query_result['ids'][0]:
        print("No article found matching the query.")
        return None, None  # No results found

    # Get the top result
    top_result_id = query_result['ids'][0][0]
    top_result_metadata = query_result['metadatas'][0][0]
    top_result_document = query_result['documents'][0][0]
    
    # Print query results in a readable and formatted manner
    print("Top Article Found:")
    print("---------------")
    print(f"Name: {top_result_metadata.get('article', 'Unknown Article')}")
    print("\Article Body:")
    print("-----------------")
    print(top_result_document)
    print("\nSteps:")
    print("----------------")
    
    return top_result_metadata.get('article', 'Unknown Article'), top_result_document

def generate_conversational_response(user_input, collection):
    # Initialize OpenAI client
     
    relevant_article_title, relevant_article_document = return_best_article(user_input, collection)
    
    if not relevant_article_title:
        return "I couldn't find any relevant article based on your input."

    messages = [
        {"role": "system", "content": "You are a skilled assistant for government services providing step-by-step instructions on how to apply for government services."},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": f"Relevant article found: {relevant_article_title}. Here are the steps you can follow: {relevant_article_document}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Streamlit app
st.set_page_config(layout="wide")
st.title("Finance Term of the Day ")

st.sidebar.success("What do you want to learn today?")
# Check if term and definition are already in session state
if 'term' not in st.session_state or 'definition' not in st.session_state:
    term, definition = get_random_term(data)
    st.session_state['term'] = term
    st.session_state['definition'] = definition
else:
    term = st.session_state['term']
    definition = st.session_state['definition']

st.subheader(f"Term: {term}")
st.write(f"Definition: {definition}")

st.subheader("Related Link")
links = get_related_links(term)
for link in links:
    st.write(link)

# st.write("Refresh the page for a new term!")

# Question related to the term of the day
st.subheader("Question about the Term")
question = f"What does the term '{term}' mean to you? Please explain in your own words."

st.write(question)
user_answer = st.text_area("Your Answer", "")

if st.button("Submit Answer"):
    feedback = get_feedback_and_explanation(term, user_answer)
    st.subheader("Feedback and Explanation")
    st.write(feedback)

