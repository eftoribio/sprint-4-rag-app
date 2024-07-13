import streamlit as st
from dotenv import load_dotenv
import openai
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import openai
from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
from openai import OpenAI

# api key
api_key = st.secrets["api_key"]
from openai import OpenAI
client = OpenAI(api_key=api_key)
SKLLMConfig.set_openai_key(api_key)

# Load environment variables from .env file
load_dotenv()
openai.api_key = api_key # os.getenv('OPENAI_API_KEY')

def return_best_articles(user_input, collection, n_results=3):
    query_result = collection.query(query_texts=[user_input], n_results=n_results)
   
    if not query_result['ids'] or not query_result['ids'][0]:
        print("No articles found matching the query.")
        return None, None  # No results found

    results = []
    for i in range(min(n_results, len(query_result['ids'][0]))):
        result_id = query_result['ids'][0][i]
        result_metadata = query_result['metadatas'][0][i]
        result_document = query_result['documents'][0][i]
        
        print(f"\nArticle {i+1}:")
        print("---------------")
        print(f"Name: {result_metadata.get('article', 'Unknown Article')}")
        print("\nArticle Body:")
        print("-----------------")
        print(result_document)
        print("\nSteps:")
        print("----------------")
        
        results.append((result_metadata.get('article', 'Unknown Article'), result_document))
    
    return results

def generate_conversational_response(user_input, collection):
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    relevant_articles = return_best_articles(user_input, collection, n_results=3)
    
    if not relevant_articles:
        return "I couldn't find any relevant article based on your input."
    
    # Prepare the content for the assistant message
    articles_content = ""
    for i, (article_title, article_document) in enumerate(relevant_articles, 1):
        articles_content += f"\n\nArticle {i}: {article_title}\n{article_document}"

    messages = [
        {"role": "system", "content": "You are a skilled assistant for government services providing step-by-step instructions on how to apply for government services."},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": f"I found the following relevant articles: {articles_content}\n\nBased on these articles, here's how I can help you:"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500
    )
    
    return response.choices[0].message.content

### CHROMADB

# Constants
CHROMA_DATA_PATH = 'phgovinfo_data'
COLLECTION_NAME = "phgovinfo_embeddings"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-ada-002")

# Create or get the collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
)


### STREAMLIT
my_page = st.sidebar.radio('Page Navigation',
                           ['Chatbot', 'Explore'])

if my_page == 'Chatbot':
    st.title('📋 GetReqPH')
    
    WELCOME_MESSAGE = """
    Welcome to GetReqPH! I'm here to help you with information about Philippine government services.

    I can provide step-by-step instructions on how to apply for various government services. Just ask me about a specific service, and I'll guide you through the process.

    For example, you can ask:
    - How do I apply for an SSS ID?
    - What are the steps to get a passport?
    - How can I register my business with DTI?

    Right now, I'm only able to give information on one service at a time.
    Feel free to ask any questions about government services, and I'll do my best to assist you!
    """
    if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": WELCOME_MESSAGE}
            ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Say something")
    if prompt:    
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate and display assistant response
        response = generate_conversational_response(prompt, collection)
        with st.chat_message('assistant'):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

elif my_page == 'Explore':
    st.title('📚 Explore Articles')
    df = pd.read_csv("PHGovInfo_categorized.csv")

    # Define the options for agency and target_user
    agency_option = ['SSS',
        'GSIS',
        'Pag-IBIG',
        'Philhealth',
        'NBI Clearance',
        'PSA (Birth Certificate)',
        'BIR',
        'PRC',
        'DFA',
        'PWD',
        'Senior Citizen',
        'Voters ID',
        'DTI'
        ]
    target_user_option = ['New User or Fresh Application',
        'Renewal or Change job Requirements or Change name and details',
        'Business Owner',
        'Personal Growth or Personal Property Acquisition and Maintenance'
        ]

    # Use Streamlit's columns feature to arrange inputs side by side with custom widths
    col1, col2 = st.columns([1, 1])
    
    # # Radio buttons for selecting user type and agency
    # user = col1.radio('Select type of user', options=target_user_option)
    # agency = col2.radio('Select agency', options=agency_option)

    # Select boxes for selecting user type and agency
    user = col1.selectbox('Select type of user', options=target_user_option, index=None)
    agency = col2.selectbox('Select agency', options=agency_option, index=None)

    # Filter the DataFrame based on user selection
    if agency and user:  # If both agency and user are selected
        filtered_df = df[(df['agency'].apply(lambda x: agency in x)) & (df['target_user'].apply(lambda x: user in x))]
    elif agency:  # If only agency is selected
        filtered_df = df[(df['agency'].apply(lambda x: agency in x))]
    elif user:  # If only user is selected
        filtered_df = df[df['target_user'].apply(lambda x: user in x)]
    else:  # If neither agency nor user is selected
        filtered_df = df
    
    # Display the filtered results in a selectbox
    if not filtered_df.empty:
        selected_title = st.selectbox('Select an article', options=filtered_df['Title'].tolist(), index=None)

        if selected_title:
            article = filtered_df[filtered_df['Title'] == selected_title].iloc[0]
            
            st.header(f"[{article['Title']}]({article['Link']})")
            st.caption(f"__Published date:__ {article['Date Published']}")
                    
            col1, col2 = st.columns([3, 1])
        
            focused_summary_toggle = col1.toggle('Make focused summary', value=False)
            
            summary_button = col2.button('Summarize article')
            
            if focused_summary_toggle:
                focus = st.text_input('Input summary focus', value='')
                if focus == '':
                    focus = None
            else:
                focus = None
            
            if summary_button:
                st.subheader('Summary')
                s = GPTSummarizer(model='gpt-3.5-turbo', max_words=50, focus=focus)
                article_summary = s.fit_transform([article['content.cleaned']])[0]
                st.write(article_summary)
            
            st.subheader('Article content')
            st.write(article['content.cleaned'])

    else:
        st.write("No results found.")

