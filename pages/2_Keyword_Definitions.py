import streamlit as st
import pandas as pd
from openai import OpenAI
import random
import re
from annotated_text import annotated_text

api_key = st.secrets["api_key"]
client = OpenAI(api_key=api_key)

def extract_keywords_with_definitions(text):
    system_prompt = 'You are a financial news analyst assistant tasked to extract keywords related to finance from news articles and provide simple definitions for each keyword.'
    main_prompt = """
    ###TASK###
    - Extract the five most crucial finance-related keywords from the news article.
    - For each keyword, provide a simple definition as if explaining to a five-year-old, in the context of the article provided.
    - Return the results as a Python dictionary, where each key is a keyword and its value is the simple definition in the context of the article.
    - Example: {"stock": "A tiny piece of a company that you can buy", "ETF": "A basket of different stocks you can buy all at once", "bitcoin": "A special kind of computer money", "mutual funds": "A collection of investments that many people put money into together", "bond": "A way to lend money to a company or government and get paid back later with extra"}
    ###ARTICLE###
    """
    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{main_prompt}\n{text}"}
            ]
        )
        keywords_with_definitions = eval(response.choices[0].message.content)
        return keywords_with_definitions
    except:
        return "Unable to extract keywords, please try again later"
    
def generate_summary(text, keywords):
    system_prompt = 'You are a financial news analyst assistant tasked to summarize articles and link key concepts together.'
    main_prompt = f"""
    ###TASK###
    - Summarize the given article in about 50 words.
    - Use and link together the following keywords in your summary: {', '.join(keywords)}
    - The summary should be easy to understand for a general audience.
    ###ARTICLE###
    """
    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{main_prompt}\n{text}"}
            ]
        )
        summary = response.choices[0].message.content
        return summary
    except:
        return "Unable to generate summary, please try again later"

def prepare_annotated_text(text, keywords):
    # Sort keywords by length (longest first) to avoid partial matches
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    # Create a regular expression pattern for all keywords
    pattern = '|'.join(re.escape(kw) for kw in sorted_keywords)
    
    # Use regex to split the text, keeping separators
    tokens = re.split(f'({pattern})', text, flags=re.IGNORECASE)
    
    result = []
    for token in tokens:
        if token.strip():  # If token is not just whitespace
            if token.lower() in [kw.lower() for kw in sorted_keywords]:
                result.append((token, "", "#155830"))
            else:
                result.append(token)
        else:
            # Append whitespace as is
            result.append(token)
    
    return result

st.set_page_config(layout="wide")
st.title('Keyword Definitions')
st.sidebar.success("What do you want to learn today?")

df = pd.read_csv("data/combined_data.csv").sort_values(
    'date', ascending=False
)

# Add a button for random article selection
if st.button('Select Random Article'):
    random_row = df.sample(n=1).iloc[0]
    random_title = random_row['title']
    st.session_state['selected_title'] = random_title

# Use the selectbox, but initialize it with the random selection if it exists
if 'selected_title' in st.session_state:
    default_index = df.index[df['title'] == st.session_state['selected_title']].tolist()[0]
else:
    default_index = None

title = st.selectbox(
    'Select article title', 
    df['title'], 
    index=default_index
)

# title = st.selectbox(
#     'Select article title', df['title'], index=None
# )

if title:
    article = df[df['title']==title].iloc[0]
                        
    st.header(f"[{article['title']}]({article['link']})")
    st.caption(f"__Published date:__ {article['date']}")
    st.caption('**TOP KEYWORDS**')
    keywords_dict = extract_keywords_with_definitions(article['paragraph'])
    
    # Create a row of columns for the keywords
    cols = st.columns(len(keywords_dict))
    
    # Display each keyword in its own column with an expander
    for i, (keyword, definition) in enumerate(keywords_dict.items()):
        with cols[i]:
            with st.expander(keyword):
                st.caption('DEFINITION')
                st.write(definition)

    # Generate and display summary
    st.subheader('Article Summary')
    summary = generate_summary(article['paragraph'], keywords_dict.keys())
    annotated_summary = prepare_annotated_text(summary, keywords_dict.keys())
    annotated_text(*annotated_summary)

    st.subheader('Full article content')    
    st.write(article['paragraph'])