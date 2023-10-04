import streamlit as st
from bs4 import BeautifulSoup
import requests
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Streamlit app layout
st.title("Web News Summarizer")
supplier_link = st.text_input("Enter the supplier's web link:")

if st.button("Fetch and Summarize"):
    # Function to fetch and summarize the article
    def fetch_and_summarize(link):
        try:
            # Fetch the web page content
            response = requests.get(link)
            if response.status_code != 200:
                st.error("Failed to retrieve the web page.")
                return

            # Parse the web page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract article text (modify as per the website's structure)
            article_text = ""
            for paragraph in soup.find_all('p'):
                article_text += paragraph.text + "\n"

            # Summarize the article using the T5 model
            input_text = "summarize: " + article_text
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4,
                                         early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Display the summary
            st.subheader("Summary:")
            st.write(summary)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


    # Call the fetch_and_summarize function with the user-provided link
    if supplier_link:
        fetch_and_summarize(supplier_link)
    else:
        st.warning("Please enter a valid web link.")
