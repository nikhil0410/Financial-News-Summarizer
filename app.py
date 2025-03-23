import streamlit as st
import re
from GoogleNews import GoogleNews
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time, random, os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings and ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory="db/chroma_db", embedding_function=embeddings)

# Streamlit UI Components
st.title("Financial News Summarizer")

predefined_topics = ["Stocks", "Mutual Funds", "Tax-Saving Investments", "RBI Policies"]
selected_topic = st.selectbox("Choose a topic:", predefined_topics)
additional_text = st.text_input("Add additional search criteria:", "")

if st.button("Fetch and Summarize"):
    if selected_topic and additional_text:
        query_topic = f"{str(selected_topic)} - {str(additional_text)}"
        st.write(f"Performing search for: {query_topic}")

        # Query pre-existing embeddings
        query_embedding = embeddings.embed_query(query_topic)
        # print(type(query_embedding))  # Should be a numpy array or similar structure
        # print(query_embedding)
        # query_embedding = query_embedding.flatten()  
        similar_docs = db.similarity_search(query_topic, k=3)
        # similar_docs = 0

        if similar_docs:
            st.write("Found similar topics in the database. Using them for insights.")

            combined_input = (
                "Here are some documents that might help answer the question: {}".format(query_topic)
                + "\n\n".join([doc.page_content for doc in similar_docs])
                + "\n\nPlease summarize the content in no more than 10 lines."
            )

            llm = ChatOpenAI(model="gpt-4o")

            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=combined_input),
            ]

            result = llm.invoke(messages)

            # st.subheader("Generated Summary")
            st.write(result.content)



            # for doc in similar_docs:
            #     st.write(f"- {doc.page_content}")
        else:
            st.write("No similar topics found in the database. Performing a new search.")

            # Google news scraping
            googlenews = GoogleNews(period='7d')
            googlenews.search(query_topic)

            all_results = []
            for i in range(1, 50):
                googlenews.getpage(i)
                result = googlenews.result()
                
                if result:
                    all_results.extend(result)

                if len(all_results) >= 100:
                    break

            df = pd.DataFrame(all_results).drop_duplicates(subset=['title'], keep='last').head(100)
            df.reset_index(drop=True, inplace=True)

            data = df.drop(columns=['media', 'date', 'datetime', 'desc', 'img'])

            latest_links = [re.split("&ved", link)[0] for link in df['link']]

            description = []
            for i in latest_links:
                try:
                    response = requests.get(i, timeout=10)
                    if response.status_code == 200:
                        html_content = response.text
                    else:
                        description.append("Failed to retrieve the webpage.")
                        continue
                   
                    soup = BeautifulSoup(html_content, "html.parser")
                    paragraphs = soup.find_all("p")
                    page_description = " ".join([p.get_text() for p in paragraphs])
                    description.append(page_description)
                except requests.exceptions.RequestException:
                    description.append("Failed to retrieve the webpage.")
                    continue 
                time.sleep(random.uniform(1, 3))

            data["description"] = description

            folder_name = "NEWS_data"
            os.makedirs(folder_name, exist_ok=True)

            for index, row in data.iterrows():
                with open(os.path.join(folder_name, f"description_{index + 1}.txt"), "w", encoding="utf-8") as f:
                    f.write(row['description'])

            st.success("Fetched and stored articles. Generating summaries...")

            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)

            documents = []
            for filename in os.listdir("NEWS_data"):
                if filename.endswith(".txt"):
                    with open(os.path.join("NEWS_data", filename), 'r', encoding='utf-8') as file:
                        documents.append(file.read())

            all_chunks = []
            for doc in documents:
                chunks = text_splitter.split_text(doc)
                all_chunks.extend(chunks)

            ids = [f"doc_{i}" for i in range(len(all_chunks))]
            db.add_texts(all_chunks, ids=ids)

            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

            query = f"{query_topic}?"
            relevant_docs = retriever.invoke(query)

            combined_input = (
                "Here are some documents that might help answer the question: {}".format(query_topic)
                + "\n\n".join([doc.page_content for doc in relevant_docs])
                + "\n\nPlease summarize the content in no more than 4 lines."
            )

            llm = ChatOpenAI(model="gpt-4o")

            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=combined_input),
            ]

            result = llm.invoke(messages)

            st.subheader("Generated Summary")
            st.write(result.content)
    else:
        st.error("Please select a topic and add additional text.")