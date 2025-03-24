# Financial News Summarizer

## Overview

The Financial News Summarizer is a Streamlit-based application designed to fetch and summarize financial news articles. Users can select specific topics and additional criteria to generate concise summaries of relevant news.

## Key Features

- **Topic Selection:** Choose predefined financial topics like Stocks, Mutual Funds, Tax-Saving Investments, and RBI Policies.
- **Custom Search:** Input additional search criteria for tailored results.
- **Embeddings and Storage:** Uses OpenAI embeddings and ChromaDB for storing and retrieving document similarities.
- **Threshold-Based Retrieval:** Only fetches new data if existing embeddings don't meet a similarity threshold.
- **Real-time Summary Generation:** Utilizes OpenAI's GPT for generating concise summaries of fetched news articles.

## Components

- **Streamlit:** Provides a user-friendly interface for input and output.
- **GoogleNews API:** Fetches recent news articles based on user inputs.
- **BeautifulSoup:** Parses HTML content to extract article text.
- **OpenAIEmbeddings:** Converts text into vector embeddings for similarity search.
- **ChromaDB:** Stores and retrieves document embeddings efficiently.
- **ChatOpenAI:** Generates human-like summaries using GPT models.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/financial-news-summarizer.git
   cd financial-news-summarizer
2. **Install Dependencies:**

```pip install -r requirements.txt```

3. **Set Up Environment Variables:**

- Create a .env file in the project directory.
- Add your OpenAI API key and any other necessary environment variables.

4. **Run the Application:**

- streamlit run app.py

## Usage

- **Select a Topic:** Use the dropdown menu to choose a financial topic.
- **Add Criteria:** Input any additional search terms.
- **Fetch and Summarize:** Click the button to retrieve and summarize news articles.

## Configuration
- **Similarity Threshold:** Adjust the SIMILARITY_THRESHOLD variable to control when new data is fetched based on document similarity scores.

## Technical Architecture

1. **User Input:** Streamlit UI gathers topic and criteria.
2. **Data Retrieval:** GoogleNews API fetches articles, and BeautifulSoup extracts contents.
3. **Embedding and Storage:** Text is converted to embeddings using OpenAI and stored in ChromaDB.
4. **Summary Generation:** If relevant documents are found above the threshold, summaries are generated using GPT; otherwise, a new search occurs.
