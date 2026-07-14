import streamlit as st
import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="Marketing Intelligence Assistant (Free Tier)",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 Social Media Marketing Intelligence Assistant")
st.info("💡 This version uses the **Google Gemini API**, which offers a free tier for students and developers.")

st.markdown("""
This assistant uses **Retrieval-Augmented Generation (RAG)** to provide data-driven marketing insights 
based on social media engagement and sponsorship data.
""")

# --- Sidebar for Configuration & Sources ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("[Get a Free Google API Key](https://aistudio.google.com/)")
    api_key = st.text_input("Enter Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    st.header("🔍 Intelligence Sources")
    st.info("When you ask a question, the AI retrieves the most relevant social media posts to ground its answer.")
    source_container = st.container()

# --- RAG Logic (Cached for Performance) ---
@st.cache_resource
def initialize_rag():
    # 1. Load Data
    data_path = 'social_media_data/social_media_dataset.csv'
    if not os.path.exists(data_path):
        data_path = '/home/ubuntu/social_media_data/social_media_dataset.csv'
        
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please ensure the 'social_media_data' folder is in your GitHub repository.")
        return None, None
    
    df = pd.read_csv(data_path)
    df['content_description'] = df['content_description'].fillna('')
    df['comments_text'] = df['comments_text'].fillna('')
    df['is_sponsored'] = df['is_sponsored'].map({True: 'Sponsored', False: 'Not Sponsored'})
    
    # 2. Create Documents
    documents = []
    for _, row in df.iterrows():
        content = f"Platform: {row['platform']}\n"
        content += f"Category: {row['content_category']}\n"
        content += f"Description: {row['content_description']}\n"
        content += f"Engagement: {row['likes']} Likes, {row['comments_count']} Comments, {row['shares']} Shares\n"
        content += f"Status: {row['is_sponsored']}\n"
        content += f"Followers: {row['follower_count']}\n"
        content += f"Top Comment: {row['comments_text']}"
        
        metadata = {
            "platform": row['platform'],
            "category": row['content_category'],
            "likes": row['likes']
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # 3. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # 4. Embeddings & Vector Store
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=st.secrets["GEMINI_API_KEY"])
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory="./streamlit_vector_db"
    )
    
    return vectorstore, embedding_function

# --- Initialize ---
vectorstore, embedding_function = initialize_rag()

# --- Main Interface ---
query = st.text_input(
    "Ask a marketing question:",
    placeholder="e.g., Which content themes drive the most engagement in the beauty category?",
    help="The assistant will search the social media database to find the answer."
)

if st.button("Generate Insights"):
    current_api_key = api_key if api_key else os.environ.get("GOOGLE_API_KEY")
    
    if not current_api_key:
        st.warning("⚠️ Please enter your Google API Key in the sidebar or set it in Streamlit Secrets.")
    elif query:
        with st.spinner("🧠 Analyzing social media data..."):
            # 1. Retrieve
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(query)
            
            # 2. Update Sidebar with Sources
            with source_container:
                st.subheader("📍 Retrieved Posts")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Source {i+1}: {doc.metadata.get('platform')} ({doc.metadata.get('category')})"):
                        st.write(doc.page_content)
            
            # 3. Generate
            template = """
            You are a Senior Marketing Analyst. Use the following social media data to answer the user's question.
            Provide data-driven insights and specific recommendations.
            
            Context:
            {context}
            
            Question: {question}
            
            Marketing Intelligence Answer:
            """
            prompt = ChatPromptTemplate.from_template(template)
            
            # Use Gemini Pro (Free Tier available via Google AI Studio)
           llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GEMINI_API_KEY"])
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            chain = (
                {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(query)
            
            # 4. Display Result
            st.success("✅ Analysis Complete")
            st.markdown("### 📊 Marketing Insights")
            st.write(response)
    else:
        st.info("💡 Enter a question above to get started.")

# --- Footer ---
st.divider()
st.caption("Built for Social Media Marketing Intelligence Assignment | Senior Data Engineer Roadmap")
