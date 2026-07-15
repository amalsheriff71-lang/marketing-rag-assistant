import streamlit as st
import pandas as pd
import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.api_core.exceptions import GoogleAPICallError

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
    api_key = st.text_input(
        "Enter Google API Key:",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
    )

    st.divider()
    st.header("🔍 Intelligence Sources")
    st.info("When you ask a question, the AI retrieves the most relevant social media posts to ground its answer.")
    source_container = st.container()

# --- RAG Logic (Cached for Performance) ---
# IMPORTANT: the api_key is now passed in as an argument so that:
#  1. st.cache_resource keys the cache on the api_key (rebuilds if it changes)
#  2. this function is never executed before the user actually supplies a key
@st.cache_resource(show_spinner="🧠 Building knowledge base (first run only)...")
def initialize_rag(_api_key: str):
    # 1. Load Data
    data_path = 'social_media_data/social_media_dataset.csv'
    if not os.path.exists(data_path):
        data_path = '/home/ubuntu/social_media_data/social_media_dataset.csv'

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please ensure the 'social_media_data' folder is in your GitHub repository.")
        return None

    df = pd.read_csv(data_path)
    df['content_description'] = df['content_description'].fillna('')
    df['comments_text'] = df['comments_text'].fillna('')
    df['is_sponsored'] = df['is_sponsored'].map({True: 'Sponsored', False: 'Not Sponsored'}).fillna('Unknown')

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
            "likes": int(row['likes']) if pd.notna(row['likes']) else 0,
        }
        documents.append(Document(page_content=content, metadata=metadata))

    # 3. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 4. Embeddings (Gemini) & Vector Store
    # Uses Google's own embedding model instead of sentence-transformers, so we
    # don't need to ship/download a multi-hundred-MB torch model on every deploy.
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=_api_key,
    )

    # Streamlit Community Cloud's filesystem is read-only outside of /tmp, so
    # persisting to "./streamlit_vector_db" can throw a permissions error there.
    # Use a temp directory instead, and don't try to persist across restarts.
    persist_dir = os.path.join(tempfile.gettempdir(), "streamlit_vector_db")

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=persist_dir,
        )
    except GoogleAPICallError as e:
        st.error(f"❌ Google API rejected the embedding request: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to build the vector store: {e}")
        return None

    return vectorstore

# --- Main Interface ---
query = st.text_input(
    "Ask a marketing question:",
    placeholder="e.g., Which content themes drive the most engagement in the beauty category?",
    help="The assistant will search the social media database to find the answer."
)

if st.button("Generate Insights"):
    if not api_key:
        st.warning("⚠️ Please enter your Google API Key in the sidebar first.")
    elif not query:
        st.info("💡 Enter a question above to get started.")
    else:
        with st.spinner("🧠 Analyzing social media data..."):
            vectorstore = initialize_rag(api_key)

            if vectorstore is None:
                st.stop()

            try:
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

                # gemini-pro was retired; gemini-2.5-flash is the current fast,
                # free-tier-friendly chat model.
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0,
                )

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                chain = (
                    {
                        "context": lambda x: format_docs(retrieved_docs),
                        "question": lambda x: x,
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                response = chain.invoke(query)

                # 4. Display Result
                st.success("✅ Analysis Complete")
                st.markdown("### 📊 Marketing Insights")
                st.write(response)

            except GoogleAPICallError as e:
                st.error(f"❌ Google API error: {e}. Double-check your API key and that the Gemini API is enabled for it.")
            except Exception as e:
                st.error(f"❌ Something went wrong while generating insights: {e}")

# --- Footer ---
st.divider()
st.caption("Built for Social Media Marketing Intelligence Assignment | Senior Data Engineer Roadmap")
