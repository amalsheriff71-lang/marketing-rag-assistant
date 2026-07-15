import streamlit as st
import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="Marketing Intelligence Assistant (Free Tier)",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 Social Media Marketing Intelligence Assistant")
st.info("💡 This version uses the **Google Gemini API**, which offers a free tier for students and developers.")
st.success("✅ Demo mode: Using a representative sample of 10,000 posts for faster performance.")

st.markdown("""
This assistant uses **Retrieval-Augmented Generation (RAG)** to provide data-driven marketing insights 
based on social media engagement and sponsorship data.
""")

# --- Sidebar for Configuration & Sources ---
# --- Sidebar for Configuration & Sources ---

with st.sidebar:
    st.header("⚙️ Configuration")

    # Load API Key from Streamlit Cloud Secrets
    try:
        api_key = st.secrets["API_KEY"]
        st.success("🔑 API Key loaded from Streamlit Secrets")

    except Exception:
        api_key = None
        st.error(
            "❌ API Key not found.\n\n"
            "Add API_KEY in Streamlit Cloud → Settings → Secrets"
        )

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    st.header("🔍 Intelligence Sources")
    st.info(
        "When you ask a question, the AI retrieves the most relevant "
        "social media posts to ground its answer."
    )

    source_container = st.container()
# --- RAG Logic (Cached for Performance) ---
@st.cache_resource
def initialize_rag():
    # 1. Load Data — resolve relative to this script's own location so it
    # works the same locally and on Streamlit Community Cloud, regardless
    # of the working directory the app happens to be launched from.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, "social_media_data", "social_media_dataset.csv"),
        os.path.join(os.getcwd(), "social_media_data", "social_media_dataset.csv"),
    ]
    data_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if data_path is None:
        st.error(
            "Dataset not found. Please ensure the 'social_media_data' folder "
            "(containing social_media_dataset.csv) is committed to your GitHub repository "
            "alongside app.py."
        )
        return None, None

    df = pd.read_csv(data_path)

    # Use a representative sample for deployment performance
    MAX_ROWS = 10000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    df['content_description'] = df['content_description'].fillna('')
    df['comments_text'] = df['comments_text'].fillna('')

    # Robust boolean normalization: handles real bools, "True"/"False" strings,
    # and 0/1 values — the original .map({True:..., False:...}) silently
    # produced NaN whenever pandas didn't infer a native bool dtype.
    def normalize_sponsored(val):
        if isinstance(val, bool):
            return 'Sponsored' if val else 'Not Sponsored'
        if isinstance(val, (int, float)):
            return 'Sponsored' if val == 1 else 'Not Sponsored'
        return 'Sponsored' if str(val).strip().lower() in ('true', '1', 'yes') else 'Not Sponsored'

    df['is_sponsored'] = df['is_sponsored'].apply(normalize_sponsored)

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
    # No persist_directory: Streamlit Community Cloud's filesystem is
    # ephemeral per session, and st.cache_resource already keeps this
    # in memory for the lifetime of the app — writing to disk just risks
    # permission/duplicate-collection errors on redeploys.
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_function,
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
    elif vectorstore is None:
        st.error("The knowledge base could not be loaded. Fix the dataset path issue above and try again.")
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

            # FIX: "gemini-pro" was retired by Google and now returns a 404
            # ("models/gemini-pro is not found for API version v1beta").
            # gemini-2.5-flash is the current, fast, free-tier-friendly model.
            try:
                llm = ChatGoogleGenerativeAI(
                    model ="gemini-2.0-flash",
                    google_api_key=current_api_key,
                    temperature=0
                )

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
            except Exception as e:
                st.error(f"❌ Error generating insights: {e}")
                st.info(
                    "Double-check that your Google API key is valid and that the Gemini API "
                    "is enabled for it at https://aistudio.google.com/."
                )
    else:
        st.info("💡 Enter a question above to get started.")

# --- Footer ---
st.divider()
st.caption("Built for Social Media Marketing Intelligence Assignment | Senior Data Engineer Roadmap")
