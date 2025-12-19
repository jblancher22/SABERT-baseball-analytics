import streamlit as st
import os
import warnings
from rag_core import BaseballRAGSystem

# Suppress warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------------------
# SETUP & CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SABERT: Baseball Analytics Chatbot",
    page_icon="⚾",
    layout="wide"
)


# ------------------------------------------------------------------------------
# CACHE THE RAG SYSTEM
# Run this once. We no longer pass docs_folder because the corpus is pre-built.
# ------------------------------------------------------------------------------
@st.cache_resource
def load_rag_system():
    """
    Initialize the RAG system.
    API key resolution is handled internally by BaseballRAGSystem:
      1. st.secrets["GROQ_API_KEY"]
      2. os.environ["GROQ_API_KEY"]
    """
    return BaseballRAGSystem(
        chroma_dir=".chroma_baseball"
    )


try:
    rag = load_rag_system()
except Exception as e:
    st.error(f"Failed to load RAG system. Did you run corpus_builder.py? Error: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------------------
st.markdown("""
<style>
.stChatMessage {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
/* Force user messages to align right/blue is tricky in pure Streamlit, 
   so we stick to default alignment but use custom colors if possible */
[data-testid="stChatMessageContent"] {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚾ Baseball Analytics")
    st.markdown("---")

    # MODE SELECTION
    st.subheader("Reasoning Mode")
    mode = st.radio(
        "Choose Strategy",
        [
            "Direct RAG",
            "Chain-of-Thought Refinement",
        ],
        index=0
    )

    st.markdown("---")


    # SAMPLE QUESTIONS
    # We use a callback to set the input immediately
    def set_question(question):
        st.session_state["clicked_question"] = question


    st.subheader("Sample Questions")
    sample_questions = [
        "Does BABIP indicate luck?",
        "What is the difference between FIP and ERA?",
        "How is WAR calculated?",
        "Explain defensive shifts",
        "What are key pitching metrics?",
        "How do aging curves influence projections?"
    ]

    for q in sample_questions:
        if st.button(q):
            set_question(q)

    st.markdown("---")

    # DOCUMENT SOURCES (Limit to first 10 to avoid sidebar clutter)
    st.subheader("Indexed Sources")
    try:
        # We access the internal collection just to peek at filenames
        # Note: listing ALL chunks is too much, so we get unique sources
        metas = rag.corpus_collection.get(limit=100)["metadatas"]
        unique_sources = list(set([m["source"].split("#")[0] for m in metas]))

        for src in unique_sources[:15]:  # Show first 15 unique files
            st.caption(f"📄 {src}")
        if len(unique_sources) > 15:
            st.caption(f"...and {len(unique_sources) - 15} more.")

    except Exception as e:
        st.caption("Could not load sources.")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------------------
# MAIN CHAT LOGIC
# ------------------------------------------------------------------------------
st.title("⚾ Baseball Analytics Chatbot")
st.caption("Powered by Groq LLaMA 3.3, BERT embeddings, and ChromaDB")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handling Input: Either from Chat Input OR Sidebar Button
prompt = st.chat_input("Ask a question...")

# If a sidebar button was clicked, override the prompt
if "clicked_question" in st.session_state and st.session_state["clicked_question"]:
    prompt = st.session_state["clicked_question"]
    # Clear it so it doesn't persist
    del st.session_state["clicked_question"]

# Process the prompt
if prompt:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Response
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking... ({mode})"):

            # Route based on sidebar selection
            if mode == "Direct RAG":
                result = rag.answer_direct(prompt)
            elif mode == "Chain-of-Thought Refinement":
                result = rag.cot_refine(prompt)

            response_text = result["answer"]
            sources_used = result["sources"]
            expert_used = mode

            # Display main answer
            st.markdown(response_text)

            # Display metadata in an expander
            with st.expander("📊 Analysis Details"):
                st.write(f"**Mode Used:** {result['mode']}")

                st.write("**Sources Referenced:**")
                if sources_used:
                    for s in sources_used:
                        st.write(f"- `{s}`")
                else:
                    st.write("(No specific corpus documents cited)")

    # 3. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": response_text})