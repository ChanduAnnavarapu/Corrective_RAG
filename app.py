import streamlit as st
import os
from chromadb import PersistentClient

from Corrective_RAG_Architecture import workflow
from Rag_flow import create_vector_store, get_all_pdfs

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Corrective RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Remove top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
}

/* Reduce sidebar top spacing */
section[data-testid="stSidebar"] {
    padding-top: 0rem;
}

/* Remove extra space above main content */
[data-testid="stAppViewContainer"] {
    padding-top: 0rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "Chat 1": []
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

# ---------------------------------------------------
# SIDEBAR (CHAT HISTORY)
# ---------------------------------------------------

with st.sidebar:
    
    st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 220px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 220px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.title("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):

        chat_num = len(st.session_state.chat_sessions) + 1

        chat_name = f"Chat {chat_num}"

        st.session_state.chat_sessions[chat_name] = []

        st.session_state.current_chat = chat_name

    st.divider()

    for chat_name in st.session_state.chat_sessions:

        if st.button(
            chat_name,
            use_container_width=True
        ):
            st.session_state.current_chat = chat_name

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------

center, right = st.columns([4, 1.5])

# ===================================================
# CENTER PANEL
# ===================================================

with center:

    st.markdown(
        """
        <div style='text-align:left'>
            <h1>🤖 Corrective RAG</h1>
            <p style='font-size:18px;color:gray'>
                Adaptive RAG pipeline that validates retrieved documents, filters irrelevant context, and automatically falls back to web search when local knowledge is insufficient.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    inner_left, inner_center, inner_right = st.columns([0.01, 5, 0.2])

    with inner_center:

        question = st.text_input(
            "",
            placeholder="Ask a question..."
        )

        submit = st.button(
            "Submit",
            use_container_width=True
        )

    # ------------------------------------------------

    if submit and question:

        initial_state = {
            "Question": question,
            "Docs": [],
            "refined_context": [],
            "web_docs": [],
            "answer": ""
        }

        with st.spinner("Generating response..."):

            result = workflow.invoke(initial_state)

        answer = result.get("answer", "")

        verdict = result.get(
            "verdict",
            "UNKNOWN"
        )
        
        try:
            with inner_center:
                #st.success("Response Generated")

                st.markdown(
                    f"### Verdict: {"Retrieved from RAG" if verdict=="CORRECT" else 
                    "Answered with websearch" if verdict=="INCORRECT" else "Retrieved documents are not sufficient to answer the question web search used to answer fully"}"
                )

                st.write(answer)

                st.session_state.chat_sessions[
                    st.session_state.current_chat
                ].append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

            # ----------------------------
            # Retrieved Docs
            # ----------------------------

                with st.expander(
                    "📄 Retrieved Documents"
                ):

                    docs = result.get(
                        "Docs",
                        []
                    )

                    if docs:

                        for idx, doc in enumerate(
                            docs,
                            start=1
                        ):

                            st.markdown(
                                f"### Document {idx}"
                            )

                            st.write(doc)

                    else:
                        st.info(
                            "No retrieved documents."
                        )

                # ----------------------------
                # Workflow State
                # ----------------------------

                with st.expander(
                    "⚙ Workflow State"
                ):

                    st.json(result)
                           
        except Exception as e:
            with inner_center:
                st.error(e)

# ===================================================
# RIGHT PANEL
# ===================================================

with right:

    st.markdown(
        "#### 📚 Knowledge Base"
    )

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    # ------------------------------------

    if st.button(
        "Create Embeddings",
        use_container_width=True
    ):

        if uploaded_pdf is None:

            st.warning(
                "Please upload a PDF."
            )

        else:

            os.makedirs(
                "uploaded_docs",
                exist_ok=True
            )

            save_path = os.path.join(
                "uploaded_docs",
                uploaded_pdf.name
            )

            with open(
                save_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_pdf.getbuffer()
                )

            with st.spinner(
                "Creating embeddings..."
            ):

                create_vector_store(
                    save_path
                )

            st.success(
                "Embeddings Created"
            )
    st.divider()
    
    st.markdown(
        "#### 📄 Indexed PDFs"
    )

    try:

        unique_pdfs=get_all_pdfs()

        if unique_pdfs:

            for pdf in sorted(
                unique_pdfs
            ):

                st.write(
                    f"📄 {pdf}"
                )

        else:

            st.info(
                "No PDFs indexed."
            )

    except Exception:

        st.info(
            "No PDFs indexed."
        )