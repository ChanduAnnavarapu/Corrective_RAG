import streamlit as st
from Corrective_RAG_Architecture import workflow

# Page configuration
st.set_page_config(
    page_title="Corrective RAG",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Corrective RAG")
st.markdown(
    """
    Ask a question about your documents.
    If retrieved documents are insufficient, the system performs corrective retrieval
    using web search before generating the final answer.
    """
)

# User input
question = st.text_input(
    "Enter your question:",
    placeholder="What is the leave policy for employees?"
)

if st.button("Submit"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Processing..."):

        try:
            initial_state = {
                "Question": question,
                "Docs": [],
                "strips": [],
                "kept_strips": [],
                "web_docs": [],
                "refined_context": "",
                "answer": ""
            }

            result = workflow.invoke(initial_state)

            st.success("Response Generated")

            # Final Answer
            st.subheader("Answer")

            answer = (
                result.get("answer")
                or "No answer generated."
            )

            st.write(answer)

            # Retrieved Documents
            docs = result.get("Docs", [])

            if docs:
                with st.expander("Retrieved Documents"):

                    for idx, doc in enumerate(docs, start=1):

                        st.markdown(f"**Document {idx}**")

                        if hasattr(doc, "page_content"):
                            st.write(doc.page_content)
                        else:
                            st.write(doc)

                        st.divider()

            # Debug State
            with st.expander("Workflow State"):

                st.json(
                    {
                        k: str(v)
                        for k, v in result.items()
                    }
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")