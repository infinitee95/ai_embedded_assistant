import streamlit as st
from query import process_query
import asyncio
import time

# Ensure an event loop is available (from previous fix)
if not asyncio.get_event_loop_policy().get_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Streamlit app setup
st.title("Document Assistant with Semantic Layer")
st.write("Ask questions about your documents (e.g., whitepapers, ebooks).")

# Query input field
query = st.text_input("Enter your question:", "What are the key trends in business analytics?")

# Process query when button is clicked
if st.button("Ask"):
    if query:
        # Use st.status to show progress
        with st.status("Processing query...", expanded=True) as status:
            status.update(label="Step 1: Enriching query with semantic terms...")
            answer, chunks, metadata = process_query(query, status=status)
            
            # Update status when done
            status.update(label="Processing complete!", state="complete")
        
        # Display the answer
        st.subheader("Answer:")
        st.write(answer)

        # Display source excerpts
        st.subheader("Sources:")
        for chunk, meta in zip(chunks, metadata):
            st.write(f"- **{meta['document']}**: {chunk[:200]}...")
    else:
        st.error("Please enter a question.")