import pandas as pd
import streamlit as st
import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")




# --------------------------------- 
# Load reservations data from Excel file
# ---------------------------------
reservations_df = pd.read_csv(r'C:\Users\ggsan\Desktop\Ironhack\2. LABS\9.WEEK_Lab\data_clean.csv')
# Convert 'checkin' and 'checkout' columns to datetime
reservations_df[['checkin', 'checkout']] = reservations_df[['checkin', 'checkout']].apply(pd.to_datetime)

# Convert 'checkin' and 'checkout' columns to date only (without time)
reservations_df['checkin'] = reservations_df['checkin'].dt.date
reservations_df['checkout'] = reservations_df['checkout'].dt.date

# Convert DataFrame to dictionary for easier access
RESERVATIONS = reservations_df.rename(columns={'checkin': 'checkin', 'checkout': 'checkout'}).set_index('resv_id').T.to_dict()

# Initialize SPA_APPOINTMENTS for each rfeservation ID
SPA_APPOINTMENTS = {resv_id: [] for resv_id in RESERVATIONS.keys()}



# --------------------------------- 
# Load knowledge base text
# ---------------------------------
# 1. Load your knowledge base text
with open("hotel_info.md", "r", encoding="utf-8") as f:
    knowledge_text = f.read()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,     # Split text into chunks of 150 characters
    chunk_overlap=150   
)
docs = text_splitter.create_documents([knowledge_text])


# ---------------------------------
# Initialize RAG Components
# ---------------------------------

os.environ["OPENAI_API_KEY"] = "Your_OpenAI_API_Key" # please replace with your OpenAI API key


def init_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))  
    # Load the local Chroma DB
    db = Chroma(
        collection_name="hotel_info",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    db.add_documents(docs)
    db.persist()
    return db.as_retriever(search_kwargs={"k": 3})

retriever = init_vectorstore()

# We can set up a custom prompt to instruct the LLM to act as a helpful hotel assistant.
prompt_template = """
You are a helpful hotel concierge. Use the following context to answer the user's question. 
If the answer is not in the context, say "I'm not sure, let me check with our Team member." 
Context: 
{context}
Question: {question}
Answer in a concise and friendly way:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=1),  # Specify GPT-4 model
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# ---------------------------------
# Additional Logic (Reservations)
# ---------------------------------
def handle_reservation_id(resv_id):
    resv_id = resv_id.upper()
    if resv_id in RESERVATIONS:
        return (
            f"Reservation ID **{resv_id}** found!\n"
            f"- Room Type: {RESERVATIONS[resv_id]['room_type']}\n"
            f"- Check-In: {RESERVATIONS[resv_id]['checkin']}\n"
            f"- Check-Out: {RESERVATIONS[resv_id]['checkout']}\n"
            f" * How else can I assist you?"
        )
    else:
        return "Sorry, I couldn't find that reservation ID. Please try again."

def schedule_spa(res_id, user_input):
    """
    Naive approach: if user says something like 
    'Schedule spa on 2025-01-30 at 10:00', 
    parse date/time and store it.
    """
    try:
        # Example parse (in production, use dateparser or a slot-filling approach)
        after_on = user_input.lower().split("on")[-1].strip()  # "2025-01-30 at 10:00"
        date_part, time_part = after_on.split("at")
        date_str = date_part.strip()
        time_str = time_part.strip()
        spa_datetime = datetime.datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M")

        SPA_APPOINTMENTS[res_id].append(spa_datetime)
        return f"Your spa appointment is scheduled on {spa_datetime}."
    except Exception:
        return ("I couldn’t parse your date/time. "
                "Please use: 'Schedule spa on YYYY-MM-DD at HH:MM' format.")

# ---------------------------------
# Streamlit Chat App
# ---------------------------------
def main():
    st.set_page_config(page_title="Hotel RAG Chatbot", layout="centered")
    st.title("Orange Hotel Service Chatbot (RAG)")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "resv_id" not in st.session_state:
        st.session_state["resv_id"] = None

    # Display chat history
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    user_input = st.chat_input("Ask about our hotel services, or type your reservation ID...")

    if user_input:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 1. Check if user provided a reservation ID directly
        if user_input.upper() in RESERVATIONS:
            st.session_state["resv_id"] = user_input.upper()
            response = handle_reservation_id(st.session_state["resv_id"])

        # 2. If user wants to schedule a spa, we look for the phrase "schedule spa"
        elif "schedule spa" in user_input.lower() and st.session_state["resv_id"]:
            response = schedule_spa(st.session_state["resv_id"], user_input)

        # 3. Else pass the query to the RAG chain
        else:
            response = qa_chain.run(user_input)
            # You could do a post-processing check if the answer is "I'm not sure..."

        # Add assistant message
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()
 
 