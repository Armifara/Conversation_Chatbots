# To built a web application

# streamlit is used to create the web interface
import streamlit as st

# langchain_groq is used to interact with Groq's language model
from langchain_groq import ChatGroq

# langchain_core is used for handling messages
# HumanMessage is used to create a message from the user
from langchain_core.messages import HumanMessage

# langgraph.checkpoint.memory is used to save the conversation memory
from langgraph.checkpoint.memory import MemorySaver

# langgraph.graph is used to create and manage the state graph
# START is the initial state of the graph
# MessagesState is used to handle message states
# StateGraph is used to create the state graph
from langgraph.graph import START, MessagesState, StateGraph

# uuid and os are standard Python libraries
# uuid is used to generate unique identifiers
# os is used to interact with the operating system
import uuid
import os

print(uuid.uuid4())

# Load the GROQ API key in the environment variable
# ensure you have set the GROQ_API_KEY in your Streamlit secrets

# environ is a dictionary in Python that contains the user's environmental variables
# st.secrets is a Streamlit feature that allows you to securely store and access sensitive information like API keys
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# Core Logic of the Chatbot

# st.cache_resource is a Streamlit decorator that caches the result of a function
# Ensures model + graph are created once and reused across reruns
@st.cache_resource

def get_app():
    # Initialize Groq LLM (non-streaming mode for full responses only)
    model = ChatGroq(model="llama-3.3-70b-versatile") # type: ignore

    # Create a LangGraph workflow with a MessagesState schema
    # state_schema defines the structure of the state, here it is MessagesState
    workflow = StateGraph(state_schema=MessagesState)

    # Define how the model is called inside the workflow
    # state is the current state of the messages
    def call_model(state: MessagesState):

        # Take conversation history, send it to the model, return its reply
        # invoke method sends the messages to the model and gets the response
        response = model.invoke(state["messages"])
        return {"messages": response}

    # Add model node to workflow
    # .add_edge connects two nodes in the graph
    # .add_node adds a new node to the graph

    # EXAMPLE: START → "model" → call_model, this means the workflow starts at START,
    # then goes to the "model" node, which calls the call_model function

    # Here, we connect the START node to the model node
    workflow.add_edge(START, "model")  # Start → model
    workflow.add_node("model", call_model) # model → call_model

    # MemorySaver
    # Used to save the conversation memory
    memory = MemorySaver()

    # Compile workflow
    # checkpointer is used to specify where to save the memory
    return workflow.compile(checkpointer=memory)

# Get the app instance
# get_app function is called to get the compiled workflow
app = get_app()


#  Streamlit UI

# st.title sets the title of the web app
st.title("Llama 3.3 Chatbot")

# st.write is used to write text to the web app
st.write("This is a simple chatbot using Llama 3.3 and LangGraph.")

# Thread ID -- used to group messages in the same conversation

# Initialize a Conversational Thread ID (used to show conversation history)
# st.session_state is a Streamlit feature that allows you to store variables across reruns

# if is used to check if "thread_id" is not in st.session_state
# If not, it generates a new UUID and assigns it to st.session_state.thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar -> Conversation Controls, Allows user to start new conversations
# st.sidebar is used to create a sidebar in the web app
# with is used to create a context for the sidebar

with st.sidebar:
    # st.header sets the header of the sidebar
    st.header("Conversation Controls")

    # st.button creates a button in the sidebar, type "primary" makes it a primary button
    if st.button("Start New Conversation", type="primary"):
        # Generate a new UUID for the new conversation
        st.session_state.thread_id = str(uuid.uuid4())
        
        # Notify the user
        st.write("Started a new conversation.")

        # rerun the app to reflect the new conversation
        st.rerun()

    # To Display the current thread ID
    # For Reference: st.write(f"Current Thread ID: {st.session_state.thread_id}")

# Thread Configuration
# Tells langgraph which conversation thread to use

# config is a dictionary that holds the configuration for the app
# "configurable" is a key in the dictionary that holds another dictionary
# "thread_id" is a key in the inner dictionary that holds the current thread ID 
config = {"configurable": {"thread_id": st.session_state.thread_id}}


# To Retrieve and Display Conversation History

# state is the current state of the app
# get_state method retrieves the state based on the provided configuration
state = app.get_state(config)  # type: ignore

# messages is a list that holds the conversation messages
# state.values.get("messages", []) retrieves the messages from the state
messages = state.values.get("messages", [])

# Loop through the messages and display them in the web app
# Show the conversation history

for msg in messages:
     # role is set to "user" if the message type is "human", otherwise it is set to "assistant"
     role = "user" if msg.type == "human" else "assistant"
     # with is used to create a context for the chat message
     # st.chat_message creates a chat message in the web app
     with st.chat_message(role):
        # st.markdown is used to display the message content in markdown format
         st.markdown(msg.content)

# Chat Input Box - To get user input and display it in the chat
if user_input := st.chat_input("Ask me anything!"):
    
    # Show user input immediately in chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # To send user input to the model and get a response
    # app.invoke sends the user input to the model and gets the response
    output = app.invoke(
        {"messages": [HumanMessage(user_input)]
        }, config
    )  # type: ignore
    # output is a dictionary that holds the response from the model

    # ai_message is the latest message from the assistant
    # output["messages"] contains all messages, we get the last one
    # [-1] gets the last item in the list
    ai_message = output["messages"][-1]  # Get the assistant's latest reply

    
    # Assistant's response
    # Display the assistant's response in the chat
    with st.chat_message("assistant"):
        st.markdown(ai_message.content)