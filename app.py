import streamlit as st
from utils import qa_pipeline

chain = qa_pipeline()

def main():
    global chain
    # Set the title of the web application
    st.title('Indian Law Q&A Bot')

    # Initialize the session state if it doesn't exist
    if 'chat_log' not in st.session_state:
        st.session_state.chat_log = []

    # Get the user's question
    user_input = st.text_input("You:")

    # On user input, generate response and add to the chat log
    if user_input:
        # Generate the answer
        bot_output = chain(user_input)
        bot_output = bot_output['result']
        # Add the user input and bot output to the chat log
        st.session_state.chat_log.append({"User": user_input, "Bot": bot_output})
        # Clear the input box
        st.text_input("You:", value="", key="unique")

    # Display the chat log
    for exchange in st.session_state.chat_log:
        st.markdown(f'**You:** {exchange["User"]}')
        st.markdown(f'**Bot:** {exchange["Bot"]}')

if __name__ == "__main__":
    main()
