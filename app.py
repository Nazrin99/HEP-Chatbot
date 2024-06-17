import streamlit as st
import random
import time
from chat_backend import chat

# Caller functions

def response_generator(user_prompt: str):
    response = chat(user_prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Frontend Code

st.image("um.jpg", use_column_width=True)
st.title("HEP Chatbot")
st.markdown("This is a chatbot to answer FAQs from the Student Affairs Department of Universiti Malaya (HEP UM)."
            "For feedback on improvements or complaints, please direct them to u2000443@siswa.um.edu.my 😁")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey, I'm HEP Chatbot! Do you have any questions for me?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)


if prompt := st.chat_input("What is up?"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})

