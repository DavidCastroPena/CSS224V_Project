import regex
import streamlit as st
import json
import os
import time
from datetime import datetime
import random
import sys
import io
import re
import markdown
from retriever.coordinator import Coordinator



class PolicyChatbot:
    def __init__(self):
        """
        Initialize the PolicyChat conversational assistant.
        """
        self.state = {
            'query': None,
            'analysis_option': None,
            'papers_folder': None,
            'local_papers': [],
            'current_stage': 'welcome'
        }
        self.conversation_history = []

    def generate_response(self, user_input=None):
        """
        Generate contextual responses based on the current conversation stage.
        """
        responses = {
            'welcome': [
                "Hi there! I'm PolicyChat, your AI research assistant. I can help you analyze policy documents. Ready to start? ",
                "Welcome! I'm PolicyChat, I specialize in helping you dive deep into policy research. Ready to start?" ,
                "Hello! I'm PolicyChat, ready to explore some policy insights? "
            ],
            'query_clarification': [
                "Could you tell me  about the specific policy situation you're interested in? Please, give me as much detail as possible.",
                "What policy area would you like to research today? Please, give me as much detail as possible.",
                "Please describe the policy context or research question you want to explore. Give me as much detail as possible."
            ],
            'analysis_option': [
                "I can help you analyze papers in two ways:\n\n1. Analyze local papers only. For this option you will provide the path to the local folders where you have this papers and I will analyze them for you. \n2. Analyze local papers and look for additional online resources using Genie API and Semantic Scholar\n\nWhich approach would you prefer?",
            ],
            'papers_folder': [
                "Please provide the full path to the folder containing your policy papers.",
                "I'll need the directory where your research papers are stored. Can you share the path?",
                "Which folder contains the policy documents you want to analyze?"
            ],
            'paper_selection': [
                "Great! Here are the papers I found in the local folder you provided. Please, select up to 5 for our analysis by typing the paper's reference numbers separated by comma.",
                "These are the documents in your folder. Which ones would you like to focus on? Please, select up to 5 for our analysis by typing the paper's reference numbers separated by comma",
                "I've discovered these papers in the provided folder. Please, select up to 5 for our analysis by typing the paper's reference numbers separated by comma"
            ]
        }

        # Add randomness to responses
        if self.state['current_stage'] in responses:
            return random.choice(responses[self.state['current_stage']])
        return "I'm not sure how to respond right now. Let's start over. Please, refresh this page to do so. "

    def process_user_input(self, user_input):
        """
        Process user input based on the current conversation stage.
        """
        if self.state['current_stage'] == 'welcome':
            self.state['current_stage'] = 'query_clarification'
            return self.generate_response()

        elif self.state['current_stage'] == 'query_clarification':
            self.state['query'] = user_input
            self.state['current_stage'] = 'analysis_option'
            return self.generate_response()

        elif self.state['current_stage'] == 'analysis_option':
            if any(option in user_input.lower() for option in ['1', 'one', 'local', 'local papers', 'first']):
                self.state['analysis_option'] = '1'
            elif any(option in user_input.lower() for option in ['2', 'two', 'online', 'additional', 'second']):
                self.state['analysis_option'] = '2'

            else:
                return "Please choose 1 or 2. Let me show the options again:\n\n" + \
                       self.generate_response()

            self.state['current_stage'] = 'papers_folder'
            return self.generate_response()

        elif self.state['current_stage'] == 'papers_folder':
            if os.path.exists(user_input):
                self.state['papers_folder'] = user_input
                files = [f for f in os.listdir(user_input) if f.endswith('.pdf') or f.endswith('.txt')]
                
                if not files:
                    return "No papers found in that folder. Please check the path."

                self.state['available_papers'] = files
                self.state['current_stage'] = 'paper_selection'
                return f"I found {len(files)} papers in the local folder provided. Here they are:\n\n" + \
                       "\n".join(f"{i+1}. {file}" for i, file in enumerate(files)) + \
                       "\n\nWhich papers would you like to analyze? (Please enter the reference numbers separated by commas)"

            else:
                return "That folder doesn't seem to exist. Please provide a valid folder path."

        elif self.state['current_stage'] == 'paper_selection':
            try:
                selected_indices = [int(x.strip())-1 for x in user_input.split(',')]
                selected_papers = [os.path.join(self.state['papers_folder'], 
                                                self.state['available_papers'][i]) 
                                   for i in selected_indices]
                
                self.state['local_papers'] = selected_papers
                self.state['current_stage'] = 'complete'
                
                # Save user inputs
                with open('user_inputs.json', 'w') as f:
                    json.dump({
                        'query': self.state['query'],
                        'analysis_option': self.state['analysis_option'],
                        'papers_folder': self.state['papers_folder'],
                        'local_papers': self.state['local_papers']
                    }, f, indent=4)
                
                return f"👍 Great! I've saved the {len(selected_papers)} papers you indicated for analysis and will now start the research. " 
            
            except (ValueError, IndexError):
                return "Invalid selection. Please enter valid paper numbers."

        return "I'm not sure what to do next. Let's start over."
    
    

def main():
    def message_output(text):
        # Add the message to the Streamlit session state messages
        st.session_state.messages.append({
            "role": "assistant", 
            "content": text
        })
        # Optional: if you want to display the message immediately
        with st.chat_message("assistant"):
            st.write(text)

    st.set_page_config(page_title="PolicyChat", page_icon="🤖", layout="wide")
    st.title("PolicyChat: Your AI Copilot for Policy Research 🤖 📄 ⚖️")

    # Initialize or retrieve session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PolicyChatbot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
        # Add initial assistant greeting
        initial_greeting = st.session_state.chatbot.generate_response()
        st.session_state.messages.append({
            "role": "assistant", 
            "content": initial_greeting
        })

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What's your policy research goal?", key="policy_research_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process user input and get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(0.5)  # Simulate thinking
                response = st.session_state.chatbot.process_user_input(prompt)
                st.write(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Check if all required inputs are collected
        chatbot_state = st.session_state.chatbot.state
        if all([
            chatbot_state['query'], 
            chatbot_state['analysis_option'], 
            chatbot_state['papers_folder'], 
            chatbot_state['local_papers']
        ]):
            # Prepare user inputs for Coordinator
            user_inputs = {
                'query': chatbot_state['query'],
                'option': chatbot_state['analysis_option'],
                'papers_folder': chatbot_state['papers_folder'],
                'local_papers': chatbot_state['local_papers']
            }
            
            # Save to JSON for Coordinator
            with open('user_inputs.json', 'w') as f:
                json.dump(user_inputs, f, indent=4)
            
            try:
                # Call Coordinator
                coordinator = Coordinator(message_output=message_output)
                coordinator.run_pipeline()
                
                # Find the most recent memo file
                directory_path = os.getcwd()
                memo_files = [f for f in os.listdir(directory_path) if f.startswith('memo_') and f.endswith('.md')]
                
                if memo_files:
                    # Sort files by modification time, most recent first
                    most_recent_memo = max(memo_files, key=lambda f: os.path.getmtime(os.path.join(directory_path, f)))
                    markdown_file_path = os.path.join(directory_path, most_recent_memo)
                    
                    # Read and display Markdown file
                    if os.path.exists(markdown_file_path):
                        with open(markdown_file_path, 'r') as file:
                            markdown_content = file.read()
                        
                        # Convert Markdown to HTML
                        html_content = markdown.markdown(markdown_content, extensions=['fenced_code', 'tables'])
                        
                        # Create an expander for the Markdown content
                        with st.expander("Policy Memo"):
                            st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.warning("No memo file found.")
                
            except Exception as e:
                st.error(f"Error in pipeline: {e}")

if __name__ == "__main__":
    main()