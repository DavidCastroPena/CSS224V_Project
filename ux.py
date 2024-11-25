import streamlit as st
import json
import os
import sys
from pathlib import Path

# Add the parent directory to Python path to import local modules
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

print('ddddd', current_dir, 'ddd')

#from retriever.finder import Finder
#from retriever.coordinator import Coordinator

def create_papers_folder():
    """Create a temporary papers folder if it doesn't exist."""
    papers_folder = current_dir / "papers"
    papers_folder.mkdir(exist_ok=True)
    return str(papers_folder)

def save_uploaded_file(uploaded_file, papers_folder):
    """Save uploaded file to the papers folder."""
    file_path = os.path.join(papers_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

def main():
    st.title("PolicyChat: Policy Research Assistant")

    # Initialize session state for tracking steps
    if 'step' not in st.session_state:
        st.session_state.step = 'query'
    
    if 'local_papers' not in st.session_state:
        st.session_state.local_papers = []

    # Query Step
    if st.session_state.step == 'query':
        st.header("Enter Your Policy Situation")
        query = st.text_area("Describe your policy situation or research question:")
        
        if st.button("Next"):
            if query.strip():
                st.session_state.query = query
                st.session_state.step = 'analysis_option'
                st.experimental_rerun()
            else:
                st.warning("Please enter a policy situation.")

    # Analysis Option Step
    elif st.session_state.step == 'analysis_option':
        st.header("Select Analysis Option")
        option = st.radio(
            "Choose an analysis option:", 
            ["Analyze up to 5 local papers", 
             "Analyze up to 5 local papers and 10 additional papers retrieved from the web"]
        )
        
        if st.button("Next"):
            st.session_state.option = "1" if option == "Analyze up to 5 local papers" else "2"
            st.session_state.step = 'upload_papers'
            st.experimental_rerun()

    # Paper Upload Step
    elif st.session_state.step == 'upload_papers':
        st.header("Upload Policy Papers")
        
        # Create a temporary papers folder
        papers_folder = create_papers_folder()
        st.session_state.papers_folder = papers_folder

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload up to 5 PDF or TXT files", 
            type=['pdf', 'txt'], 
            accept_multiple_files=True
        )

        # Save uploaded files
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("You can only upload up to 5 files. Please remove some files.")
            else:
                local_papers = []
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file, papers_folder)
                    local_papers.append(file_path)
                st.session_state.local_papers = local_papers

        # Proceed to analysis
        if st.button("Run Analysis"):
            st.session_state.step = 'analysis'
            st.experimental_rerun()

    # Analysis Step
    elif st.session_state.step == 'analysis':
        st.header("Policy Research Analysis")
        
        # Capture print outputs
        class StreamlitOutput:
            def __init__(self):
                self.output = []
            def write(self, text):
                self.output.append(text)
            def flush(self):
                pass

        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = streamlit_output = StreamlitOutput()

        # Prepare user inputs JSON
        user_inputs = {
            "query": st.session_state.query,
            "option": st.session_state.option,
            "papers_folder": st.session_state.papers_folder,
            "local_papers": st.session_state.local_papers
        }

        # Save user inputs
        inputs_path = current_dir / "user_inputs.json"
        with open(inputs_path, 'w') as f:
            json.dump(user_inputs, f, indent=4)

        # Run Coordinator
        coordinator = Coordinator()
        coordinator.run_pipeline()

        # Restore stdout
        sys.stdout = old_stdout

        # Display process outputs
        st.subheader("Analysis Process")
        for line in streamlit_output.output:
            st.text(line)

        # Check for generated memo
        memo_files = list(current_dir.parent.glob("memo_*.md"))
        if memo_files:
            most_recent_memo = max(memo_files, key=os.path.getctime)
            st.subheader("Generated Policy Memo")
            with open(most_recent_memo, 'r') as f:
                st.markdown(f.read())
            
            # Option to download memo
            with open(most_recent_memo, 'r') as f:
                st.download_button(
                    label="Download Policy Memo",
                    data=f.read(),
                    file_name=most_recent_memo.name,
                    mime="text/markdown"
                )
        else:
            st.warning("No memo was generated.")

        # Reset for potential new analysis
        if st.button("Start New Analysis"):
            st.session_state.step = 'query'
            st.experimental_rerun()

if __name__ == "__main__":
    main()