import streamlit as st

# Read the content of README.md
with open('README.md', 'r') as readme_file:
    readme_content = readme_file.read()

# Display the README content in the Streamlit app
st.markdown(readme_content)
