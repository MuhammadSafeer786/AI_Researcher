# PDF QA with LLMs and Streamlit

This project leverages Large Language Models (LLMs) to create a practical tool that allows users to input as many PDF files and ask questions related to it. This is particularly useful for researchers working with specialized content (LAW etc).

## Features

- **Data Loading**: Uses PDF loader to fetch and process the PDF's content.
- **Data Splitting**: Optimizes the usage of OpenAI API tokens by splitting the article into manageable chunks.
- **Vector Database**: Implements FAISS to create a vector database for faster and more efficient searches.
- **Streamlit Integration**: The entire application is built with Streamlit, providing a user-friendly interface.

## Setup
### Installation

1. Clone this repository:
    ```bash
    https://github.com/user_name/AI_Researcher.git
    cd AI_Researcher
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
    ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```
    
## Acknowledgments

- Huge thanks to the [codebasics YouTube channel](https://www.youtube.com/c/codebasics) for their amazing tutorials that helped me build this project.
