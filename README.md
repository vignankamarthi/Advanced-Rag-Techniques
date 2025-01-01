# Expansion by Answers: Advanced RAG Techniques

This project demonstrates the implementation of the **Expansion by Answers** technique, an advanced Retrieval-Augmented Generation (RAG) method. By utilizing **ChromaDB's embedding functions** and **OpenAI's API**, this technique enhances the performance of a Large Language Model (LLM) by providing more contextually relevant information from PDF documents.

### Workflow Overview:

1. **Extract and Process Text**: Extract and process raw text from PDF documents to desired formatting.
2. **Chunk and Embed**: Split the text into manageable chunks and generate vector embeddings.
3. **Generate Answers**: Query the LLM to generate potential answers.
4. **Retrieve Context**: Use embeddings to retrieve relevant context for the query.
5. **Expand and Refine**: Feed the expanded query back into the LLM for a more accurate, context-rich response.

This project explores and refines RAG methodologies, focusing on **query expansion** to maximize the contextual understanding of the LLM. The approach is applicable across various domains to improve information retrieval and response accuracy.

## Getting Started

Follow these steps to set up the project and install the necessary dependencies:

1. **Clone the repository**:
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. Install Dependencies
    Install the required dependencies using the requirements.txt file:
    ```bash
    pip install -r requirements.txt

3. Set up OpenAI API access:
    This project requires access to OpenAI’s API. Follow these steps:
	* Create an OpenAI account: Visit OpenAI’s website to sign up if you don’t already have an account.
	* Generate an API key: 
         - Go to the API Keys page in your OpenAI dashboard.
	     - Click “Create new secret key” and copy the key.
	* Purchase API usage credits: Ensure that your OpenAI account has sufficient credits or a billing plan set up for API usage.

4. Add you OpenAI API Key to the Project
    * Create a .env file in the root of the repository:
     ```bash
        touch .env
     ```
    * Add the following line to the .env file, replacing your-api-key-here with your actual OpenAI API key:
     - OPENAI_API_KEY=place-your-api-key-here

## Conclusions

The results from the graph illustrate key insights about the effectiveness of the **Expansion by Answers** technique:

- **Grey Dots**: Represent the embedded text of the entire document.
- **Red X**: Denotes the original query embedding.
- **Orange X**: Denotes the joint query embedding (a combination of the original query and additional retrieved context).
- **Green Circles**: Represent the outputs from the joint query.

The visualization clearly shows that the **green circles**, representing the outputs of the joint query, are significantly closer to the **orange X** (joint query embedding) compared to the **red X** (original query embedding). This indicates a marked improvement in the relevance of retrieved information after applying the joint query, thereby enhancing the contextual understanding and performance of the LLM.

These results highlight the potential of this technique to significantly refine the retrieval process and improve response accuracy in Retrieval-Augmented Generation workflows.
