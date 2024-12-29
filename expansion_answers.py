from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import umap
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import matplotlib.pyplot as plt





# Load environment variables from .env file
load_dotenv()

# Get OPENAI variables
openai_key = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=openai_key)

# Read PDF file
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]
# print(
#     word_wrap(
#         pdf_texts[10],
#         width=100,
#     )
# )


# split the text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)

character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

#print(word_wrap(character_split_texts[]))
#print(f"\nTotal chunks: {len(character_split_texts)}")

# Assigning 256 tokens per chunk, making the data more manageable to process. 
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

#print(word_wrap(token_split_texts[0]))
#print(f"\nTotal chunks: {len(token_split_texts)}")

# Creating the embedding function
embedding_function = SentenceTransformerEmbeddingFunction()
#print(embedding_function([token_split_texts [0]]))

# vectorize the microsoft document 
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# Add the tokenized chunks of text to the empty array of vecotrized inf
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

#count = chroma_collection.count()
#print(count)

# initial query
query = "What was the total revenue for the year?"

# get the raw results from a simple inital query
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# print out the inital results
#for document in retrieved_documents:
#    print(word_wrap(document))
#    print("\n")

# A function utilizing OpenAI's models to generate a reponse 
# in the context of a financial research assistant from the given query
def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

# creating the initial query for the first LLM (OpenAI's model) and generating a hypotheitcal answer
original_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
#print(word_wrap(joint_query))

# To generate the final output with the joint query: The original query combined with the 
# joint query created with OpenAI's API
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

# for doc in retrieved_documents:
#     print(word_wrap(doc))
#     print("")

# To retreive the most relevant embeddings and project them for further analysis
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# To retreive the embeddings from the final reuslt of inputting the joint query,
# the embeddings from the original query, and the embeddings from the joint query respectively. 
retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

# Generating the projection objects
projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

# Plotting the embeddings from the full dataset (derived from the entire document), the embeddings from the retrieved 
# documents after inputting the joint query, the embeddings from the original query, and the embeddings from the orange query. 
# The joint query is much closer to the retrieved documents than the original query.
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f'Original Query: "{original_query}"')
plt.axis("off")
plt.show()  # display the plot
