from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()  # Load environment variables

# Constants
INDEX_NAME = "langchain-doc-index"

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vector_store.as_retriever()

# Initialize the LLM
chat = ChatOpenAI(verbose=True, temperature=0)

# Load the prompt template from LangChain hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Runnable to retrieve documents
retrieve_runnable = RunnableLambda(retriever.invoke)

# Runnable to format the prompt and pass to LLM
def format_prompt(input_data):
    """Formats the prompt using retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in input_data["documents"]])

    # Debugging: Print input_data keys to check
    print(f"Keys in input_data: {input_data.keys()}")  # Temporary debug line

    return retrieval_qa_chat_prompt.format(context=context, input=input_data["input"])

format_runnable = RunnableLambda(format_prompt)

# Runnable chain: Retrieve -> Format Prompt -> Call LLM
retrieval_chain = RunnableParallel(
    {"documents": retrieve_runnable, "input": RunnableLambda(lambda x: x)}
) | format_runnable | chat

def run_llm(query: str):
    """Runs the full retrieval and LLM response chain."""
    result = retrieval_chain.invoke(query)
    return result

# Main execution
if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")

    print(res.content)  # Access response from LLM
