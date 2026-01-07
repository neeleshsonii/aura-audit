import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

class AuraAgent:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
        self.vector_store = None
    
    def setup_rag(self, texts):
        """
        Task 3.1: Chunk texts and load into FAISS vector store.
        """
        # TODO: Candidate implements this
        pass

    def run_agentic_loop(self, user_query):
        """
        Task 3.2: Implement a ReAct loop.
        - The agent should decide to use tools (e.g., retrieve from Vector DB).
        - Use fixed prompt templates for reproducibility.
        """
        # TODO: Candidate implements this
        pass

if __name__ == "__main__":
    print("Setting up Aura Agent...")
