import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Note: OpenAI components are optional - the system works without them
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Note: OpenAI components not available. Using local models only.")

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class AuraAgent:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
    
        self.vector_store = None
        self.classifier = None
        self.q_learning_agent = None
        self.vectorizer = None    
        # Load trained models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models."""
        try:
            if os.path.exists('models/random_forest_model.pkl'):
                self.classifier = joblib.load('models/random_forest_model.pkl')
                print("Loaded Random Forest classifier")
            if os.path.exists('models/tfidf_vectorizer.pkl'):
                self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
                print("Loaded TF-IDF vectorizer")
            if os.path.exists('models/q_learning_agent.pkl'):
                self.q_learning_agent = joblib.load('models/q_learning_agent.pkl')
                print("Loaded Q-Learning agent")
        except Exception as e:
            print(f"Note: Some models not loaded: {e}")
            
    
    def setup_rag(self, texts):
        """
        Task 3.1: Chunk texts and load into FAISS vector store.
        """
        print("\nSetting up RAG pipeline...")
        
        # Use FAISS with OpenAI embeddings
        documents = [
            Document(page_content=text, metadata=meta if meta else {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        
        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )
        print(f"FAISS vector store created with {len(texts)} documents")
        
        
    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents with scores
        """
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            for doc, score in results
        ]
        
        
    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify user intent using trained classifier.
        
        Returns:
            Dictionary with intent prediction and confidence
        """
        if self.classifier is None or self.vectorizer is None:
            return {'intent': 'unknown', 'confidence': 0.0}
        
        # Clean and vectorize text
        from src.foundation.pipeline import clean_text
        clean = clean_text(text)
        X = self.vectorizer.transform([clean]).toarray()
        
        # Predict
        intent = self.classifier.predict(X)[0]
        probas = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probas))
        
        intent_names = ['billing', 'tech_support', 'login_issue', 
                       'refund_request', 'feature_request']
        if intent < len(intent_names):
            intent_name = intent_names[intent]
        else:
            intent_name = f'cluster_{intent}'
        return {
            'intent': intent_name,
            'intent_id': int(intent),
            'confidence': confidence
        }
        
    
    def get_routing_decision(self, intent: str, sentiment: str = 'Neutral') -> Dict[str, Any]:
        """
        Get routing decision from Q-Learning agent.
        
        Returns:
            Dictionary with recommended action
        """
        if self.q_learning_agent is None:
            return {'action': 'escalate_to_human', 'reason': 'No routing model available'}
        
        # Map intent to ID
        intent_map = {
            'billing': 0,
            'tech_support': 1,
            'login_issue': 2,
            'refund_request': 3,
            'feature_request': 4
        }
        
        sentiment_map = {
            'Satisfied': 0, # 0=Satisfied, 1=Neutral, 2=Frustrated
            'Neutral': 1,
            'Frustrated': 2
        }
        
        intent_id = intent_map.get(intent, 1)
        sentiment_id = sentiment_map.get(sentiment, 1)
        
        state = self.q_learning_agent.get_state_id(intent_id, sentiment_id)
        policy = self.q_learning_agent.get_policy()
        action_id = policy[state]
        action_name = self.q_learning_agent.action_names[action_id]
        
        return {
            'action': action_name,
            'action_id': int(action_id),
            'confidence': 'high'
        }

    def run_agentic_loop(self, user_query):
        """
        Task 3.2: Implement a ReAct loop.
        - The agent should decide to use tools (e.g., retrieve from Vector DB).
        - Use fixed prompt templates for reproducibility.
        """
        print("\n" + "="*70)
        print("AGENTIC LOOP EXECUTION (ReAct Pattern)")
        print("="*69)
        print(f"\nUser Query: {user_query}")
        
        actions_taken = []
        
        # Step 1: Classify Intent
        print("\n[THOUGHT] I should first understand what the user needs.")
        print("[ACTION] Classifying user intent...")
        
        intent_result = self.classify_intent(user_query)
        actions_taken.append({
            'step': 1,
            'action': 'classify_intent',
            'result': intent_result
        })
        
        print(f"[OBSERVATION] Intent: {intent_result['intent']} "
              f"(confidence: {intent_result['confidence']:.2f})")
        
        # Step 2: Retrieve Relevant Context
        print("\n[THOUGHT] I should find relevant information from our knowledge base.")
        print("[ACTION] Retrieving similar support cases...")
        
        retrieved_docs = self.retrieve_context(user_query, k=3)
        actions_taken.append({
            'step': 2,
            'action': 'retrieve_context',
            'result': {'num_docs': len(retrieved_docs)}
        })
        
        print(f"[OBSERVATION] Found {len(retrieved_docs)} relevant documents")
        for i, doc in enumerate(retrieved_docs[:2], 1):
            print(f"  {i}. Score: {doc['score']:.3f} - {doc['text'][:100]}...")
        
        # Step 3: Determine Routing Decision
        print("\n[THOUGHT] Based on the intent, I should decide how to route this.")
        print("[ACTION] Consulting routing policy...")
        
        routing = self.get_routing_decision(
            intent_result['intent'],
            sentiment='Neutral'  # In production, would detect sentiment
        )
        actions_taken.append({
            'step': 3,
            'action': 'get_routing_decision',
            'result': routing
        })
        
        print(f"[OBSERVATION] Recommended action: {routing['action']}")
        
        # Step 4: Formulate Response
        print("\n[THOUGHT] Now I can provide a comprehensive response.")
        
        response = self._formulate_response(
            user_query,
            intent_result,
            retrieved_docs,
            routing
        )
        
        print("\n[FINAL RESPONSE]")
        print("-" * 70)
        print(response['message'])
        print("-" * 70)
        
        return {
            'query': user_query,
            'intent': intent_result,
            'routing': routing,
            'retrieved_docs': retrieved_docs,
            'actions_taken': actions_taken,
            'response': response
        }
        
        
    def _formulate_response(self, query, intent_result, retrieved_docs, routing):
        """Generate final response based on agent's observations."""
        
        # Create response based on routing decision
        if routing['action'] == 'escalate_to_human':
            message = (
                f"I understand you're experiencing a {intent_result['intent']} issue. "
                f"I'm connecting you with a human agent who can better assist you.\n\n"
                f"Based on similar cases, here's some information that might help:\n"
            )
        elif routing['action'] == 'use_automated_response':
            message = (
                f"I can help you with your {intent_result['intent']} request. "
                f"Here's what I found:\n\n"
            )
        else:  # request_more_info
            message = (
                f"To better assist with your {intent_result['intent']} request, "
                f"I need a bit more information.\n\n"
            )
        
        # Add context from retrieved documents
        if retrieved_docs:
            message += "Relevant information from our knowledge base:\n"
            for i, doc in enumerate(retrieved_docs[:2], 1):
                message += f"{i}. {doc['text'][:150]}...\n"
        
        return {
            'message': message,
            'intent': intent_result['intent'],
            'action': routing['action'],
            'confidence': intent_result['confidence']
        }

def setup_knowledge_base():
    """Create a knowledge base from support logs."""
    print("\nBuilding knowledge base from support logs...")
    
    # Load processed support logs
    if os.path.exists('aura_audit/data/processed/support_logs_clustered.csv'):
        df = pd.read_csv('aura_audit/data/processed/support_logs_clustered.csv')
    elif os.path.exists('aura_audit/data/raw/support_logs.csv'):
        df = pd.read_csv('aura_audit/data/raw/support_logs.csv')
    else:
        print("No support logs found. Please run pipeline.py first.")
        return [], []
    
    # Sample representative examples from each cluster/region
    texts = []
    metadatas = []
    
    # Get diverse sample
    if 'cluster' in df.columns:
        for cluster in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster].head(20)
            texts.extend(cluster_df['clean_text'].tolist() if 'clean_text' in df.columns else cluster_df['text'].tolist())
            metadatas.extend([
                {'cluster': cluster, 'region': row['region']}
                for _, row in cluster_df.iterrows()
            ])
    else:
        sample_df = df.sample(min(100, len(df)), random_state=RANDOM_SEED)
        texts = sample_df['text'].tolist()
        metadatas = [{'region': row['region']} for _, row in sample_df.iterrows()]
    
    print(f"✓ Prepared {len(texts)} documents for knowledge base")
    return texts, metadatas

if __name__ == "__main__":    
    # Initialize agent with API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    agent = AuraAgent(api_key=api_key)
    
    # Setup RAG pipeline
    texts, metadatas = setup_knowledge_base()
    if texts:
        agent.setup_rag(texts, metadatas)
    
    # Test queries
    test_queries = [
        "I was charged twice on my credit card for the same invoice",
        "The app keeps crashing when I try to login",
        "I want a refund for my last purchase"
    ]
    
    print("\n" + "="*70)
    print("TESTING AGENT WITH SAMPLE QUERIES")
    print("="*70)
    
    for query in test_queries:
        result = agent.run_agentic_loop(query)
        
        # Save result
        os.makedirs('outputs', exist_ok=True)
        output_file = f"outputs/agent_response_{hash(query) % 10000}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    print("\nAgent execution complete. Results saved to outputs/ directory.")
