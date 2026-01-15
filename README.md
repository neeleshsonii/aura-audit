## Aura-Audit: Customer Support AI
I built this end-to-end AI system for the Aura-Audit evaluation. It covers everything from raw data cleaning to deploying a responsible ReAct agent. The project is split into three modular phases: Foundation, Intelligence, and Governance.

## 🛠️ How it Works
### 1. Foundation (Data & Baselines)

   Data Cleaning: Cleaned 1,000 logs and stripped out all PII.

    Labeling: Used K-Means to find issue clusters and propagated those labels to the full set.

    Baseline: Set up a Random Forest classifier as a starting point.

### 3. Intelligence (Neural & RL)

   MLP Classifier: Built a neural network for intent classification.

    RL Optimization: Used Q-Learning to make the decision-making script smarter via rewards.

    Bias Audit: Applied re-weighting during training to make sure the model stays fair.

### 4. Governance (Agents & Compliance)

   RAG Pipeline: Set up a FAISS Vector Store so the agent can retrieve actual logs.

    ReAct Agent: Implemented an agent that uses the classifier and RAG as tools to solve tasks.

    Audit & Explainability: Used SHAP to explain model decisions and finalized the Model Card/Impact Assessment.

## 📁 Structure
foundation.py: Data cleaning and initial ML.

intelligence.py: Neural nets, RL, and bias mitigation.

governance.py: RAG, ReAct agent, and SHAP audits.

Model_Card.md: Compliance and impact details.

## Contact: Neelesh Soni – neeleshsoni54@gmail.com
