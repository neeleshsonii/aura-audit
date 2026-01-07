# Candidate Briefing: Project Aura-Audit

**Duration:** 180 Minutes (Sequential Challenge)

## 1. Objective
You are to build and audit a responsive, responsible customer support AI system. This challenge tests your end-to-end proficiency across Machine Learning fundamentals, Advanced Agentic AI, and Algorithmic Governance.

---

## 2. Project Flow (The 11-Step Lifecycle)

### Phase 1: Foundation & Data Discovery (0-60 min)
1.  **Normalization:** Clean 1,000 raw logs in `data/raw/`. Perform NLP normalization and PII removal.
2.  **Unsupervised Discovery:** Implement **K-Means** clustering to identify core customer intent patterns.
3.  **Label Generation & Semi-supervised:** Assign labels to clusters and use a **Semi-supervised** algorithm (e.g., Label Spreading) to label the unlabelled dataset.
4.  **Supervised Baseline:** Train a **Random Forest** intent classifier on the generated labels.

### Phase 2: Neural Architecture & Decisioning (60-120 min)
5.  **Neural Network:** Develop a **Neural Network** (MLP) for classification and compare it against the baseline.
6.  **Reinforcement Learning:** Optimize a reward-based system using **Q-Learning** for automated support escalation decisions.
7.  **In-processing Fairness:** Audit the model/training loop for bias. Apply **In-processing mitigation** (e.g., re-weighting) to ensure regional parity.

### Phase 3: Intelligence & Post-processing Governance (120-180 min)
8.  **RAG Pipeline:** Implement a **Vector Store (FAISS)** and retrieval logic for customer logs.
9.  **Agentic loop:** Build a **Reasoning Agent** (ReAct) that uses the classifier and Vector DB as tools to resolve tickets autonomously.
10. **Post-processing & XAI:** Perform a final post-processing audit on agent outputs. Use **SHAP** to provide explainability for high-risk decisions.
11. **Compliance Artifacts:** Complete an **Algorithmic Impact Assessment (AIA)** and a **Model Card** documenting the system's risk profile.

---

## 3. Mandatory technical constraints
- **Determinism:** You MUST use **Random Seed 42** for all stochastic operations (Scikit-learn, NumPy, etc.).
- **Modularity:** Keep Foundation, Intelligence, and Governance logic clearly separated.
- **Tools:** Use the provided `requirements.txt` environment.

---

## 4. Evaluation Rubric
*   **Technical Logic (Foundations & Intelligence):** Accuracy of clustering, model performance, and agentic reasoning paths.
*   **Responsible AI (Governance):** Depth of bias detection at Pre, In, and Post-processing stages.
*   **Critical Thinking:** Content of your AIA and the quality of your explainability reports.

Good luck with the Aura-Audit challenge!
