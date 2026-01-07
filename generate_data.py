import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility as requested
random.seed(42)
np.random.seed(42)

def generate_data(n=1000):
    regions = ['North', 'South', 'East', 'West']
    # Simulate a bias: 'South' region has more complex/frustrated language proxies
    languages = ['English', 'Spanish', 'Hindi', 'French']
    
    intents = ['billing', 'tech_support', 'login_issue', 'refund_request', 'feature_request']
    
    data = []
    for i in range(n):
        region = random.choice(regions)
        language = random.choice(languages)
        
        # Base templates for intents
        if random.random() < 0.2:
            intent = 'billing'
            text = f"I have a question about my invoice {random.randint(1000, 9999)}. My credit card was charged twice."
        elif random.random() < 0.4:
            intent = 'tech_support'
            text = "The application keeps crashing on my mobile device. I tried restarting but it didn't help."
        elif random.random() < 0.6:
            intent = 'login_issue'
            text = "I forgot my password and the recovery email is not arriving. Pin code: 1234 (PII Example)"
        elif random.random() < 0.8:
            intent = 'refund_request'
            text = "I am not satisfied with the service. Please refund my last payment immediately."
        else:
            intent = 'feature_request'
            text = "Could you add a dark mode to the dashboard? It's too bright at night."

        # Add "Noise" and "PII"
        if random.random() < 0.1:
            text += f" My email is user{i}@example.com and my phone is 555-0199."
        
        # Simulate Bias: South region logs are intentionally more "noisy" or "frustrated" 
        # to test candidate's ability to detect demographic skew in sentiment or error rates later.
        if region == 'South':
            text = text.upper() + " !!! URGENT HELP NEEDED !!!"
            sentiment = 'Frustrated'
        else:
            sentiment = random.choice(['Neutral', 'Satisfied']) if random.random() > 0.3 else 'Frustrated'

        data.append({
            'log_id': f'LOG_{i:04d}',
            'text': text,
            'region': region,
            'language': language,
            'actual_intent': intent, # This will be hidden from the candidate for Phase 1
            'actual_sentiment': sentiment # This will be hidden for Phase 1
        })
    
    df = pd.DataFrame(data)
    
    # Save raw (unlabelled for candidate) and reference (labeled for grading)
    raw_path = 'aura_audit/data/raw/support_logs.csv'
    ref_path = 'aura_audit/data/reference/support_logs_labeled.csv'
    
    os.makedirs('aura_audit/data/raw', exist_ok=True)
    os.makedirs('aura_audit/data/reference', exist_ok=True)
    
    df[['log_id', 'text', 'region', 'language']].to_csv(raw_path, index=False)
    df.to_csv(ref_path, index=False)
    
    print(f"Generated {n} logs at {raw_path}")

if __name__ == "__main__":
    generate_data()
