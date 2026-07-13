import os
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

class AudioSignalAgent:
    def __init__(self, api_key, project_id, url="https://us-south.ml.cloud.ibm.com"):
        """
        Initializes the Agent using IBM watsonx.ai and Granite Model.
        """
        self.credentials = {
            "url": url,
            "apikey": api_key
        }
        self.project_id = project_id
        
        # Configure model parameters for technical precision
        self.params = {
            GenParams.DECODING_METHOD: "greedy", # High precision for engineering tasks
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.TEMPERATURE: 0.1,
            GenParams.STOP_SEQUENCES: ["\n\n"]
        }
        
        # Use IBM Granite 13b Instruct Model
        self.model_id = "ibm/granite-13b-instruct-v2"
        self.model = Model(
            model_id=self.model_id,
            params=self.params,
            credentials=self.credentials,
            project_id=self.project_id
        )

    def get_expert_advice(self, user_query):
        """
        System prompt to guide Granite to act as an Electronics & Telecom Engineer.
        """
        system_prompt = (
            "You are an expert Electronics and Telecommunications Engineer specializing in "
            "Audio Signal Processing. Your task is to help debug circuits like preamps, "
            "filters, and amplifiers. Provide detailed engineering advice on grounding tips, "
            "noise elimination, and component selection. Use technical terms like THD, SNR, "
            "and Op-Amp Biasing where appropriate.\n\n"
            f"User Query: {user_query}\n"
            "Expert Advice:"
        )
        
        response = self.model.generate_text(prompt=system_prompt)
        return response

# --- Example Usage ---
if __name__ == "__main__":
    # Get these from your IBM Cloud Account
    API_KEY = "YOUR_IBM_CLOUD_API_KEY"
    PROJECT_ID = "YOUR_WATSONX_PROJECT_ID"
    
    agent = AudioSignalAgent(API_KEY, PROJECT_ID)
    
    # Test Query 1: Troubleshooting Noise
    query_1 = "Why is my audio amplifier producing a 60Hz hum even when no input is connected?"
    print(f"Query: {query_1}\nResponse: {agent.get_expert_advice(query_1)}\n")
    
    # Test Query 2: Filter Design
    query_2 = "How do I design a 2nd order Butterworth low-pass filter with a cutoff frequency of 1kHz?"
    print(f"Query: {query_2}\nResponse: {agent.get_expert_advice(query_2)}\n")
