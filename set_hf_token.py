from google.colab import userdata
import os

# Fetch the token from Colab Secrets
hf_token = userdata.get('HF_TOKEN')

# Optional: Set it as an environment variable for custom scripts
os.environ["HF_TOKEN"] = hf_token 