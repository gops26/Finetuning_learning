from sarvamai import SarvamAI
from dotenv import load_dotenv
import os

load_dotenv()
client = SarvamAI(
    api_subscription_key = os.environ["SARVAM"]
)
client.document_intelligence.initialise()

response = client.chat.completions(
    messages = [
        {"role":"user", "content":"how to batch process with sarvam ai ? save the output to a filepath ``C:/Users/Gopinath/Documents/outputs.md``"}
    ]
)

print(response.choices[0].message.content)