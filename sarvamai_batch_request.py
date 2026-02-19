import requests
import json
import os,dotenv

dotenv.load_dotenv()
api_key = os.environ["SARVAM"]
API_URL = "https://api.sarvam.ai/v1/process"

input_data  = [
    {"text": "Hello, how are you"},
    {"text": "what are the large language models . train a large language model"},
]

headers = {
    "Authorization" : f"Bearer {api_key}",
    "Content-Type" : "application/json"
}

outputs = []

for item in input_data:
    response = requests.post(
        url=API_URL, headers=headers,json=item) 
    if response.status_code == 200:
        outputs.append(response.json)


output_path = "C:/Users/Gopinath/Finetuning_learning/outputs.md"

with open(output_path , "w", encoding="utf-8")as f:
    for output in outputs:
        f.write(json.dumps(output, indent=2) + '\n\n')


print(f"ouput")