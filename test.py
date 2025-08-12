import requests

response = requests.post("http://127.0.0.1:8000/api/chat", json={"question": "What is Breast Cancer?"})
print(response.json())
