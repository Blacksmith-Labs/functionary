import requests
data = {"messages": [{"role": "assistant"}]}
result = requests.post("https://elyasmehtabuddin--functionary-fastapi-app-dev.modal.run/v1/chat/completions", json=data, timeout=1000.0)
print(result)
print(type(result))
print(result.text)