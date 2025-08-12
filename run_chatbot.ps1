# run_chatbot.ps1

# Step 1: Activate virtual environment
D:\GenAi_Project\gemini-env\Scripts\Activate.ps1

# Step 2: Start backend (FastAPI)
Start-Process powershell -ArgumentList 'cd D:\breasthealth\breastchat\backend; uvicorn app:app --reload --host 127.0.0.1 --port 8000'

# Step 3: Wait a few seconds
Start-Sleep -Seconds 7

# Step 4: Start frontend (Next.js)
Start-Process powershell -ArgumentList 'cd D:\breasthealth\breastchat-ui; npm run dev; pause'
