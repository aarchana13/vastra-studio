```markdown
# VastraStudio

## Local Development Setup

To run VastraStudio on your own computer, follow these steps:

### 1. Prerequisites
* **Python 3.8+** installed on your machine.
* An API Key from **[Runware](https://runware.ai/)**.
* An API Key from **[Google AI Studio (Gemini)](https://aistudio.google.com/)**.

### 2. Configure Environment Variables
Create a file named `.env` in the root of your project folder (in the same folder as `main.py`) and add your API keys:
```env
GEMINI_API_KEY=your_gemini_key_here
RUNWARE_API_KEY=your_runware_key_here

```

### 3. Install Dependencies

Open your terminal, navigate to your project folder, and run:

```bash
pip install -r requirements.txt

```

### 4. Start the Server

Run the FastAPI development server using Uvicorn:

```bash
uvicorn main:app --reload

```

Once the server boots up, open your web browser and navigate to `http://localhost:8000` to view and use the application.

```

```
