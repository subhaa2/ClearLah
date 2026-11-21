# ClearLah

ClearLah explains letters in plain language and reads the explanation and action steps aloud in English, Tamil, Malay, or Mandarin.

- Gemini extracts the sender, amount due, deadline, and actions from a photo or PDF.
- Cloud Vision transcribes JPG and PNG uploads so readers can compare the result with the original.
- Cloud Text-to-Speech reads the explanation and actions aloud.

## Run locally

Create and activate a Python virtual environment, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a local `.env` file beside `app.py` and add `GEMINI_API_KEY=your_key_here`. Git ignores this file. A key already set in the terminal takes precedence. Enable Cloud Vision and Cloud Text-to-Speech in a Google Cloud project and set up Application Default Credentials for that project:

```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
python -m streamlit run app.py
```

Use a fictional letter for testing. When you use Explain letter or View original text, the app sends the document to the corresponding Google service. Listen sends the explanation and action text to Cloud Text-to-Speech.

## Hosting

Use a Python host that runs Streamlit, with `app.py` as the entry point. Set the Gemini key through the host's secrets settings. Cloud Vision and Text-to-Speech also need Google Cloud credentials available to the running app. Local gcloud sign-in applies only to your machine.
