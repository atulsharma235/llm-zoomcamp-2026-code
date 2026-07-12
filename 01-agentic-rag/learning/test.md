# Learning in Public: Building a RAG Bot for LLM Zoomcamp

As part of my journey through the DataTalks.Club LLM Zoomcamp, I'm practicing the "learning in public" approach. Instead of keeping my notes tucked away in a local folder, I'm sharing my execution steps, code snippets, and mental models as I build out our running project: a question-answering bot tailored specifically for the course itself. Here is a breakdown of my first hands-on steps with LLMs and the core mechanics of Retrieval-Augmented Generation (RAG).

```python
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
openai_client = OpenAI()
import os

# Reuse the existing client if it was already created in another cell
if "openai_client" not in globals():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_client

```
