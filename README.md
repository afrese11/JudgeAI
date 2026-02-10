# JudgeAI

.env file:
include a .env file that contains OPENAI_API_KEY = [INSERT KEY HERE] in the same directory as backend, frontend, etc.

This Repo consists of three components:

1. Blackbox Model (GPT 5 Nano)
  This model uploads case documents to GPT and prompts it to predict case outcomes and other variables.

To run: cd backend_model. Then, run >python blackbox_predictor_gpt5.py --redact. redact is an optional variable to redact the documents.
