# LLM Observability with FastHTML, Google Gemini, and Langfuse

This project provides a chatbot demo using FastHTML, Google Gemini (GenAI), and Langfuse for observability and feedback. It includes instructions to launch Langfuse using Docker Compose and to run the chat app locally.

Refer to the associated blog posts for detailed explanations:
- [LLM Observability with Langfuse and a FastHTML Chatbot - Part 1](https://nbrosse.github.io/posts/llm-observability-part1/)

## Installation and Setup Guide

- **Docker** and **Docker Compose** installed ([Install Docker](https://docs.docker.com/get-docker/))
- [uv](https://github.com/astral-sh/uv) (fast Python package installer)  


**Launch Langfuse with Docker Compose**

Get a copy of the latest Langfuse repository:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
```

Update the secrets in the `docker-compose.yml` (for demo, defaults are fine) and start langfuse:

```bash
docker compose up -d
```

Langfuse UI will be available at [http://localhost:3000](http://localhost:3000).

**Set Up Environment Variables**

Create a `.env` file in your project root with the following content:

```env
GOOGLE_API_KEY=your_google_api_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=http://localhost:3000
```

- Get your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- Get Langfuse keys from the Langfuse UI under "Project Settings" > "API Keys".

**Install Python Dependencies**

From your project root:

```bash
uv sync
```

## Run the Chat App

From the project root:

```bash
uv run app/chat_fasthtml.py
```

This will start the FastHTML chat app, which uses Google Gemini for responses and Langfuse for observability.

Open your browser and go to the address shown in your terminal (by default [http://localhost:5001](http://localhost:5001)).

## Troubleshooting

- Ensure Docker containers for Langfuse are running:  
  ```bash
  docker compose ps
  ```
- Check `.env` variables are correct and exported.
- For Google API errors, verify your API key and model name.
- For Langfuse errors, ensure the service is running and keys/host are correct.

## Useful Links

- [Langfuse Documentation](https://langfuse.com/docs)
- [Google Generative AI Python SDK](https://github.com/googleapis/python-genai)
- [FastHTML](https://www.fastht.ml/docs)


