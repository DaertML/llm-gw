OST http://localhost:8000/v1/ingest   -H "Content-Type: application/json"   -d '{
    "texts": [
      "LiteLLM is an open-source proxy that unifies multiple LLM providers.",
      "Qdrant is a high-performance vector database written in Rust."
    ],
    "metadatas": [
      {"source": "docs", "topic": "litellm"},
      {"source": "docs", "topic": "qdrant"}
    ]
  }'


curl -s http://localhost:8000/v1/ingest | head  # sanity check API is up

curl -s -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model":"rag","messages":[{"role":"user","content":"What is LiteLLM?"}],"stream":false}' | python3 -m json.tool

docker-compose logs --tail=50 rag-api


