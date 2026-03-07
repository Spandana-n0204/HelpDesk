from llm_generator import generate_response

# Sample context (pretend this came from FAISS retrieval)
context = """
Dayananda Sagar College of Engineering (DSCE), Bangalore
provides hostel facilities for both boys and girls.
The hostel includes mess facilities, WiFi, and recreation areas.
"""

# Example question
question = "Does DSCE provide hostel facilities?"

# Generate answer using Ollama
answer = generate_response(context, question)

print("\nGenerated Answer:\n")
print(answer)