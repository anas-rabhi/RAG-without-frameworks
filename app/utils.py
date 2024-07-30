from openai import OpenAI

client = OpenAI()


def get_embedding(text):
    """
    Call OpenAI API to create embeddings for a given text.
    """
    # Call OpenAI API to generate embedding

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def query_chroma(query_embedding, collection, n_results=5):
    """
    Query ChromaDB for relevant documents using the query embedding.
    """
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    # Return the first list of documents
    return results['documents'][0]

def generate_answer(messages, context, system_prompt):
    """
    Generate an answer using OpenAI's API with the given context.
    """
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt.format(context = context)},
        ] + messages,
        # stream=True
    )
    # Extract and return the generated answer
    return response.choices[0].message.content

def rag_pipeline(question, collection, system_prompt, messages):
    """
    Executes the RAG pipeline.
    """
    # Generate embedding for the question
    question_embeddings = get_embedding(question)

    # Retrieve relevant chunks from ChromaDB
    relevant_chunks = query_chroma(question_embeddings, collection)

    # Combine chunks into a single context string
    context = "\n\n".join(relevant_chunks)

    # Generate answer using the question and context
    answer = generate_answer(messages, context, system_prompt)

    return answer