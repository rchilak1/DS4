# app.py
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
import requests
import random

# Ensure 'punkt' is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Function to load and preprocess the text
@st.cache_data
def load_data():
    # Download the text from Project Gutenberg
    url = 'https://www.gutenberg.org/files/84/84-0.txt'
    response = requests.get(url)
    frankenstein_text = response.text

    # Remove Project Gutenberg header and footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN ***"
    start_index = frankenstein_text.find(start_marker) + len(start_marker)
    end_index = frankenstein_text.find(end_marker)
    clean_text = frankenstein_text[start_index:end_index].strip()

    # Tokenize the text into sentences
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(clean_text)

    # Group sentences into chunks
    chunk_size = 3  # Reduced chunk size for finer granularity
    chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

    return clean_text, chunks  # Return clean_text in addition to chunks

# Function to create embeddings and build the FAISS index
@st.cache_resource
def create_index(chunks):
    # Load the embedding model optimized for QA tasks
    embedder = SentenceTransformer('all-mpnet-base-v2')

    # Create embeddings
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    chunk_embeddings_np = chunk_embeddings.cpu().detach().numpy()

    # Build the FAISS index
    embedding_dim = chunk_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(chunk_embeddings_np)

    return embedder, index

# Function to answer questions
def answer_question(question, embedder, index, chunks, top_k=10):
    # Embed the user's question
    question_embedding = embedder.encode([question], convert_to_tensor=True).cpu().detach().numpy()

    # Retrieve the most relevant chunks
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]

    # Initialize the QA pipeline with a more powerful model
    qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

    # Collect answers from each relevant chunk
    answers = []
    for context in relevant_chunks:
        try:
            result = qa_pipeline({'question': question, 'context': context})
            answers.append((result['score'], result['answer']))
        except Exception as e:
            continue  # Skip contexts that are too long or cause errors

    # Return the best answer
    if answers:
        best_answer = max(answers, key=lambda x: x[0])[1]
        return best_answer
    else:
        return "I'm sorry, I couldn't find an answer to your question."

# New function: Load GPT-2 model and tokenizer (cached)
@st.cache_resource
def load_gpt2_model():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    return model, tokenizer

# Modified function: Text generation with perplexity calculation
def generate_text(seed_text, chunks, model, tokenizer, max_length=100):
    import torch

    # Sample a random chunk from the book as additional context
    context_chunk = random.choice(chunks)
    prompt_text = f"{context_chunk}\n\n{seed_text}"

    # Encode the combined prompt (context + seed text)
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

    # Generate text
    output_sequences = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),  # Include initial prompt length
        num_return_sequences=1,  # Generate multiple sequences
        no_repeat_ngram_size=2,  # Reduced to allow some repetition
        early_stopping=True,
        temperature=0.9,
        top_p=0.95,
    )

    # Function to calculate perplexity
    def calculate_perplexity(model, tokenizer, text):
        encodings = tokenizer(text, return_tensors='pt')
        max_length = model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = i + stride
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            lls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    # Decode and collect generated texts and their perplexities
    generated_texts = []
    perplexities = []
    for output in output_sequences:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_text_without_context = generated_text[len(context_chunk):].strip()  # Remove the context part
        perplexity = calculate_perplexity(model, tokenizer, generated_text_without_context)
        generated_texts.append(generated_text_without_context)
        perplexities.append(perplexity)

    # Select the generated text with the lowest perplexity
    best_index = perplexities.index(min(perplexities))
    best_generated_text = generated_texts[best_index]
    best_perplexity = perplexities[best_index]

    return best_generated_text, best_perplexity

# Streamlit App
def main():
    st.title("Frankenstein App")
    st.write("Ask questions about Mary Shelley's *Frankenstein* or generate text based on a seed.")

    # Load data and models
    with st.spinner("Loading data and building index..."):
        clean_text, chunks = load_data()
        embedder, index = create_index(chunks)
        model, tokenizer = load_gpt2_model()
    st.success("Setup complete!")

    # Create tabs for QA and Text Generation
    tab1, tab2 = st.tabs(["Question Answering", "Text Generation"])

    with tab1:
        st.header("Question Answering")
        # User input
        question = st.text_input("Your Question:", key='question_input')
        if st.button("Get Answer", key='qa_button') and question:
            with st.spinner("Searching for the answer..."):
                answer = answer_question(question, embedder, index, chunks)
            st.write("**Answer:**", answer)

    with tab2:
        st.header("Text Generation")
        # User input for text generation
        seed_text = st.text_input("Enter a seed text from the book:", key='seed_input')
        max_length = st.slider("Select the maximum length of generated text:", min_value=50, max_value=500, value=100)
        if st.button("Generate Text") and seed_text:
            with st.spinner("Generating text..."):
                generated_text, perplexity = generate_text(seed_text, chunks, model, tokenizer, max_length=max_length)
            st.write("**Generated Text:**")
            st.write(generated_text)
            st.write(f"**Perplexity of Generated Text:** {perplexity:.2f}")

if __name__ == '__main__':
    main()
