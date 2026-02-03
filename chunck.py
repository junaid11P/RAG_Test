import read
import numpy as np
from sentence_transformers import SentenceTransformer

def manual_chunk_text(text, chunk_size=500, chunk_overlap=50):
    words = text.split(' ')
    
    chunks = [] # chunks is the list of chunks
    current_chunk = []# current_chunk is the current chunk of words
    current_length = 0 # current_length is the length of current_chunk
    
    for word in words:
        word_len = len(word) + 1 # word_len is the length of word + 1 for space
        if current_length + word_len > chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            overlap_buffer = [] # overlap_buffer is the list of words to be added to the next chunk
            overlap_len = 0 # overlap_len is the length of overlap_buffer
            for w in reversed(current_chunk): # w is one word from current_chunk
                if overlap_len + len(w) + 1 <= chunk_overlap:
                    overlap_buffer.insert(0, w)
                    overlap_len += len(w) + 1
                else:
                    break
            
            current_chunk = overlap_buffer + [word]
            current_length = overlap_len + word_len
            
        else:
            current_chunk.append(word)
            current_length += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

if __name__ == "__main__":
    try:
        # create chunks
        my_chunks = manual_chunk_text(read.text, chunk_size=500, chunk_overlap=50)
        print(f"Total Chunks: {len(my_chunks)}\n")

        for i, chunk in enumerate(my_chunks):
            print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
            print(chunk)

        # create model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Convert chunks to vectors
        embeddings = model.encode(my_chunks)

        # Check results
        print("--- Embedding Check ---")
        print(f"Embedding vector length: {len(embeddings[0])}")
        print(f"First embedding (sample):\n{embeddings[0][:10]}")

        # save embeddings
        np.save("embeddings.npy", embeddings)
        print("Embeddings saved successfully")

        # load embeddings
        loaded_embeddings = np.load("embeddings.npy")
        print(loaded_embeddings)
        print(loaded_embeddings.shape)
        print("Embeddings loaded successfully")

    except FileNotFoundError:
        print("Please make sure 'story.txt' exists.")
    except AttributeError:
        print("Error: Could not find 'text' in 'read' module.")