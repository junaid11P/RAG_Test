def manual_chunk_text(text, chunk_size=500, chunk_overlap=50):
    words = text.split(' ')

    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_len = len(word) + 1

        if current_length + word_len > chunk_size:
            chunks.append(" ".join(current_chunk))

            overlap_buffer = []
            overlap_len = 0

            for w in reversed(current_chunk):
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
