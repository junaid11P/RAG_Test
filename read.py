with open("story.txt", "r", encoding="utf-8") as f:
    content = f.read()

print("--- File Read Check ---")
print(f"Total characters read: {len(content)}")
print(f"First 100 chars: {content[:100]}...")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_text(content)

print(f"\n--- Splitter Check ---")
print(f"Total chunks created: {len(chunks)}")
print(f" chunck 1\n{chunks[0]}")
print(f" chunck 2\n{chunks[1]}")
print(f" chunck 3\n{chunks[2]}")
print(f" chunck 4\n{chunks[3]}")
print(f" chunck 5\n{chunks[4]}")
print(f" chunck 6\n{chunks[5]}")
print(f" chunck 7\n{chunks[6]}")
print(f" chunck 8\n{chunks[7]}")
print(f" chunck 9\n{chunks[8]}")
# print(f" chunck 10\n{chunks[9]}")

