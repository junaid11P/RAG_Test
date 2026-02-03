with open("story.txt", "r", encoding="utf-8") as f:
    text = f.read()

if __name__ == "__main__":
    print("--- File Read Check ---")
    print(f"Total characters read: {len(text)}")
    print(f"First 100 chars: {text[:100]}...")