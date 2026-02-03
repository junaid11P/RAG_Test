from pymilvus import connections

try:
    connections.connect(alias="default", host="localhost", port="19530")
    print("✅ Successfully connected to Milvus!")
    connections.disconnect("default")
except Exception as e:
    print(f"❌ Connection failed: {e}")