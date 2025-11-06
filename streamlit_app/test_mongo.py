from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Test med nytt passord
uri = "mongodb+srv://ind320_user:calanka12@ind320.762ezjs.mongodb.net/?retryWrites=true&w=majority&appName=ind320"

client = MongoClient(uri, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("✅ Password WORKS locally!")
except Exception as e:
    print(f"❌ Local test failed: {e}")