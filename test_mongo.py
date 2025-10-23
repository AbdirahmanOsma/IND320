from pymongo import MongoClient

url = "mongodb+srv://ind320_user:Calanka12@ind320.762ezjs.mongodb.net/?retryWrites=true&w=majority&appName=ind320"
client = MongoClient(url)

try:
    client.admin.command("ping")
    print("✅ Du er koblet til MongoDB Atlas!")
except Exception as e:
    print(" Feil:", e)
finally:
    client.close()