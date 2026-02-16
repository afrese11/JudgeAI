from dotenv import load_dotenv
import os
load_dotenv()


API_KEY = os.getenv("FINN_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")