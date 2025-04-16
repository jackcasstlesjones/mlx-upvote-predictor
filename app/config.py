import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()

default_config = {
    "DB_HOST": "localhost",
    "DB_PORT": 5432,
    "DB_USER": "user",
    "DB_PASSWORD": "password",
    "DB_NAME": "dbname"
}

def get_config():
    config = default_config.copy()
    # Override with environment variables if present
    config["DB_HOST"] = os.getenv("DB_HOST", config["DB_HOST"])
    config["DB_PORT"] = int(os.getenv("DB_PORT", config["DB_PORT"]))
    config["DB_USER"] = os.getenv("DB_USER", config["DB_USER"])
    config["DB_PASSWORD"] = os.getenv("DB_PASSWORD", config["DB_PASSWORD"])
    config["DB_NAME"] = os.getenv("DB_NAME", config["DB_NAME"])
    return config
