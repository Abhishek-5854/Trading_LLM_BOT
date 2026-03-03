"""Simple config loader using python-dotenv."""
from dotenv import load_dotenv
import os

load_dotenv()


def get_env(key: str, default=None):
    return os.getenv(key, default)
