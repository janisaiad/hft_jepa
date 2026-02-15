import pytest
import os
from dotenv import load_dotenv


def test_import_env():
    load_dotenv()
    assert os.getenv("FOLDER_PATH") is not None


def test_import_library():
    import hft_jepa as hft
    assert hft is not None
    
if __name__ == "__main__":
    test_import_env()
    