# mapping_store.py
from tinydb import TinyDB, Query
from pathlib import Path

# Store file
DB_PATH = Path(__file__).parent / "mapping_db.json"
db = TinyDB(DB_PATH)
Mapping = Query()

def get_mapping(instrument: str) -> dict | None:
    """Return mapping if exists for given instrument."""
    result = db.search(Mapping.instrument == instrument.lower())
    return result[0]["mapping"] if result else None

def save_mapping(instrument: str, mapping: dict) -> None:
    """Save or overwrite mapping for given instrument."""
    # remove existing first
    db.remove(Mapping.instrument == instrument.lower())
    db.insert({"instrument": instrument.lower(), "mapping": mapping})
