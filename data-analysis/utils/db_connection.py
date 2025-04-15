import psycopg
from contextlib import contextmanager

DB_URL = "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"


@contextmanager
def get_connection():
    conn = None
    try:
        conn = psycopg.connect(DB_URL)
        yield conn
    finally:
        if conn is not None:
            conn.close()


@contextmanager
def get_cursor(commit=False):
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        finally:
            cursor.close()
