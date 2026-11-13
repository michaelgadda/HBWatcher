import psycopg2 as psy
from typing import Any
import os

#TODO: Need to add actual error handling and loggings

def db_query(query: str, params: list[str | int]) -> tuple[Any, Any]:
    db_user = os.getenv("DB_USER")
    db_pwd = os.getenv("DB_PWD")
    db_name = os.getenv("DB_NAME")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")

    conn = f"dbname={db_name} user={db_user} password={db_pwd} host={db_host} port={db_port}"
    connection = psy.connect(conn) 
    cur = connection.cursor()
    
    cur.execute(query, params)

    if cur.description is not None:
        return cur.fetchall()

    conn.commit()

    return None
