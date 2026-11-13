from src.db.util_db import db_query

def get_curr_classes() -> list[str]:
    query = """
            SELECT class
            FROM clsf_type
            """
    rows = db_query(query, [])
    clss = []
    for row in rows:
        clss.append(row[0])
    return clss

def insert_new_clsf_type(clsf_type:str):
    query = """INSERT INTO clsf_type (class)
               VALUES(%s)"""

    db_query(query, [clsf_type])

def insert_detection(clsf_class: str, image_path: str):
    # Make sure the class is in clsf_type table 
    
    query = """
            INSERT into detection (class, image_path)
            VALUES ( %s, %s )
            """
    
    db_query(query, [clsf_class, image_path])