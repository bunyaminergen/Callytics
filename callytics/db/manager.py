# Standard library imports
import sqlite3
from typing import Annotated, List, Tuple, Optional


class Database:
    """
    A class to interact with an SQLite database.

    This class provides methods to fetch data, insert data, and handle specific
    tasks like fetching or inserting topic IDs in a database.

    Parameters
    ----------
    db_path : str
        The path to the SQLite database file.

    Attributes
    ----------
    db_path : str
        The path to the SQLite database file.
    """

    def __init__(self, db_path: Annotated[str, "Path to the SQLite database"]):
        """
        Initializes the Database class with the provided database path.

        Parameters
        ----------
        db_path : str
            The path to the SQLite database file.
        """
        self.db_path = db_path

    def fetch(
            self,
            sql_file_path: Annotated[str, "Path to the SQL file"]
    ) -> Annotated[List[Tuple], "Results fetched from the query"]:
        """
        Executes a SELECT query from an SQL file and fetches the results.

        Parameters
        ----------
        sql_file_path : str
            Path to the SQL file containing the SELECT query.

        Returns
        -------
        List[Tuple]
            A list of tuples representing rows returned by the query.

        Examples
        --------
        >>> db = Database("example.db")
        >>> result = db.fetch("select_query.sql")
        >>> print(results)
        [(1, 'data1'), (2, 'data2')]
        """
        with open(sql_file_path, encoding='utf-8') as f:
            query = f.read()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        return results

    def insert(
            self,
            sql_file_path: Annotated[str, "Path to the SQL file"],
            params: Optional[Annotated[Tuple, "Query parameters"]] = None
    ) -> Annotated[int, "ID of the last inserted row"]:
        """
        Executes an INSERT query from an SQL file and returns the last row ID.

        Parameters
        ----------
        sql_file_path : str
            Path to the SQL file containing the INSERT query.
        params : tuple, optional
            Parameters for the query. Defaults to None.

        Returns
        -------
        int
            The ID of the last inserted row.

        Examples
        --------
        >>> db = Database("example.db")
        >>> last_id_ = db.insert("insert_query.sql", ("value1", "value2"))
        >>> print(last_id)
        3
        """
        with open(sql_file_path, encoding='utf-8') as f:
            query = f.read()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if params is not None:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    def get_or_insert_topic_id(
            self,
            detected_topic: Annotated[str, "Topic to detect or insert"],
            topics: Annotated[List[Tuple], "Existing topics with IDs"],
            db_topic_insert_path: Annotated[str, "Path to the SQL file for inserting topics"]
    ) -> Annotated[int, "Topic ID"]:
        """
        Fetches an existing topic ID or inserts a new one and returns its ID.

        Parameters
        ----------
        detected_topic : str
            The topic to be detected or inserted.
        topics : List[Tuple[int, str]]
            A list of existing topics as (id, name) tuples.
        db_topic_insert_path : str
            Path to the SQL file for inserting a new topic.

        Returns
        -------
        int
            The ID of the detected or newly inserted topic.

        Examples
        --------
        >>> db = Database("example.db")
        >>> topics_ = [(1, 'Python'), (2, 'SQL')]
        >>> topic_id_ = db.get_or_insert_topic_id("AI", topics, "insert_topic.sql")
        >>> print(topic_id)
        3
        """
        detected_topic_lower = detected_topic.lower()
        topic_map = {t[1].lower(): t[0] for t in topics}

        if detected_topic_lower in topic_map:
            return topic_map[detected_topic_lower]
        else:
            topic_id = self.insert(db_topic_insert_path, (detected_topic,))
            return topic_id
