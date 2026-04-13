from sqlalchemy import create_engine, MetaData, Table, select
from typing import List, Dict, Any

class SQLAdapter:
    def __init__(self, source_db_url: str):
        """
        Connects to the user's existing database.
        Example URLs:
        - MySQL: "mysql+pymysql://user:pass@localhost/dbname"
        - Postgres: "postgresql+psycopg://user:pass@localhost/dbname"
        """
        self.engine = create_engine(source_db_url)
        self.metadata = MetaData()

    def fetch_data(self, table_name: str, content_column: str, metadata_columns: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches rows from a specific table and separates the main text from the metadata.
        """
        # Automatically read the structure of the user's table
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        
        # Select only the columns we actually need to save memory
        columns_to_fetch = [getattr(table.c, content_column)] + [getattr(table.c, col) for col in metadata_columns]
        query = select(*columns_to_fetch)
        
        extracted_documents = []

        # Execute the query and format the results
        with self.engine.connect() as conn:
            result = conn.execute(query)
            
            for row in result:
                # Convert the row to a dictionary
                row_dict = dict(row._mapping)
                
                # Pop out the main content text
                text_content = row_dict.pop(content_column)
                
                # The remaining items in row_dict are our metadata
                extracted_documents.append({
                    "text": text_content,
                    "metadata": row_dict
                })
                
        return extracted_documents