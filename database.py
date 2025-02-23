import logging
from supabase import create_client, Client
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str, user_id: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.user_id = user_id  # Store the user_id for schema creation
        self.schema_name = f"user_{user_id}"  # Dynamically generate the schema name based on user_id
        # Initialize the Supabase client
        self.supabase = create_client(self.supabase_url, self.supabase_key)

    def create_user_schema(self):
        """Creates a user-specific schema if it doesn't exist."""
        try:
            # SQL query to create a schema for the user
            create_schema_sql = f"""
            CREATE SCHEMA IF NOT EXISTS {self.schema_name};
            """
            # Execute the query
            response = self.supabase.rpc('sql', {'query': create_schema_sql}).execute()
            return response
        except Exception as e:
            print(f"Error creating schema for user {self.user_id}: {e}")
            return None

    def get_conversation_history(self):
        """Retrieves conversation history from the user's schema."""
        try:
            # SQL query to retrieve conversation history from the user's schema
            query = f"SELECT * FROM {self.schema_name}.conversation_history ORDER BY timestamp DESC"
            response = self.supabase.rpc('sql', {'query': query}).execute()
            return response['data']  # Assuming response contains 'data'
        except Exception as e:
            print(f"Error fetching conversation history: {e}")
            return []

    def add_interaction(self, data_point):
        """Adds interaction data to the user's schema."""
        try:
            query = f"""
            INSERT INTO {self.schema_name}.interactions (context, question, answer, metadata)
            VALUES ('{data_point['context']}', '{data_point['question']}', '{data_point['answer']}', '{json.dumps(data_point['metadata'])}');
            """
            response = self.supabase.rpc('sql', {'query': query}).execute()
            return response
        except Exception as e:
            print(f"Error adding interaction for user {self.user_id}: {e}")
            return None
    
    def get_interaction_history(self, user_id: str):
        """ Get interaction history from the user's schema """
        schema_name = f'user_{user_id}'
        logger.info(f"Retrieving interaction history for user: {user_id} with schema {schema_name}")
        
        # Call SQL function to retrieve interaction history
        sql_query = f"SELECT * FROM get_interaction_history('{schema_name}')"
        response = self.supabase.rpc('sql', {'command': sql_query}).execute()
        
        if response.data:
            history = response.model_dump_json()
            logger.info(f"Retrieved interaction history for user {user_id}: {history}")
            return history
        else:
            logger.error(f"Error retrieving interaction history for user {user_id}: {response.json()}")
            return None
    
    def ensure_user_schema_view(self, user_id: str):
        """ Ensure the view for the user schema exists in the public schema """
        schema_name = f'user_{user_id}'
        logger.info(f"Ensuring view exists for user: {user_id} with schema {schema_name}")
        
        # Call SQL function to ensure the view exists
        sql_query = f"SELECT ensure_user_schema_view('{schema_name}')"
        response = self.supabase.rpc('sql', {'command': sql_query}).execute()

        if response.data:
            logger.info(f"View for user {user_id} confirmed.")
        else:
            logger.error(f"Error confirming view for user {user_id}: {response.json()}")
