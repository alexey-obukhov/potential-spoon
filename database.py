""" 2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com """
import logging
from supabase import create_client, Client
from typing import List, Dict
import json
import re
import traceback
from utilities.text_utils import clean_text

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
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        logger.debug(f"DatabaseManager initialized for user: {self.user_id}")

    def create_user_schema(self):
        """Creates a user-specific schema and tables if they don't exist."""
        try:
            # Call the stored procedure to create the schema and tables
            response = self.supabase.rpc('create_user_schema_and_tables', {'schema_name': self.schema_name}).execute()
            if response.data:
                logger.error(f"Error creating schema for user {self.user_id}: {response.data}")
                return False

            logger.info(f"Schema '{self.schema_name}' and tables created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error creating schema for user {self.user_id}: {e}")
            return False

    def create_user_schema_sync(self):
        """Creates a user-specific schema and tables if they don't exist (synchronous version)."""
        try:
            # Call the stored procedure to create the schema and tables
            response = self.supabase.rpc('create_user_schema_and_tables', {'schema_name': self.schema_name}).execute()
            if response.data:
                logger.error(f"Error creating schema for user {self.user_id}: {response.data}")
                return False

            logger.info(f"Schema '{self.schema_name}' and tables created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error creating schema for user {self.user_id}: {e}")
            return False

    def get_conversation_history(self):
        """Retrieves conversation history from the user's schema."""
        try:
            # Call the stored procedure to retrieve conversation history
            response = self.supabase.rpc('get_conversation_history', {'schema_name': self.schema_name}).execute()

            # Check if the response contains data
            if response.data:
                # Transform the data to have the expected field names
                transformed_data = []
                for item in response.data:
                    transformed_item = {
                        'interactionID': item.get('interactionid'),
                        'questionText': item.get('question'),  # in db it is question
                        'answerText': item.get('answer'),      # in db it is answer
                        'context': item.get('context'),
                        'metadata': item.get('metadata'),
                        'created_at': item.get('created_at')
                    }
                    transformed_data.append(transformed_item)
                return transformed_data
            else:
                logger.warning("No conversation history found.")
                return []
        except Exception as e:
            # Log the error with additional context
            logger.error(f"Error fetching conversation history: {e}")
            return []

    def add_interaction(self, data_point):
        """Adds interaction data to the user's schema."""
        try:
            # Clean and escape the values
            context = clean_text(data_point['context'])
            question = clean_text(data_point['question'])
            answer = clean_text(data_point['answer'])

            # Add RETURNING clause to the INSERT statement to ensure it returns data
            query = f"""
            INSERT INTO {self.schema_name}.interactions (context, question, answer, metadata)
            VALUES ('{context}', '{question}', '{answer}', '{json.dumps(data_point['metadata'])}')
            RETURNING interactionID;
            """
            response = self.supabase.rpc('sql', {'command': query}).execute()

            if response.data:
                # Now response.data contains the returned interactionID
                logger.info(f"Added interaction with ID: {response.data} for user {self.user_id}")
                return response
            else:
                logger.error(f"No data returned when adding interaction for user {self.user_id}")
                return None
        except Exception as e:
            logger.error(f"Error adding interaction for user {self.user_id}: {e}")
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

    def _sanitize_schema_name(self, user_id: str) -> str:
        """Sanitizes the user ID to be a valid PostgreSQL schema name (private method)."""
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        if not (safe_name[0].isalpha() or safe_name[0] == '_'):
            safe_name = "_" + safe_name
        return safe_name[:63]

    def _get_table_name(self, table_name: str) -> str:
        """
        Returns the fully qualified table name with the correct schema.
        Uses the provided `user_id` or falls back to `self.user_id`.
        """
        schema = self._sanitize_schema_name(self.user_id)
        return f'"{schema}"."{table_name}"'

    def get_all_documents_and_embeddings(self, table_name: str = "knowledge_base") -> List[Dict]:
        """Retrieves all documents and their embeddings from the knowledge base."""
        try:
            # Use the dedicated function instead of the generic SQL function
            response = self.supabase.rpc('get_knowledge_base_documents', 
                                        {'schema_name': self.schema_name}).execute()

            if response.data:
                logger.info(f"Retrieved {len(response.data)} documents from knowledge base")
                return response.data
            else:
                logger.warning("No documents found in knowledge base")
                return []
        except Exception as e:
            logger.error(f"Error retrieving documents and embeddings: {e}")
            traceback.print_exc()  
            return []
