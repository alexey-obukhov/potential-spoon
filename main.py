import os
from dotenv import load_dotenv
import torch
import multiprocessing as mp
from flask import Flask, request, jsonify, g
import asyncio

from rag_processor import RAGProcessor
from database import DatabaseManager
from text_generator import TextGenerator

# --- Disable tokenizer parallelism ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    load_dotenv()

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
        exit()

    # --- Use GPU if available ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize TextGenerator (only initialize once)
    generator = TextGenerator("microsoft/phi-1_5", device, use_bfloat16=False)

    print("Welcome to the Therapy AI Assistant!")

    def before_request():
        """Initialize DatabaseManager and RAGProcessor before each request."""
        user_id = request.headers.get('X-User-ID')
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        # Store user_id in Flask's 'g' object
        g.user_id = user_id

        # Initialize DatabaseManager and store in 'g'
        g.db_manager = DatabaseManager(supabase_url, supabase_key, g.user_id)

        # Create user schema for the current user (asynchronously)
        schema_creation_result = g.db_manager.create_user_schema()

        # Check if schema creation was successful
        if not schema_creation_result:
            return jsonify({'error': 'Failed to create user schema'}), 500

        # Initialize RAGProcessor and store in 'g'
        g.rag_processor = RAGProcessor(g.db_manager, generator)

    app.before_request(before_request)  # Register the function

    @app.route('/chat', methods=['POST'])
    def chat():
        """Handles chat requests."""
        data = request.get_json()
        user_question = data.get('question')
        question_id = data.get('questionID')  # Get questionID from request

        if not user_question:
            return jsonify({'error': 'Missing question'}), 400

        # Access rag_processor from Flask's 'g' object
        response = g.rag_processor.generate_response(user_question, device, question_id)  # Pass questionID
        return jsonify({'response': response})

    @app.route('/add_document', methods=['POST'])
    def add_document():
        """Handles document addition requests."""
        data = request.get_json()
        content = data.get('content')

        if not content:
            return jsonify({'error': 'Missing content'}), 400

        # Generate embedding and store the document
        embedding = generator.get_embedding(content)
        if embedding is not None:
            g.db_manager.add_document_to_knowledge_base(content, embedding.tolist())
            return jsonify({'message': 'Document added successfully'})
        else:
            return jsonify({'error': 'Failed to generate embedding'}), 500

    @app.route('/get_documents', methods=['GET'])
    def get_documents():
        """Retrieves all documents for the authenticated user."""
        documents = g.db_manager.get_all_documents_and_embeddings()
        return jsonify({'documents': documents})

    app.run(debug=False, port=5003)
