""" 2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com """
from text_generator import TextGenerator
from database import DatabaseManager
import torch
from typing import List
from utilities.therapeutic_promt import prompt_templates
import json
import numpy as np
from flask import g
from safety_handler import SafetyHandler

class RAGProcessor:
    """Handles retrieval-augmented generation logic."""

    def __init__(self, db_manager: DatabaseManager, generator: TextGenerator):
        self.db_manager = db_manager
        self.generator = generator
        self.safety_handler = SafetyHandler()  # Initialize safety handler

    def get_relevant_documents(self, query_embedding: List[float], table_name: str = "knowledge_base", top_k: int = 5) -> List[str]:
        """Retrieves the most relevant documents using cosine similarity (calculated in Python)."""
        try:
            all_docs = self.db_manager.get_all_documents_and_embeddings(table_name)
            if not all_docs:
                return []

            similarities = []
            for doc in all_docs:
                # Handle potential JSON parsing issues and ensure embedding is a list
                if isinstance(doc['embedding'], str):
                    try:
                        doc_embedding = json.loads(doc['embedding'])
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for document ID {doc['id']}: {doc['embedding']}")
                        continue #skip this doc
                elif isinstance(doc['embedding'], list):
                    doc_embedding = doc['embedding']
                else:
                    print(f"Unexpected embedding type for document ID {doc['id']}: {type(doc['embedding'])}")
                    continue  # Skip

                if not isinstance(doc_embedding, list):
                    print(f"Embedding for document ID {doc['id']} is not a list: {doc_embedding}")
                    continue

                try:
                    # Convert to NumPy arrays for calculation
                    doc_embedding_np = np.array(doc_embedding, dtype=np.float32)
                    query_embedding_np = np.array(query_embedding, dtype=np.float32) # No tolist()

                    similarity = np.dot(query_embedding_np, doc_embedding_np) / (np.linalg.norm(query_embedding_np) * np.linalg.norm(doc_embedding_np))
                    similarities.append((doc['content'], similarity))

                except Exception as e:
                    print(f"Error in similarity calculation: {e}")
                    continue # Skip this document on error

            sorted_documents = sorted(similarities, key=lambda x: x[1], reverse=True)  # Sort by similarity (descending)
            return [doc[0] for doc in sorted_documents[:top_k]]

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def generate_response(self, user_question: str, device: str, question_id: int = 0) -> str:
        """Generates a response using RAG."""
        
        # Check for harmful content first
        is_harmful, safety_response, metadata = self.safety_handler.process_input(user_question)
        
        if is_harmful:
            # Store safety response in interaction history
            data_point = {
                'context': "Safety response",
                'question': user_question,
                'answer': safety_response,
                'metadata': metadata or {'topic': 'Safety Response', 'questionID': question_id}
            }
            self.db_manager.add_interaction(data_point)
            return safety_response

        # Retrieve the conversation history
        history = self.db_manager.get_conversation_history()

        max_history_turns = 5
        context_string = "".join(
            f"USER: {turn['questionText']}\nTHERAPIST: {turn['answerText']}\n"
            for turn in history[-max_history_turns:]
        )

        query_embedding = self.generator.get_embedding(user_question)
        if query_embedding is None:
            return "I'm sorry, I encountered an error processing your question."

        # Convert the embedding to a list *before* passing it.
        query_embedding_list = query_embedding.tolist()

        relevant_documents = self.get_relevant_documents(query_embedding_list, top_k=3)
        knowledge_base_context = "\n\n".join(relevant_documents)

        selected_template_name = "Others"
        if any(word in user_question.lower() for word in ["sad", "depressed", "down", "unhappy"]):
            selected_template_name = "Empathy and Validation"
        elif any(word in user_question.lower() for word in ["anxious", "worried", "stressed", "nervous"]):
            selected_template_name = "Affirmation and Reassurance"
        elif any(word in user_question.lower() for word in ["help", "advice", "tips"]):
            selected_template_name = "Providing Suggestions"
        elif any(word in user_question.lower() for word in ["why", "explain", "understand"]):
            selected_template_name = "Information"

        selected_template = prompt_templates[selected_template_name]

        prompt = selected_template.format(topic="General Conversation", question=user_question)
        prompt = prompt.replace("THERAPIST:", f"""Previous Conversation:
        {context_string}

        Knowledge Base Context:
        {knowledge_base_context}

        THERAPIST:""")

        response = self.generator.generate_text(prompt)

        if self.generator.is_toxic(response):
            return "I'm sorry, I can't respond to that in a helpful way."

        data_point = {
            'context': prompt,
            'question': user_question,
            'answer': response,
            'metadata': {
                'topic': 'General Conversation',
                'questionTitle': '',
                'questionID': question_id,  # Use the passed questionID
            }
        }
        self.db_manager.add_interaction(data_point)
        return response
