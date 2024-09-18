from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from fuzzywuzzy import fuzz
import re
from spellchecker import SpellChecker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
CONTEXT_FILE = 'school_contexts/school_context.txt'

# Global variables
sentence_model = None
context_chunks = []
context_embeddings = None
spell = SpellChecker()


def load_models():
    global sentence_model
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def load_and_process_context():
    global context_chunks, context_embeddings
    try:
        with open(CONTEXT_FILE, 'r') as f:
            full_context = f.read()
        context_chunks = full_context.split('\n\n')
        context_embeddings = sentence_model.encode(context_chunks)
        logger.info(f"Loaded and processed {len(context_chunks)} context chunks")
    except Exception as e:
        logger.error(f"Error loading or processing context: {str(e)}")
        raise


def preprocess_question(question):
    question = question.lower()
    # replace evhs with school so that these words are interchangeable.
    question = question.replace(" u ", " you ").replace(" evhs ", " school ")
    question = correct_spelling(question)
    question = re.sub(r'[^\w\s]', '', question)
    question = ' '.join(question.split())
    return question


def get_relevant_context(question, top_k=3, threshold=0.3):
    question_embedding = sentence_model.encode([question])
    similarities = cosine_similarity(question_embedding, context_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = []
    for i in top_indices:
        if similarities[i] > threshold:
            relevant_chunks.append(context_chunks[i])
        else:
            break
    return '\n\n'.join(relevant_chunks)


def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return ' '.join(corrected_words)


def is_greeting(message):
    greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    return any(greeting in message.lower() for greeting in greetings)


def is_farewell(message):
    farewells = ['bye', 'goodbye', 'see you', 'take care', 'later']
    return any(farewell in message.lower() for farewell in farewells)


def is_thank_you(message):
    thank_yous = ['thank you', 'thanks', 'thank you very much', 'thanks a lot']
    return any(thank_you in message.lower() for thank_you in thank_yous)


def validate_input(question):
    question = question.strip().lower()
    words = question.split()

    # List of invalid single words to reject
    invalid_single_words = {'ok', 'what', 'when', 'where', 'who', 'why', 'how', 'a', 'an', 'the', 'is', 'are', 'was',
                            'were'}

    # Check if it's a single word and in the invalid list
    if len(words) == 1 and words[0] in invalid_single_words:
        return False

    # Allow valid greetings
    if len(words) == 1 and words[0] in {'hi', 'hello', 'hey', 'greetings','bye', 'goodbye', 'later', 'thanks'}:
        return True

    # Additional check for very short questions
    if len(question) < 3:
        return False

    return True


def is_bot_capability_question(question):
    capability_patterns = [
        r'what can you (do|help|assist with)',
        r'how can you (help|assist)',
        r'what are you capable of',
        r'what do you know about',
        r'what kind of (questions|information) can you (answer|provide)',
        r'tell me about your capabilities'
    ]
    return any(re.search(pattern, question.lower()) for pattern in capability_patterns)


def is_yes_no_question(question):
    yes_no_starters = ['does', 'do', 'is', 'are', 'can', 'has', 'have', 'will', 'should', 'would', 'could']
    words = question.lower().split()

    # Check if any of the first three words is a yes/no starter
    for word in words[:3]:
        if word in yes_no_starters:
            return True

    # Check for inverted questions
    if len(words) >= 2 and words[1] in yes_no_starters:
        return True

    return False


def classify_question(question):
    question = question.lower()
    if is_greeting(question):
        return 'greeting'
    elif is_farewell(question):
        return 'farewell'
    elif is_thank_you(question):
        return 'thankyou'
    elif is_bot_capability_question(question):
        return 'bot_capability'
    elif any(word in question for word in ['how', 'when', 'what', 'where', 'who', 'why']):
        return 'wh_question'
    elif any(word in question for word in ['where', 'location']):
        return 'location'
    elif is_yes_no_question(question):
        return 'yes_no'
    else:
        return 'general'


def answer_question(question):
    if not validate_input(question):
        return ("I need more context to provide a helpful answer. Could you please ask a more specific question about "
                "Evergreen Valley High School?")

    original_question = question
    corrected_question = preprocess_question(question)
    logger.info("Question asked - " + corrected_question)
    question_type = classify_question(corrected_question)

    if question_type == 'greeting':
        return "Hello! Welcome to the Evergreen Valley High School chat. How can I assist you today?"
    if question_type == 'farewell':
        return "Feel free to come back anytime if you have more questions. Goodbye!"
    if question_type == 'thankyou':
        return "It was my pleasure! If you have any more questions, feel free to ask!"
    if question_type == 'bot_capability':
        return ("I'm an AI assistant for Evergreen Valley High School. I can provide information about the school's "
                "programs, facilities, staff, policies, and answer general inquiries. Feel free to ask me anything "
                "related to the school, and I'll do my best to assist you!")

    try:
        relevant_context = get_relevant_context(corrected_question)

        if not relevant_context:
            return "I'm sorry, but I don't have enough relevant information to answer that question. Could you please ask something specific about Evergreen Valley High School?"

        # Split the relevant context into sentences
        sentences = re.split(r'(?<=[.!?])\s+', relevant_context)

        # Find the most relevant sentence(s)
        relevant_sentences = []
        for sentence in sentences:
            if fuzz.partial_ratio(corrected_question, sentence.lower()) > 70:  # Adjust this threshold as needed
                relevant_sentences.append(sentence)

        if relevant_sentences:
            answer = ' '.join(relevant_sentences)
        else:
            # If no highly relevant sentence is found, return the most similar chunk
            answer = max(sentences, key=lambda s: fuzz.partial_ratio(corrected_question, s.lower()))

        # Check if the answer is relevant enough
        if fuzz.partial_ratio(corrected_question, answer.lower()) < 50:
            return "I'm sorry, but I don't have enough relevant information to answer that question accurately. Could you please ask something specific about Evergreen Valley High School?"

        if question_type == 'yes_no':
            question_terms = set(corrected_question.lower().split()) - {'does', 'do', 'is', 'are', 'can', 'has', 'have',
                                                                        'will', 'should', 'would', 'could'}
            if any(term in answer.lower() for term in question_terms):
                return f"Yes. {answer}"
            else:
                return f"No. Based on the available information, {answer}"
        elif question_type in ['wh_question', 'general']:
            return answer.strip()
        else:
            return "I'm sorry, I couldn't understand your question. Could you please rephrase it?"

    except Exception as e:
        logger.error(f"Error in answer_question: {str(e)}")
        return "I'm sorry, an error occurred while processing your question. Please try again."


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data['message']
        bot_response = answer_question(user_message)
        return jsonify({'answer': bot_response})
    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500


@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    load_models()
    load_and_process_context()
    app.run(debug=True)