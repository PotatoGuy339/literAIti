from flask import Flask, render_template, request, jsonify, session
from services import OpenAIService, TinyfishService, ResearchOrchestrator
from utils.config import load_config
import uuid

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

config = load_config()

llm_service = OpenAIService(
    api_key=config['openai']['api_key'],
    model=config['openai'].get('model', 'gpt-4'),
    temperature=config['openai'].get('temperature', 0.7),
    max_tokens=config['openai'].get('max_tokens', 2000)
)

tinyfish_service = TinyfishService(
    api_key=config['tinyfish']['api_key'],
    base_url=config['tinyfish'].get('base_url', 'https://api.tinyfish.ai/v1'),
    max_results=config['tinyfish'].get('max_results', 20),
    timeout=config['tinyfish'].get('timeout', 30)
)

orchestrator = ResearchOrchestrator(llm_service, tinyfish_service)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/session', methods=['POST'])
def create_session():
    session_id = orchestrator.create_session()
    return jsonify({"session_id": session_id})


@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.json
    session_id = data.get('session_id')
    user_query = data.get('query')
    user_background = data.get('background', '')
    
    if not session_id or not user_query:
        return jsonify({"error": "Missing session_id or query"}), 400
    
    result = orchestrator.full_session_flow(session_id, user_query, user_background)
    print(f"[DEBUG] API returning: {result}")
    return jsonify(result)


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    
    print(f"[DEBUG] /api/chat called with session_id={session_id}, message={message[:50]}...")
    
    if not session_id or not message:
        return jsonify({"error": "Missing session_id or message"}), 400
    
    response = orchestrator.answer_query(session_id, message)
    print(f"[DEBUG] answer_query returned: {response[:100] if isinstance(response, str) else response}...")
    return jsonify({"response": response})


@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    session_id = data.get('session_id')
    feedback = data.get('feedback')
    is_positive = data.get('is_positive', True)
    
    if not session_id or feedback is None:
        return jsonify({"error": "Missing session_id or feedback"}), 400
    
    result = orchestrator.process_feedback(session_id, feedback, is_positive)
    return jsonify(result)


@app.route('/api/context/<session_id>', methods=['GET'])
def get_context(session_id):
    session = orchestrator.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify({
        "field_context": session.field_context.to_dict() if session.field_context else None,
        "user_context": session.user_context.to_dict() if session.user_context else None
    })


@app.route('/api/refresh-context/<session_id>', methods=['POST'])
def refresh_context(session_id):
    session = orchestrator.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    orchestrator._refine_contexts_from_conversation(session_id)
    
    return jsonify({
        "field_context": session.field_context.to_dict() if session.field_context else None,
        "user_context": session.user_context.to_dict() if session.user_context else None
    })


if __name__ == '__main__':
    app.run(
        host=config['app'].get('host', '0.0.0.0'),
        port=config['app'].get('port', 5000),
        debug=config['app'].get('debug', True)
    )
