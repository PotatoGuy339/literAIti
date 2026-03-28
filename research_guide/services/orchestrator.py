from typing import Dict, Any, Optional, List
from models.context import ResearchSession, FieldContext, UserContext, PromptBreakdown
from services.openai_service import OpenAIService
from services.tinyfish_service import TinyfishService
import uuid


class ResearchOrchestrator:
    def __init__(self, openai_service: OpenAIService, tinyfish_service: TinyfishService):
        self.openai = openai_service
        self.tinyfish = tinyfish_service
        self.sessions: Dict[str, ResearchSession] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        self.sessions[session_id] = ResearchSession(session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        return self.sessions.get(session_id)

    def process_initial_query(self, session_id: str, user_query: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        breakdown = self.openai.breakdown_prompt(user_query)
        session.breakdown = breakdown
        
        if breakdown.needs_more_info():
            return {
                "needs_more_info": True,
                "missing_info": breakdown.missing_info,
                "breakdown": breakdown.to_dict()
            }
        
        session.field_context = FieldContext(breakdown.field_context)
        session.user_context = UserContext()
        session.user_context.background = breakdown.user_context
        
        return {
            "needs_more_info": False,
            "breakdown": breakdown.to_dict(),
            "field_context": session.field_context.to_dict() if session.field_context else None
        }

    def gather_field_data(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session or not session.field_context:
            return {"error": "Session or field context not found"}
        
        raw_data = self.tinyfish.research_field(session.field_context.field_name)
        session.field_context.raw_data = raw_data
        
        queries = self.openai.generate_search_queries(
            session.field_context.field_name,
            session.user_context
        )
        session.field_context.search_queries = queries
        
        return {
            "raw_data_collected": len(raw_data),
            "search_queries": queries
        }

    def generate_field_context(self, session_id: str) -> FieldContext:
        session = self.get_session(session_id)
        if not session or not session.field_context:
            raise ValueError("Session or field context not found")
        
        field_context = self.openai.generate_field_context(
            session.field_context.raw_data,
            session.field_context.field_name
        )
        
        session.field_context = field_context
        return field_context

    def generate_user_model(self, session_id: str) -> UserContext:
        session = self.get_session(session_id)
        if not session or not session.field_context:
            raise ValueError("Session or field context not found")
        
        user_context = self.openai.generate_user_model(
            session.breakdown.user_context,
            session.field_context
        )
        
        session.user_context = user_context
        return user_context

    def answer_query(self, session_id: str, user_query: str) -> str:
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
        
        if not session.field_context or not session.user_context:
            return "Please complete the context gathering phase first."
        
        conversation = session.messages[-10:] if session.messages else None
        
        response = self.openai.generate_response(
            user_query,
            session.field_context,
            session.user_context,
            conversation
        )
        
        session.add_message("user", user_query)
        session.add_message("assistant", response)
        
        return response

    def process_feedback(self, session_id: str, feedback: str, is_positive: bool) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        session.user_context.update_from_feedback(feedback, is_positive)
        
        return {
            "user_context_updated": True,
            "feedback_history_length": len(session.user_context.feedback_history)
        }

    def full_session_flow(self, session_id: str, user_query: str, user_background: str = "") -> Dict[str, Any]:
        result = self.process_initial_query(session_id, user_query)
        
        if result.get("needs_more_info"):
            return result
        
        if user_background:
            session = self.get_session(session_id)
            session.breakdown.user_context = user_background
            session.user_context = UserContext()
            session.user_context.background = user_background
        
        self.gather_field_data(session_id)
        field_context = self.generate_field_context(session_id)
        user_context = self.generate_user_model(session_id)
        
        session = self.get_session(session_id)
        return {
            "breakdown": session.breakdown.to_dict(),
            "field_context": field_context.to_dict(),
            "user_context": user_context.to_dict(),
            "ready_for_queries": True
        }
