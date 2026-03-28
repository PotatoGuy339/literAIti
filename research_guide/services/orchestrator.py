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
    
    def _extract_field_from_text(self, text: str) -> str:
        common_fields = [
            "Physics", "Chemistry", "Biology", "Computer Science", "Mathematics",
            "Machine Learning", "Artificial Intelligence", "Quantum Computing", "Neuroscience",
            "Genetics", "Climate Science", "Astrophysics", "Materials Science",
            "Engineering", "Medicine", "Psychology", "Economics", "Sociology",
            "Philosophy", "Linguistics", "Environmental Science", "Biotechnology",
            "Data Science", "Robotics", "Nanotechnology", "Pharmacology"
        ]
        
        text_lower = text.lower()
        for field in common_fields:
            if field.lower() in text_lower:
                return field
        
        words = text.split()
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}".lower()
            for field in common_fields:
                if field.lower() in phrase:
                    return field
        
        return "Research"

    def process_initial_query(self, session_id: str, user_query: str) -> Dict[str, Any]:
        print(f"[DEBUG] process_initial_query called with query: '{user_query[:50]}...'")
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        breakdown = self.openai.breakdown_prompt(user_query)
        print(f"[DEBUG] breakdown result: field='{breakdown.field_context}', request='{breakdown.user_request[:50] if breakdown.user_request else ''}...'")
        session.breakdown = breakdown
        
        if breakdown.needs_more_info():
            print(f"[DEBUG] breakdown needs more info: {breakdown.missing_info}")
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
            session.user_context or None
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
        print(f"[DEBUG] answer_query: session={session is not None}")
        if not session:
            raise ValueError("Session not found")
        
        print(f"[DEBUG] answer_query: field_context={session.field_context is not None}, user_context={session.user_context is not None}")
        
        if not session.field_context or not session.user_context:
            return "Please complete the context gathering phase first."
        
        print(f"[DEBUG] answer_query: major_areas={session.field_context.major_areas}, summary={session.field_context.summary}")
        
        if not session.field_context.major_areas and not session.field_context.summary:
            return "Context is empty. Please ensure context gathering completed successfully."
        
        conversation = session.messages[-10:] if session.messages else None
        
        response = self.openai.generate_response(
            user_query,
            session.field_context,
            session.user_context,
            conversation
        )
        
        session.add_message("user", user_query)
        session.add_message("assistant", response)
        
        if len(session.messages) % 3 == 0:
            self._refine_contexts_from_conversation(session_id)
        
        return response
    
    def _refine_contexts_from_conversation(self, session_id: str):
        session = self.get_session(session_id)
        if not session or not session.messages:
            return
        
        conversation_summary = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in session.messages[-6:]])
        
        field_context = self.openai.generate_field_context(
            conversation_summary,
            f"Refined research area based on conversation"
        )
        
        if field_context.major_areas:
            session.field_context.major_areas = field_context.major_areas
        if field_context.debate_criticisms:
            session.field_context.debate_criticisms = field_context.debate_criticisms
        if field_context.summary:
            session.field_context.summary = field_context.summary
        
        user_context = self.openai.generate_user_model(
            session.user_context.background or "",
            session.field_context
        )
        
        if user_context.current_focus:
            session.user_context.current_focus = user_context.current_focus
        if user_context.expertise_areas:
            session.user_context.expertise_areas = user_context.expertise_areas
        if user_context.suggested_directions:
            session.user_context.suggested_directions = user_context.suggested_directions

    def process_feedback(self, session_id: str, feedback: str, is_positive: bool) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if not session or not session.user_context:
            return {"error": "Session or user context not found"}
        
        session.user_context.update_from_feedback(feedback, is_positive)
        
        return {
            "user_context_updated": True,
            "feedback_history_length": len(session.user_context.feedback_history)
        }

    def full_session_flow(self, session_id: str, user_query: str, user_background: str = "") -> Dict[str, Any]:
        print(f"[DEBUG] full_session_flow called with session_id={session_id}")
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        result = self.process_initial_query(session_id, user_query)
        print(f"[DEBUG] process_initial_query result: {result}")
        
        if user_background:
            session.breakdown.user_context = user_background
            session.user_context = UserContext()
            session.user_context.background = user_background
        
        missing_info = result.get("missing_info", [])
        has_field_context = session.field_context and bool(session.field_context.field_name) and session.field_context.field_name not in ["", "missing_info"]
        
        if not has_field_context:
            field_from_query = self._extract_field_from_text(user_query + " " + user_background)
            session.field_context = FieldContext(field_from_query)
            if missing_info:
                session.breakdown.missing_info = missing_info
        
        if session.field_context:
            print(f"[DEBUG] Session field_context field_name: '{session.field_context.field_name}'")
        
        try:
            field_data_result = self.gather_field_data(session_id)
            print(f"[DEBUG] gather_field_data result: {field_data_result}")
            
            if field_data_result.get("error"):
                return field_data_result
            
            field_context = self.generate_field_context(session_id)
            print(f"[DEBUG] field_context generated: major_areas={len(field_context.major_areas) if field_context.major_areas else 0}")
            session.field_context = field_context
            user_context = self.generate_user_model(session_id)
            session.user_context = user_context
            
            response_data = {
                "breakdown": session.breakdown.to_dict(),
                "field_context": field_context.to_dict(),
                "user_context": user_context.to_dict(),
                "ready_for_queries": True
            }
            
            print(f"[DEBUG] Returning field_context major_areas: {response_data['field_context']['major_areas']}")
            print(f"[DEBUG] Returning field_context debates: {response_data['field_context']['debate_criticisms']}")
            print(f"[DEBUG] Returning field_context summary: {response_data['field_context']['summary']}")
            
            has_meaningful_data = (
                len(field_context.major_areas) > 0 or 
                len(field_context.debate_criticisms) > 0 or 
                field_context.summary
            )
            field_is_generic = session.field_context.field_name in ["General Research", ""] if session.field_context else True
            
            summary_parts = []
            if field_context.major_areas:
                summary_parts.append(f"I've identified {len(field_context.major_areas)} major research areas: {', '.join(field_context.major_areas[:3])}")
            if field_context.debate_criticisms:
                summary_parts.append(f"There are {len(field_context.debate_criticisms)} key debates worth exploring.")
            if field_context.summary:
                summary_parts.append(field_context.summary[:200] + "..." if len(field_context.summary) > 200 else field_context.summary)
            
            welcome_message = "Welcome! Here's what I found:\n\n" + "\n".join(f"• {p}" for p in summary_parts) + "\n\nFeel free to ask me about these areas, explore specific topics, or tell me more about your background so I can refine my suggestions."
            
            if missing_info and (not has_meaningful_data or field_is_generic):
                response_data["clarification_needed"] = missing_info
                if isinstance(missing_info, dict):
                    missing_items = list(missing_info.values())[:2]
                elif isinstance(missing_info, list):
                    missing_items = missing_info[:2]
                else:
                    missing_items = [str(missing_info)]
                welcome_message += "\n\n💡 P.S. To give you more tailored advice, could you share: " + "; ".join(missing_items) + "?"
            
            response_data["message"] = welcome_message
            
            return response_data
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Context generation failed: {str(e)}"}
