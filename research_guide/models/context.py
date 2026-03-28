import json
from datetime import datetime
from typing import List, Optional, Dict, Any


class PromptBreakdown:
    def __init__(self):
        self.field_context: str = ""
        self.user_context: str = ""
        self.user_request: str = ""
        self.missing_info: List[str] = []
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_context": self.field_context,
            "user_context": self.user_context,
            "user_request": self.user_request,
            "missing_info": self.missing_info,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def needs_more_info(self) -> bool:
        return len(self.missing_info) > 0


class FieldContext:
    def __init__(self, field_name: str = ""):
        self.field_name: str = field_name
        self.major_areas: List[str] = []
        self.debate_criticisms: List[str] = []
        self.institution_analysis: Dict[str, str] = {}
        self.sources: List[Dict[str, str]] = []
        self.search_queries: List[str] = []
        self.raw_data: str = ""
        self.summary: str = ""
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    def update(self, major_areas: List[str] = None, debates: List[str] = None,
               institutions: Dict[str, str] = None, sources: List[Dict[str, str]] = None,
               summary: str = None):
        if major_areas is not None:
            self.major_areas = major_areas
        if debates is not None:
            self.debate_criticisms = debates
        if institutions is not None:
            self.institution_analysis = institutions
        if sources is not None:
            self.sources = sources
        if summary is not None:
            self.summary = summary
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "major_areas": self.major_areas,
            "debate_criticisms": self.debate_criticisms,
            "institution_analysis": self.institution_analysis,
            "sources": self.sources,
            "search_queries": self.search_queries,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def to_prompt_context(self) -> str:
        context = f"Field: {self.field_name}\n\n"
        context += f"Major Areas of Development:\n"
        for area in self.major_areas:
            context += f"  - {area}\n"
        context += f"\nPopular Debates and Criticisms:\n"
        for debate in self.debate_criticisms:
            context += f"  - {debate}\n"
        context += f"\nInstitutional Analysis:\n"
        for inst, analysis in self.institution_analysis.items():
            context += f"  {inst}: {analysis}\n"
        if self.sources:
            context += f"\nKey Sources:\n"
            for source in self.sources[:5]:
                context += f"  - {source.get('title', 'Unknown')}: {source.get('url', '')}\n"
        return context


class UserContext:
    def __init__(self, user_id: str = ""):
        self.user_id: str = user_id
        self.background: str = ""
        self.interests: List[str] = []
        self.expertise_areas: List[str] = []
        self.known_gaps: List[str] = []
        self.suggested_directions: List[str] = []
        self.feedback_history: List[Dict[str, Any]] = []
        self.current_focus: Optional[str] = None
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    def update_from_feedback(self, feedback: str, is_positive: bool):
        self.feedback_history.append({
            "feedback": feedback,
            "is_positive": is_positive,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()

    def add_suggested_direction(self, direction: str):
        if direction not in self.suggested_directions:
            self.suggested_directions.append(direction)

    def set_focus(self, focus: str):
        self.current_focus = focus
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "background": self.background,
            "interests": self.interests,
            "expertise_areas": self.expertise_areas,
            "known_gaps": self.known_gaps,
            "suggested_directions": self.suggested_directions,
            "feedback_history": self.feedback_history,
            "current_focus": self.current_focus,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def to_prompt_context(self) -> str:
        context = f"User Background: {self.background}\n\n"
        context += f"Expertise Areas: {', '.join(self.expertise_areas) if self.expertise_areas else 'Not specified'}\n\n"
        context += f"Known Gaps: {', '.join(self.known_gaps) if self.known_gaps else 'Not specified'}\n\n"
        context += f"Current Focus: {self.current_focus if self.current_focus else 'Exploring options'}\n\n"
        if self.suggested_directions:
            context += f"Suggested Research Directions:\n"
            for direction in self.suggested_directions:
                context += f"  - {direction}\n"
        return context


class ResearchSession:
    def __init__(self, session_id: str = ""):
        self.session_id: str = session_id
        self.breakdown: PromptBreakdown = PromptBreakdown()
        self.field_context: Optional[FieldContext] = None
        self.user_context: Optional[UserContext] = None
        self.messages: List[Dict[str, Any]] = []
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "breakdown": self.breakdown.to_dict(),
            "field_context": self.field_context.to_dict() if self.field_context else None,
            "user_context": self.user_context.to_dict() if self.user_context else None,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
