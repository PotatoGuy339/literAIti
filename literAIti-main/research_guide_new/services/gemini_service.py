import google.generativeai as genai
from typing import Dict, Any, List, Optional
from models.context import FieldContext, UserContext, PromptBreakdown


SYSTEM_PROMPTS = {
    "breakdown": """You are a research assistant that helps break down user queries into structured components.
Given a user's research query, identify and extract:
1. Field Context: What academic/research field are they asking about?
2. User Context: What is the user's background, expertise, and what are they familiar/unfamiliar with?
3. User Request: What exactly are they asking for?

If you need more information to complete any section, list what's missing in a "missing_info" field.
Be thorough but concise. Return your analysis in JSON format.""",

    "field_context_generator": """You are an expert research analyst. Given raw research data about a field, generate a comprehensive field context that includes:
1. Major Areas of Development/Excitement - Key research frontiers and active areas
2. Popular Debates/Criticisms - Ongoing discussions, controversies, and critiques
3. Institutional Analysis - Strengths and weaknesses of key institutions/organizations

Format your response as JSON with fields: major_areas (array), debate_criticisms (array), institution_analysis (object), summary (string).""",

    "user_model_generator": """You are a research advisor helping users discover their research direction.
Given the user's background and current field context, suggest 2-3 specific research directions they might find interesting.
Consider:
- Their existing expertise and how it could transfer
- Gaps they might fill
- Current trends and opportunities in the field

Return JSON with fields: expertise_areas (array), known_gaps (array), suggested_directions (array), current_focus (string).""",

    "research_assistant": """You are a thoughtful research advisor helping users clarify their research questions.
Use the provided Field Context and User Context to generate informed, helpful responses.
Always reference the contexts to ensure your answers are grounded in current research landscape.
If the user seems uncertain, help them explore different angles.""",

    "response_synthesizer": """You synthesize research information into clear, actionable insights.
Given scraped research data and a user's query, provide a comprehensive but focused response.
Highlight key findings, contradictions, and gaps in the literature."""
}


class GeminiService:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 2000,
        }

    def _call_llm(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        model = genai.GenerativeModel(self.model_name)
        
        if json_mode:
            self.generation_config["response_mime_type"] = "application/json"
        
        response = model.generate_content(
            combined_prompt,
            generation_config=self.generation_config
        )
        return response.text

    def breakdown_prompt(self, user_query: str) -> PromptBreakdown:
        result = self._call_llm(
            SYSTEM_PROMPTS["breakdown"],
            f"Analyze this research query:\n\n{user_query}",
            json_mode=True
        )
        
        import json
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            data = self._extract_json_from_text(result)
        
        breakdown = PromptBreakdown()
        breakdown.field_context = data.get("field_context", "")
        breakdown.user_context = data.get("user_context", "")
        breakdown.user_request = data.get("user_request", "")
        breakdown.missing_info = data.get("missing_info", [])
        
        return breakdown

    def generate_field_context(self, raw_data: str, field_name: str) -> FieldContext:
        result = self._call_llm(
            SYSTEM_PROMPTS["field_context_generator"],
            f"Research Field: {field_name}\n\nCollected Research Data:\n{raw_data[:8000]}",
            json_mode=True
        )
        
        import json
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            data = self._extract_json_from_text(result)
        
        context = FieldContext(field_name)
        context.update(
            major_areas=data.get("major_areas", []),
            debates=data.get("debate_criticisms", []),
            institutions=data.get("institution_analysis", {}),
            sources=data.get("sources", []),
            summary=data.get("summary", "")
        )
        
        return context

    def generate_search_queries(self, field_name: str, user_context: UserContext = None) -> List[str]:
        prompt = f"Generate 5-7 specific search queries for the research field: {field_name}\n\n"
        prompt += "Focus on finding:\n"
        prompt += "- Major recent developments and breakthroughs\n"
        prompt += "- Key debates and criticisms in the field\n"
        prompt += "- Important institutions and their work\n"
        prompt += "- Publication venues (conferences, journals)\n"
        
        if user_context and user_context.expertise_areas:
            prompt += f"\nUser expertise: {', '.join(user_context.expertise_areas)}\n"
            prompt += "Also query for how this expertise might connect to the field.\n"
        
        prompt += "\nReturn JSON with a 'queries' array."
        
        result = self._call_llm(
            "You are a research search expert. Generate targeted search queries.",
            prompt,
            json_mode=True
        )
        
        import json
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            data = self._extract_json_from_text(result)
        return data.get("queries", [])

    def generate_user_model(self, user_background: str, field_context: FieldContext) -> UserContext:
        context_summary = field_context.to_prompt_context()
        
        result = self._call_llm(
            SYSTEM_PROMPTS["user_model_generator"],
            f"User Background:\n{user_background}\n\nField Context:\n{context_summary}",
            json_mode=True
        )
        
        import json
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            data = self._extract_json_from_text(result)
        
        user = UserContext()
        user.background = user_background
        user.expertise_areas = data.get("expertise_areas", [])
        user.known_gaps = data.get("known_gaps", [])
        user.suggested_directions = data.get("suggested_directions", [])
        user.current_focus = data.get("current_focus", "")
        
        return user

    def generate_response(self, user_query: str, field_context: FieldContext,
                         user_context: UserContext, conversation_history: List[Dict] = None) -> str:
        context_prompt = f"""Field Context:\n{field_context.to_prompt_context()}

User Context:\n{user_context.to_prompt_context()}
"""
        
        if conversation_history:
            context_prompt += "\nConversation History:\n"
            for msg in conversation_history[-5:]:
                context_prompt += f"{msg['role']}: {msg['content'][:500]}\n"
        
        context_prompt += f"\n\nCurrent Query: {user_query}"
        
        return self._call_llm(
            SYSTEM_PROMPTS["research_assistant"],
            context_prompt
        )

    def refine_contexts(self, user_feedback: str, field_context: FieldContext,
                       user_context: UserContext) -> tuple[FieldContext, UserContext]:
        prompt = f"""Analyze this user feedback and suggest updates to the contexts.

User Feedback: {user_feedback}

Current Field Context:\n{field_context.to_prompt_context()}

Current User Context:\n{user_context.to_prompt_context()}

Return JSON with fields: updated_field_context, updated_user_context, direction_change."""
        
        result = self._call_llm(
            "You help refine context based on user feedback.",
            prompt,
            json_mode=True
        )
        
        return field_context, user_context

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        import json
        import re
        
        json_patterns = [
            r'\{[^{}]*\}',
            r'\[[^\[\]]*\]',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return {}
