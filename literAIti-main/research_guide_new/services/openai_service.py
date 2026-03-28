import json
import openai
from typing import Dict, Any, List, Optional
from models.context import FieldContext, UserContext, PromptBreakdown


SYSTEM_PROMPTS = {
    "breakdown": """You are a research assistant that helps break down user queries into structured components.
Given a user's research query, identify and extract:
1. Field Context: What academic/research field are they asking about?
2. User Context: What is the user's background, expertise, and what are they familiar/unfamiliar with?
3. User Request: What exactly are they asking for?

If you need more information to complete any section, list what's missing in a "missing_info" field.
Be thorough but concise. Return your analysis in a structured format.""",

    "field_context_generator": """You are an expert research analyst. Given raw research data about a field, generate a comprehensive field context that includes:
1. Major Areas of Development/Excitement - Key research frontiers and active areas
2. Popular Debates/Criticisms - Ongoing discussions, controversies, and critiques
3. Institutional Analysis - Strengths and weaknesses of key institutions/organizations

Return JSON with fields:
- major_areas: array of strings
- debate_criticisms: array of strings
- institution_analysis: object mapping institution names to short analyses
- sources: array of objects with title and url
- summary: short summary string""",

    "user_model_generator": """You are a research advisor helping users discover their research direction.
Given the user's background and current field context, suggest 2-3 specific research directions they might find interesting.
Consider:
- Their existing expertise and how it could transfer
- Gaps they might fill
- Current trends and opportunities in the field

Always ask for feedback to refine suggestions.""",

    "research_assistant": """You are a thoughtful research advisor helping users clarify their research questions.
Use the provided Field Context and User Context to generate informed, helpful responses.
Always reference the contexts to ensure your answers are grounded in current research landscape.
If the user seems uncertain, help them explore different angles.""",

    "response_synthesizer": """You synthesize research information into clear, actionable insights.
Given scraped research data and a user's query, provide a comprehensive but focused response.
Highlight key findings, contradictions, and gaps in the literature."""
}


class OpenAIService:
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 2000):
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_llm(self, system_prompt: str, user_prompt: str, response_format: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        
        response = openai.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
            raise

    def breakdown_prompt(self, user_query: str) -> PromptBreakdown:
        result = self._call_llm(
            SYSTEM_PROMPTS["breakdown"],
            f"Analyze this research query:\n\n{user_query}",
            response_format="json"
        )
        
        data = self._parse_json(result)
        
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
            response_format="json"
        )
        
        data = self._parse_json(result)
        
        context = FieldContext(field_name)
        context.update(
            major_areas=data.get("major_areas", []),
            debates=data.get("debate_criticisms", []),
            institutions=data.get("institution_analysis", {}),
            sources=data.get("sources", []),
            summary=data.get("summary", "")
        )
        
        return context

    def plan_source_extraction(self, user_query: str, field_name: str, user_background: str = "") -> Dict[str, Any]:
        prompt = f"""You are planning a web research pass for a research assistant.

User query: {user_query}
Field name: {field_name}
User background: {user_background or "Not provided"}

Return JSON with:
- extraction_strategy: short string explaining the plan
- search_queries: 4-6 strings for discoverability/logging
- sources: array of 3-5 objects, each with:
  - title: source title
  - url: full URL
  - rationale: why this source matters
  - extraction_goal: exact instruction for a browser automation tool to extract the needed information from that page and return JSON

Choose sources that together cover recent developments, debates, and key institutions/publication venues. Prefer trustworthy sources such as labs, conferences, journals, universities, surveys, and reputable explainers."""

        result = self._call_llm(
            "You are a research planning assistant that selects concrete sources and extraction instructions.",
            prompt,
            response_format="json"
        )

        data = self._parse_json(result)
        return {
            "extraction_strategy": data.get("extraction_strategy", ""),
            "search_queries": data.get("search_queries", []),
            "sources": data.get("sources", []),
        }

    def generate_user_model(self, user_background: str, field_context: FieldContext) -> UserContext:
        context_summary = field_context.to_prompt_context()
        
        result = self._call_llm(
            SYSTEM_PROMPTS["user_model_generator"],
            f"User Background:\n{user_background}\n\nField Context:\n{context_summary}",
            response_format="json"
        )
        
        data = self._parse_json(result)
        
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

Provide updated contexts in JSON format with fields: updated_field_context, updated_user_context, direction_change (what changed)."""
        
        result = self._call_llm(
            "You help refine context based on user feedback.",
            prompt,
            response_format="json"
        )
        
        data = self._parse_json(result)
        
        return field_context, user_context
