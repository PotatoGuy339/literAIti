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

Format your response clearly with headers. Be analytical and insightful.""",

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
            if "json" not in user_prompt.lower():
                messages[1] = {"role": "user", "content": user_prompt + "\n\nPlease respond in JSON format."}
        
        response = openai.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def breakdown_prompt(self, user_query: str) -> PromptBreakdown:
        result = self._call_llm(
            SYSTEM_PROMPTS["breakdown"],
            f"Analyze this research query:\n\n{user_query}",
            response_format="json"
        )
        
        import json
        data = json.loads(result)
        
        breakdown = PromptBreakdown()
        breakdown.field_context = data.get("field_context", "")
        breakdown.user_context = data.get("user_context", "")
        breakdown.user_request = data.get("user_request", "")
        breakdown.missing_info = data.get("missing_info", [])
        
        return breakdown

    def _extract_items_from_response(self, raw_data) -> List[str]:
        items = []
        if isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, str):
                    items.append(key)
                elif isinstance(value, dict):
                    if "title" in value:
                        items.append(value["title"])
                    else:
                        items.append(key)
        elif isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, str):
                    items.append(item)
                elif isinstance(item, dict):
                    if "title" in item:
                        items.append(item["title"])
                    elif "name" in item:
                        items.append(item["name"])
        return items

    def _extract_dict_from_response(self, raw_data) -> Dict[str, str]:
        result = {}
        if isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, str):
                    result[key] = value
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str):
                            result[k] = v
                        else:
                            result[k] = str(v)
        return result

    def _extract_list_from_response(self, raw_data) -> List:
        if isinstance(raw_data, list):
            if all(isinstance(item, dict) for item in raw_data):
                return raw_data
        return []

    def generate_field_context(self, raw_data: str, field_name: str) -> FieldContext:
        print(f"[DEBUG] generate_field_context raw_data length: {len(raw_data)}")
        print(f"[DEBUG] raw_data preview: {raw_data[:500]}...")
        
        result = self._call_llm(
            SYSTEM_PROMPTS["field_context_generator"],
            f"Research Field: {field_name}\n\nCollected Research Data:\n{raw_data[:8000]}",
            response_format="json"
        )
        
        print(f"[DEBUG] LLM result: {result[:500]}...")
        
        import json
        data = json.loads(result)
        
        print(f"[DEBUG] parsed data keys: {list(data.keys())}")
        print(f"[DEBUG] parsed data: {data}")
        
        context = FieldContext(field_name)
        
        major_areas_raw = data.get("major_areas") or data.get("Major Areas of Development/Excitement") or data.get("major_areas_of_development") or data.get("majorAreas") or data.get("areas") or []
        major_areas = self._extract_items_from_response(major_areas_raw)
        
        debates_raw = data.get("debate_criticisms") or data.get("Popular Debates/Criticisms") or data.get("popular_debates") or data.get("debates") or data.get("criticisms") or []
        debates = self._extract_items_from_response(debates_raw)
        
        institutions_raw = data.get("institution_analysis") or data.get("institutional_analysis") or data.get("Institutions") or data.get("institutions") or data.get("institutionAnalysis") or {}
        institutions = self._extract_dict_from_response(institutions_raw)
        
        sources_raw = data.get("sources") or data.get("Sources") or data.get("key_sources") or data.get("references") or []
        sources = self._extract_list_from_response(sources_raw)
        
        summary = data.get("summary") or data.get("Summary") or data.get("summary_text") or data.get("field_summary") or ""
        
        context.update(
            major_areas=major_areas,
            debates=debates,
            institutions=institutions,
            sources=sources,
            summary=summary
        )
        
        return context

    def generate_search_queries(self, field_name: str, user_context: Optional[UserContext] = None) -> List[str]:
        prompt = f"Generate 5-7 specific search queries for the research field: {field_name}\n\n"
        prompt += "Focus on finding:\n"
        prompt += "- Major recent developments and breakthroughs\n"
        prompt += "- Key debates and criticisms in the field\n"
        prompt += "- Important institutions and their work\n"
        prompt += "- Publication venues (conferences, journals)\n"
        
        if user_context and user_context.expertise_areas:
            prompt += f"\nUser expertise: {', '.join(user_context.expertise_areas)}\n"
            prompt += "Also query for how this expertise might connect to the field.\n"
        
        result = self._call_llm(
            "You are a research search expert. Generate targeted search queries.",
            prompt,
            response_format="json"
        )

        import json
        data = json.loads(result)
        return data.get("queries", [])

    def generate_user_model(self, user_background: str, field_context: FieldContext) -> UserContext:
        context_summary = field_context.to_prompt_context()
        print(f"[DEBUG] generate_user_model context_summary length: {len(context_summary)}")
        
        result = self._call_llm(
            SYSTEM_PROMPTS["user_model_generator"],
            f"User Background:\n{user_background}\n\nField Context:\n{context_summary}",
            response_format="json"
        )
        
        print(f"[DEBUG] user_model LLM result: {result[:300]}...")
        
        import json
        data = json.loads(result)
        
        print(f"[DEBUG] user_model data keys: {list(data.keys())}")
        
        user = UserContext()
        user.background = user_background
        
        expertise = data.get("expertise_areas") or data.get("expertise") or data.get("skills") or []
        if isinstance(expertise, dict):
            expertise = list(expertise.keys())
        user.expertise_areas = expertise if isinstance(expertise, list) else []
        
        gaps = data.get("known_gaps") or data.get("gaps") or data.get("weaknesses") or []
        if isinstance(gaps, dict):
            gaps = list(gaps.keys())
        user.known_gaps = gaps if isinstance(gaps, list) else []
        
        directions = data.get("suggested_directions") or data.get("research_directions") or data.get("directions") or data.get("suggested_research_directions") or []
        if isinstance(directions, list):
            suggested = []
            for d in directions:
                if isinstance(d, dict):
                    suggested.append(f"{d.get('title', '')}: {d.get('description', '')}")
                else:
                    suggested.append(str(d))
            directions = suggested
        user.suggested_directions = directions if isinstance(directions, list) else []
        
        user.current_focus = data.get("current_focus") or data.get("focus") or data.get("current_interest") or "Exploring options"
        
        return user

    def generate_response(self, user_query: str, field_context: FieldContext, 
                         user_context: UserContext, conversation_history: Optional[List[Dict]] = None) -> str:
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
        
        import json
        data = json.loads(result)
        
        return field_context, user_context
