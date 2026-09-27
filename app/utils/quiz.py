
import uuid
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from app.rag.rag import get_hybrid_retriever

load_dotenv()

client = OpenAI()


# -----------------------------------------
# Data structure
# -----------------------------------------

@dataclass
class Question:
    id: str
    question: str
    options: List[str]
    correct_answer: str
    explanation: str


# -----------------------------------------
# Quiz engine
# -----------------------------------------

import json
import re
import uuid
from typing import Dict, List


class QuizEngine:

    OPTION_LETTERS = ["A", "B", "C", "D"]
    TOPIC_CONFIG = {
        "general chestnut knowledge": {
            "label": "General",
            "retrieval_query": (
                "general chestnut knowledge Castanea sativa chestnut forests "
                "management ecology silviculture pathology economics cultural role"
            ),
            "focus_instruction": (
                "Cover broad chestnut knowledge across ecology, management, uses, risks, and cultural relevance."
            ),
        },
        "chestnut silviculture": {
            "label": "Silviculture",
            "retrieval_query": (
                "chestnut silviculture Castanea sativa coppice high forest stand treatment "
                "regeneration thinning pruning rotation management"
            ),
            "focus_instruction": (
                "Focus on stand structure, regeneration, tending, thinning, coppice systems, and silvicultural decisions."
            ),
        },
        "chestnut forest operations": {
            "label": "Forest Operations",
            "retrieval_query": (
                "chestnut forest operations harvesting extraction logging access machinery "
                "worksite planning safety assortments transport"
            ),
            "focus_instruction": (
                "Focus on harvesting, extraction, operational planning, timber assortments, access, and worksite constraints."
            ),
        },
        "chestnut pests and diseases": {
            "label": "Pests and diseases",
            "retrieval_query": (
                "chestnut pests and diseases pathology blight ink disease gall wasp "
                "symptoms diagnosis prevention management"
            ),
            "focus_instruction": (
                "Focus on pests, pathogens, symptoms, diagnosis, prevention, and management responses."
            ),
        },
        "chestnut ecology": {
            "label": "Ecology",
            "retrieval_query": (
                "chestnut ecology Castanea sativa habitat soils climate biodiversity "
                "ecosystem site suitability resilience"
            ),
            "focus_instruction": (
                "Focus on habitat, soils, climate, biodiversity, ecosystem dynamics, and environmental suitability."
            ),
        },
        "chestnut economics": {
            "label": "Economics",
            "retrieval_query": (
                "chestnut economics value chain production timber fruit market costs labor "
                "rural income management economics"
            ),
            "focus_instruction": (
                "Focus on production value, market uses, costs, labor, revenues, and economic trade-offs."
            ),
        },
        "chestnut cultural role": {
            "label": "Cultural role",
            "retrieval_query": (
                "chestnut cultural role heritage tradition rural landscape food culture "
                "community historical importance chestnut"
            ),
            "focus_instruction": (
                "Focus on heritage, traditions, landscape identity, food culture, and social importance."
            ),
        },
    }

    def __init__(self):
        self.active_quizzes: Dict[str, List[Question]] = {}
        self.recent_question_history: Dict[str, List[str]] = {}
        self.quiz_generation_counts: Dict[str, int] = {}

    def _settings_key(
        self,
        topic: str,
        difficulty: str,
        format_type: str,
        language: str,
    ) -> str:
        return "|".join(
            [
                str(topic).strip().lower(),
                str(difficulty).strip().lower(),
                str(format_type).strip().lower(),
                str(language).strip().lower(),
            ]
        )

    def _question_signature(self, question_text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(question_text or "").strip().lower())
        normalized = re.sub(r"[^a-z0-9àèéìíîòóùüäößç ]+", "", normalized)
        return normalized[:220]

    def _split_option_label(self, option: str) -> tuple[Optional[str], str]:
        text = str(option or "").strip()
        match = re.match(r"^([A-D])[\.\):\-]?\s*(.+)$", text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), match.group(2).strip()
        return None, text

    def _normalize_options(self, options: List[str]) -> List[str]:
        normalized = []
        for idx, raw_option in enumerate(options[: len(self.OPTION_LETTERS)]):
            _, option_text = self._split_option_label(raw_option)
            option_text = option_text.strip()
            if not option_text:
                continue
            normalized.append(f"{self.OPTION_LETTERS[idx]}. {option_text}")
        return normalized

    def _normalize_correct_answer(self, correct_answer: str, options: List[str]) -> str:
        raw_answer = str(correct_answer or "").strip()
        if not raw_answer:
            raise ValueError("Missing correct answer")

        option_map = {}
        text_map = {}
        for option in options:
            label, option_text = self._split_option_label(option)
            if label:
                option_map[label] = option
            text_map[option_text.casefold()] = option

        answer_label, answer_text = self._split_option_label(raw_answer)

        if answer_label and answer_label in option_map:
            return option_map[answer_label]

        if answer_text.casefold() in text_map:
            return text_map[answer_text.casefold()]

        raise ValueError(f"Correct answer does not match quiz options: {correct_answer}")

    # -----------------------------------------
    # Retrieve knowledge from RAG
    # -----------------------------------------

    def _topic_settings(self, topic: str) -> dict:
        return self.TOPIC_CONFIG.get(topic, self.TOPIC_CONFIG["general chestnut knowledge"])

    def retrieve_context(self, topic: str, variant_index: int = 0, top_k: int = 5):
        topic_settings = self._topic_settings(topic)
        retrieval_query = topic_settings["retrieval_query"]

        # Topic queries already contain detailed search terms; skip LLM expansion.
        nodes = get_hybrid_retriever(expand_queries=False).retrieve(retrieval_query)

        node_list = list(nodes)
        if not node_list:
            return ""

        pool_size = min(len(node_list), max(top_k * 3, top_k))
        candidate_nodes = node_list[:pool_size]
        start = 0 if pool_size <= top_k else (variant_index * 2) % pool_size

        selected_nodes = []
        used_texts = set()
        for step in range(pool_size):
            node = candidate_nodes[(start + step) % pool_size]
            try:
                node_text = node.node.text
            except Exception:
                node_text = str(node)
            signature = self._question_signature(node_text[:300])
            if signature in used_texts:
                continue
            used_texts.add(signature)
            selected_nodes.append(node)
            if len(selected_nodes) >= top_k:
                break

        context_blocks = []
        for n in selected_nodes:
            try:
                context_blocks.append(n.node.text)
            except Exception:
                context_blocks.append(str(n))

        return "\n\n".join(context_blocks)

    # -----------------------------------------
    # Helper: safely extract JSON from LLM
    # -----------------------------------------

    def _extract_json(self, content: str):

        try:
            return json.loads(content)
        except Exception:
            pass

        # Try extracting JSON array
        match = re.search(r"\[.*\]", content, re.S)
        if match:
            return json.loads(match.group())

        # Try extracting JSON object
        match = re.search(r"\{.*\}", content, re.S)
        if match:
            return json.loads(match.group())

        raise ValueError("No valid JSON found in LLM response")

    # -----------------------------------------
    # Generate quiz
    # -----------------------------------------

    def generate(
        self,
        topic: str = "general chestnut knowledge",
        difficulty: str = "intermediate",
        n_questions: int = 5,
        format_type: str = "multiple_choice",
        language: str = "auto"
    ):

        topic_settings = self._topic_settings(topic)
        settings_key = self._settings_key(topic, difficulty, format_type, language)
        variant_index = self.quiz_generation_counts.get(settings_key, 0)
        context = self.retrieve_context(topic, variant_index=variant_index)
        recent_questions = self.recent_question_history.get(settings_key, [])[-10:]

        lang_map = {
            "en": "English",
            "it": "Italiano",
            "de": "Deutsch"
        }

        lang_name = lang_map.get(language, "English")

        if language == "auto":
            language_instruction = "Use the same language as the user interface."
        else:
            language_instruction = f"""
The entire quiz (questions, options, and explanations) must be written in {lang_name}.
Do not use any other language.
"""

        avoid_repeat_instruction = ""
        if recent_questions:
            formatted_recent = "\n".join(f"- {question}" for question in recent_questions)
            avoid_repeat_instruction = f"""
Avoid repeating or lightly rephrasing these recent quiz questions for the same settings:
{formatted_recent}

The new questions must test different aspects, scenarios, or source details.
"""

        prompt = f"""
You are an expert in forestry and chestnut cultivation.
Use ONLY the knowledge in the context below to generate quiz questions.
Prefer conceptual forestry knowledge and management practices.
Avoid overusing any single regional case study or country background across the quiz.

Avoid questions about:
- dates
- institutions
- museums
- historical trivia

Focus on conceptual and applied forestry knowledge.
For this quiz, prioritize the following thematic focus:
{topic_settings["focus_instruction"]}
{avoid_repeat_instruction}

Context:
{context}

Task:
Generate {n_questions} {difficulty} quiz questions.

Topic:
{topic_settings["label"]}

Format:
{format_type}

Language:
{language_instruction}

Return ONLY valid JSON.

Structure:

[
  {{
    "question": "...",
    "options": ["A. first option", "B. second option", "C. third option", "D. fourth option"],
    "correct_answer": "A. first option",
    "explanation": "..."
  }}
]
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You generate quiz questions."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content

        # -----------------------------------------
        # Robust JSON parsing
        # -----------------------------------------

        try:
            parsed = self._extract_json(content)

            if isinstance(parsed, dict) and "questions" in parsed:
                data = parsed["questions"]
            else:
                data = parsed

        except Exception as e:
            print("LLM response was:", content)
            raise ValueError(f"LLM returned invalid JSON: {e}")

        # -----------------------------------------
        # Validate and build questions
        # -----------------------------------------

        questions = []
        fallback_questions = []
        recent_signatures = set(recent_questions)

        for q in data:

            if not isinstance(q, dict):
                continue

            if not all(k in q for k in ["question", "options", "correct_answer", "explanation"]):
                continue

            normalized_options = self._normalize_options(q["options"])
            if len(normalized_options) < 2:
                continue

            try:
                normalized_correct_answer = self._normalize_correct_answer(
                    q["correct_answer"],
                    normalized_options,
                )
            except ValueError:
                continue

            question = Question(
                id=str(uuid.uuid4()),
                question=q["question"],
                options=normalized_options,
                correct_answer=normalized_correct_answer,
                explanation=q["explanation"]
            )

            signature = self._question_signature(question.question)
            if signature in recent_signatures:
                fallback_questions.append(question)
            else:
                questions.append(question)

        if len(questions) < n_questions:
            for question in fallback_questions:
                questions.append(question)
                if len(questions) >= n_questions:
                    break

        if len(questions) == 0:
            raise ValueError("No valid quiz questions generated")

        stored_signatures = self.recent_question_history.get(settings_key, [])
        stored_signatures.extend(
            self._question_signature(question.question) for question in questions
        )
        self.recent_question_history[settings_key] = stored_signatures[-24:]
        self.quiz_generation_counts[settings_key] = variant_index + 1

        quiz_id = str(uuid.uuid4())
        self.active_quizzes[quiz_id] = questions

        return {
            "quiz_id": quiz_id,
            "topic": topic,
            "difficulty": difficulty,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "options": q.options
                }
                for q in questions
            ]
        }

    # -----------------------------------------
    # Grade quiz
    # -----------------------------------------

    def grade(self, quiz_id: str, user_answers: Dict[str, str]):

        questions = self.active_quizzes.get(quiz_id)

        if not questions:
            return {"error": "Quiz not found"}

        score = 0
        results = []


        for q in questions:

            user_answer = user_answers.get(str(q.id))

            if user_answer:
                user_letter = user_answer.strip()[0]
                #user_letter = user_answer.split(".")[0].strip()
            else:
                user_letter = None

            correct = user_answer == q.correct_answer

            if correct:
                score += 1

            results.append({
                "question": q.question,
                "your_answer": user_answer,
                "correct_answer": q.correct_answer,
                "is_correct": correct,
                "explanation": q.explanation
            })

        return {
            "score": score,
            "total": len(questions),
            "results": results
        }


quiz = QuizEngine()
