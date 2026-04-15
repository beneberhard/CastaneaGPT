
import uuid
import json
import re
from dataclasses import dataclass
from typing import List, Dict

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

    def __init__(self):
        self.active_quizzes: Dict[str, List[Question]] = {}

    # -----------------------------------------
    # Retrieve knowledge from RAG
    # -----------------------------------------

    def retrieve_context(self, topic: str, top_k: int = 5):

        nodes = get_hybrid_retriever().retrieve(topic)

        context_blocks = []
        for n in nodes[:top_k]:
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

        context = self.retrieve_context(topic)

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

        prompt = f"""
You are an expert in forestry and chestnut cultivation.
Use ONLY the knowledge in the context below to generate quiz questions.
Avoid trivia such as dates or institutions.
Prefer conceptual forestry knowledge and management practices.

Question style examples:
- Which silvicultural system is traditionally used for managing Castanea sativa stands in southern Europe?
- What is a typical ecological advantage of coppice management in chestnut forests?
- Which symptom most clearly indicates infection by chestnut blight (Cryphonectria parasitica)?
- Why can chestnut trees respond well to coppicing after harvesting?
- Why is coppicing effective for chestnut stands?

Avoid questions about:
- dates
- institutions
- museums
- historical trivia

Focus on conceptual and applied forestry knowledge.

Context:
{context}

Task:
Generate {n_questions} {difficulty} quiz questions.

Topic:
{topic}

Format:
{format_type}

Language:
{language_instruction}

Return ONLY valid JSON.

Structure:

[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "...",
    "explanation": "..."
  }}
]
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
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

        for q in data:

            if not isinstance(q, dict):
                continue

            if not all(k in q for k in ["question", "options", "correct_answer", "explanation"]):
                continue

            question = Question(
                id=str(uuid.uuid4()),
                question=q["question"],
                options=q["options"],
                correct_answer=q["correct_answer"],
                explanation=q["explanation"]
            )

            questions.append(question)

        if len(questions) == 0:
            raise ValueError("No valid quiz questions generated")

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
