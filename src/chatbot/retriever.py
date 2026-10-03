import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base.json")

class LeafDoctorChatbot:
    def __init__(self, kb_path: str = KB_PATH):
        if os.path.exists(kb_path):
            with open(kb_path, "r") as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = {}

    def get_advisory(self, predicted_class: str, user_query: str = "") -> dict:
        info = self.knowledge_base.get(predicted_class, None)
        
        if not info:
            return {
                "advisory": f"No specific knowledge base entry found for {predicted_class}.",
                "citation": "General Agriculture Knowledge Base"
            }

        query_lower = user_query.lower().strip()

        # 1. Prevention query
        if any(word in query_lower for word in ["prevent", "prevention", "avoid", "stop", "protect"]):
            advisory_text = f"Prevention Guidelines for {predicted_class}: {info.get('prevention', 'No prevention data available.')}"
            
        # 2. Cause query
        elif any(word in query_lower for word in ["cause", "why", "reason", "source", "origin"]):
            advisory_text = f"Cause of {predicted_class}: {info.get('cause', 'No cause data available.')}"
            
        # 3. Treatment query
        elif any(word in query_lower for word in ["treat", "treatment", "cure", "heal", "fungicide", "spray", "medicine", "fix"]):
            advisory_text = f"Treatment Guidance for {predicted_class}: {info.get('treatment', 'No treatment data available.')}"
            
        # 4. Default / General query
        else:
            advisory_text = (
                f"Advisory for {predicted_class}:\n"
                f"• Cause: {info.get('cause', 'N/A')}\n"
                f"• Treatment: {info.get('treatment', 'N/A')}\n"
                f"• Prevention: {info.get('prevention', 'N/A')}"
            )

        return {
            "advisory": advisory_text,
            "citation": info.get("citation", "Agriculture Knowledge Base")
        }


# Global helper instance / function if referenced directly
_chatbot = LeafDoctorChatbot()

def get_advisory(predicted_class: str, user_query: str = "") -> dict:
    return _chatbot.get_advisory(predicted_class, user_query)