import numpy as np

class RewardManager:
    def __init__(self, persona="functional"):
        self.persona = persona
        self.reset()

    def reset(self):
        self.prev_page_count = 0
        self.prev_successes = 0
        self.prev_errors = 0

    def compute(self, info: dict) -> float:
        """
        Compute reward based on persona.
        """

        persona = info.get("persona", self.persona)
        success = info.get("success", 0)
        error = info.get("error", 0)
        latency = info.get("latency", 0)
        visited_pages = info.get("visited_pages", set())
        touched = info.get("touched_selectors", set())
        step = info.get("step", 0)

        reward = 0.0

        # ----- General rewards ------ 
        if not info.get("logged_in", False):
            reward -= 0.5

        # Reward for logging in first
        if info.get("page") == "login" and info.get("success"):
            reward += 3.0

        # Reward for following a sequence of pages
        if info.get("logged_in", False) and "cart" in info.get("page", ""):
            reward += 2.0

        if "checkout" in info.get("page", ""):
            reward += 3.0

        if "finish" in info.get("page", ""):
            reward += 5.0

        # Encourage the agent to avoid repeating the same page
        if info.get("page") == getattr(self, "last_page", None):
            reward -= 0.2 
        self.last_page = info.get("page")
        
        # ----- Persona-specific rewards ----- 
        if persona == "functional":
            reward += 5 * success
            reward -= 3 * error
            reward -= 0.05 * latency

            if success > self.prev_successes:
                reward += 0.5

        elif persona == "explorer":
            new_pages = len(visited_pages) - self.prev_page_count

            reward += 1.5 * new_pages
            reward += 0.5 * len(touched) / 20.0
            reward += 1.5 * error

            if new_pages == 0 and step > 5:
                reward -= 1.0

        self.prev_page_count = len(visited_pages)
        self.prev_successes = success
        self.prev_errors = error

        return float(np.clip(reward, -10.0, 10.0))
