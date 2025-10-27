import numpy as np

class RewardManager:
    def __init__(self, persona="functional"):
        self.persona = persona
        self.reset()

    def reset(self):
        self.prev_page_count = 0
        self.prev_successes = 0
        self.prev_errors = 0
        self.logged_in_once = False 
        self.last_page = None
        self.visited_pages = set() 

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
        logged_in = info.get("logged_in", False)
        action = info.get("action", None)
        page = info.get("page", "")

        reward = 0.0

        # ----- General rewards ------ 

        # Penalty for trying to log in again when already logged in
        if logged_in and action == 0:
            reward -= 3.0 + 0.05 * step 

        # Penalty for not being logged in after a few steps
        if not logged_in and step > 3:
            reward -= 0.5

        # Reward for logging in first
        if page == "login" and info.get("success") and not self.logged_in_once:
            reward += 3.0
            self.logged_in_once = True
        elif page == "login" and self.logged_in_once:
            reward -= 2.0 + 0.1 * step # penalize repeated logins to avoid spam

        # Encourage the agent to avoid staying on the same page
        if page == self.last_page:
            reward -= 1.0 + 0.05 * max(0, step - 1)
        else:
            self.last_page = page

        # ----- Persona-specific rewards ----- 
        if persona == "functional":
            reward += 2.0 * success
            reward -= 3 * error
            reward -= 0.05 * latency

            # Reward for following a sequence of pages (functional wants to follow the purchase flow)
            if logged_in and "cart" in page and "cart" not in self.visited_pages:
                reward += 1.0
                self.visited_pages.add("cart")

            if "checkout" in page and "checkout" not in self.visited_pages:
                reward += 4.0
                self.visited_pages.add("checkout")

            if "finish" in page and "finish" not in self.visited_pages:
                reward += 12.0
                self.visited_pages.add("finish")

        elif persona == "explorer":
            new_pages = len(visited_pages) - self.prev_page_count

            reward += 1.5 * new_pages
            reward += 0.5 * len(touched) / 20.0
            reward += 0.5 * min(error, 1)

            if new_pages == 0 and step > 5:
                reward -= 1.0

            if "finish" in visited_pages:
                reward += 3.0

        # Small reward for exploring new pages
        reward += 0.05 * (len(visited_pages))

        self.prev_page_count = len(visited_pages)
        self.prev_successes = success
        self.prev_errors = error

        return float(np.clip(reward, -10.0, 10.0))
