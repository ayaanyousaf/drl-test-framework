import numpy as np

class RewardManager:
    """
    Reward shaping for LunarLander-v2 game.
    If no persona is provided, use "baseline", 
    meaning it uses the default env rewards from Gym.

    Basic Rewards: 
        - For both personas, penalize if they don't land between the flags

    Implements 2 personas: 
        Speedrunner
            - Focus on landing as fast as possible.
            - More reward for fast landing.
            - Less penalty for crash landing.

        Safe Lander: 
            - Focus on landing as safe as possible.
            - More reward for smooth and steady landing.
            - More penalty for crashes and fast landing.
    """

    def __init__(self, persona="baseline"):
        self.persona = persona
        self.prev_y_vel = 0.0
        self.prev_angle = 0.0
        self.crash_penalty_applied = False
        self.steps = 0

    def reset(self):
        self.prev_y_vel = 0.0
        self.prev_angle = 0.0
        self.crash_penalty_applied = False
        self.steps = 0

    def compute(self, info):
        """
        Compute general rewards.
        Call persona-specific methods and add to rewards
        `info` includes:
          - x_pos, y_pos
          - x_vel, y_vel
          - angle, angular_vel
          - leg1_contact, leg2_contact
          - landed, crashed
        """
        x_vel = info.get("x_vel", 0.0)
        y_vel = info.get("y_vel", 0.0)
        angle = info.get("angle", 0.0)
        landed = info.get("landed", False)
        crashed = info.get("crashed", False)

        reward = 0.0

        # -------- General Rewards (all personas) --------
        # Penalty for going off center too much so it knows where to land
        reward -= abs(info.get("x_pos", 0.0)) * 0.1

        # Reward for landing
        if landed: 
            reward += 5.0

        # Persona based rewards
        if self.persona == "speedrunner":
            reward += self.speedrunner_reward(x_vel, y_vel, crashed)
        elif self.persona == "safe":
            reward += self.safe_lander_reward(x_vel, y_vel, angle, crashed)

        # Time penalty so the agent can't stall too long
        reward -= 0.01

        self.prev_y_vel = y_vel
        self.prev_angle = angle
        self.steps += 1

        return float(reward)

    def speedrunner_reward(self, x_vel, y_vel, crashed):
        """
        Speedrunner persona:
        - Prioritize faster landing.
        - Less penalty for crashing.
        - More reward for speed
        """
        reward = 0.0

        # Reward for descending faster, but not too fast
        if y_vel < -0.5:
            reward += 0.2
        elif y_vel < -2.0: 
            reward -= 0.2
        elif y_vel > -0.2: 
            reward -= 2.0 # large penalty for going up or descending slow


        reward += max(0, -y_vel) * 0.6 # reward for staying fast vertically
        reward += abs(x_vel) * 0.02 # small reward for moving fast horizontally
        reward += (1.0 - min(self.steps / 1000, 1.0)) * 0.5 # reward for landing in fewer steps

        # Small penalty for crashing
        if crashed and not self.crash_penalty_applied:
            reward -= 1.0
            self.crash_penalty_applied = True

        return reward

    def safe_lander_reward(self, x_vel, y_vel, angle, crashed):
        """
        Safe Lander persona:
        - Reward for lander slow and steady
        - More penalty for crashing.
        - More penalty for being too fast.
        """
        reward = 0.0

        # Penalize fast horizontal movement
        reward -= abs(x_vel) * 0.3

        # Reward safe and steady descent
        if -0.4 < y_vel < -0.1:
            reward += 0.2
        elif y_vel > -0.1: 
            reward -= 0.1 # penalty for moving up or hovering too much
        else:
            reward -= abs(y_vel) * 0.1  # penalty for falling to fast

        # Penalize lander tilting (less stable)
        reward -= abs(angle) * 0.2

        # Large penalty for crashing
        if crashed and not self.crash_penalty_applied:
            reward -= 2.0
            self.crash_penalty_applied = True

        return reward
