import os
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .reward import RewardManager


class SwagLabsEnv(gym.Env): 
    """
    Swag Labs automated DRL testing environment.
    This environment uses Selenium to interact with the Swag Labs web application,
    allowing agents to perform actions and receive observations and rewards.
    """
    def __init__(self, persona="functional", url="https://www.saucedemo.com/"):
        super().__init__()

        self.persona = persona
        self.url = url
        self.driver = None # will choose later

        self.max_steps = 25
        self.current_step = 0

        self.visited_pages = set()
        self.touched_selectors = set()
        self.latencies = []
        self.validation_errors = 0
        self.successes = 0
        self.logged_in = False

        # We will define 8 discrete actions for the agent to choose from
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.reward_manager = RewardManager(persona=self.persona)
                                            
    
    def set_driver(self):
        """
        Automatically chooses the right driver based on browser. 
        Detects Chrome, Edge, or Firefox.
        """

        try:
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager

            options = ChromeOptions()

            # ------ Uncomment these lines to run in headless mode ------ 
            #options.add_argument("--headless")
            #options.add_argument("--no-sandbox")
            #options.add_argument("--disable-dev-shm-usage")

            print("Using Chrome WebDriver")

            service = ChromeService(ChromeDriverManager().install())
            return webdriver.Chrome(service=service, options=options)

        except Exception as e:
            print(f"Error: Chrome not available ({e}), if you want to use a different browser, please modify the set_driver() method in envs/swaglabs/env.py")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Restart the web driver for a new episode
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

        self.driver = self.set_driver()

        if not self.driver:
            self.driver = self.set_driver()

        if self.reward_manager:
            self.reward_manager.reset()

        # Reset metrics
        self.visited_pages.clear()
        self.touched_selectors.clear()
        self.latencies.clear()
        self.validation_errors = 0
        self.successes = 0

        obs = np.zeros(3, dtype=np.float32)
        info = {"persona": self.persona, "page": "home", "success": False, "error": False}

        return obs, info

    def login(self):
        """
        Logs in to the Swag Labs website using admin credentials.
        """

        self.driver.get(self.url)
        time.sleep(1)  # small initial pause for redirects

        try:
            # Fetch login elements (input fields and button)
            username = self.driver.find_element(By.ID, "user-name")
            password = self.driver.find_element(By.ID, "password")
            login_button = self.driver.find_element(By.ID, "login-button")

            # Attempt logging in with standard credentials
            username.send_keys("standard_user")
            password.send_keys("secret_sauce")
            login_button.click()

            # Wait until user logs in and inventory page loads
            WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "inventory_list")))

            self.logged_in = True
            print("Login successful.")

        except Exception as e:
            # If element isn't found, assume already logged in or redirected
            print(f"Login failed or already logged in: {e}.")

    def perform_action(self, action):
        """
        Perform actions in Swag Labs based on the action index.
        Each agent can perform these 8 actions: 

        0 - Login to the website
        1 - Add item to cart
        2 - Remove item from cart
        3 - Go to cart page
        4 - Proceed to checkout
        5 - Fill in checkout information
        6 - Finish purchase
        7 - Logout of the website
        """

        page_name = "unknown"
        success = 0.0
        error = 0.0

        try:
            if action == 0:
                self.login()
                success = 1.0

            elif action == 1:   # Add item to cart
                items = self.driver.find_elements(By.CLASS_NAME, "inventory_item")

                if items:
                    # Pick an unseen item in inventory
                    available_indices = [i for i in range(len(items)) if i not in self.touched_selectors]

                    if not available_indices:  # all items have been seen, allow repeats
                        available_indices = list(range(len(items)))

                    idx = random.choice(available_indices)

                    self.touched_selectors.add(idx)

                    btn = items[idx].find_element(By.CLASS_NAME, "btn_inventory")
                    btn.click()

                    page_name = f"add_item_{idx}"
                    success = 1.0
                else:
                    error = 1.0

            elif action == 2:   # Remove item from cart
                remove_buttons = self.driver.find_elements(By.CLASS_NAME, "btn_secondary")

                if remove_buttons:
                    random.choice(remove_buttons).click()
                    page_name = "remove_random"
                    success = 1.0
                else:
                    error = 1.0

            elif action == 3:   # Go to cart page
                self.driver.find_element(By.CLASS_NAME, "shopping_cart_link").click()
                page_name = "cart"
                success = 1.0

            elif action == 4:   # Proceed to checkout
                self.driver.find_element(By.ID, "checkout").click()
                page_name = "checkout"
                success = 1.0

            elif action == 5:   # Fill in checkout information
                self.driver.find_element(By.ID, "first-name").send_keys("John")
                self.driver.find_element(By.ID, "last-name").send_keys("Doe")
                self.driver.find_element(By.ID, "postal-code").send_keys("A1B2C3")
                self.driver.find_element(By.ID, "continue").click()
                page_name = "checkout_info"
                success = 1.0

            elif action == 6:   # Finish purchase
                self.driver.find_element(By.ID, "finish").click()
                page_name = "finish"
                success = 1.0

            elif action == 7:   # Logout of the website
                self.driver.find_element(By.ID, "react-burger-menu-btn").click()
                time.sleep(0.5)
                self.driver.find_element(By.ID, "logout_sidebar_link").click()
                self.logged_in = False
                page_name = "logout"
                success = 1.0

            time.sleep(0.5)

        except Exception as e:
            error = 1.0
            print(f"Action {action} failed: {e}")

        return page_name, success, error
    
    def step(self, action):
        """
        Takes a step in the environment using the given action.
        Auto-collects metrics in info dictionary for RewardManager to use.

        Returns: 
            obs: observation vector of the next state.
            reward: a float reward value (shaped in RewardManager) for the step.
            terminated: a boolean that tracks if an episode has ended.
            truncated: a boolean that tracks if an episode reached its time limit.
            info: contains episode metrics and action info.
        """

        self.current_step += 1
        terminated = False
        truncated = False

        start_time = time.time()

        # Choose action based on module or page (Customer or ToDo)
        page_name, success, error = self.perform_action(action)

        # Update metrics
        latency = time.time() - start_time
        self.latencies.append(latency)
        self.visited_pages.add(page_name)

        if success:
            self.successes += 1
        if error:
            self.validation_errors += 1

        # Termination conditions (when to end episode)
        if self.current_step >= self.max_steps:
            truncated = True
        if self.validation_errors > 3:
            terminated = True

        # Create the observation vector
        obs = np.array([
            len(self.visited_pages) / 10.0,
            float(success),
            float(error)
        ], dtype=np.float32)

        # Create the info dictionary
        info = {
            "persona": self.persona,
            "page": page_name,
            "success": bool(success),
            "error": bool(error),
            "latency": latency,
            "visited_pages": self.visited_pages,
            "touched_selectors": self.touched_selectors,
            "validation_errors": self.validation_errors,
            "successes": self.successes,
            "step": self.current_step
        }

        # Calculate the reward using RewardManager
        reward = self.reward_manager.compute(info)

        return obs, reward, terminated, truncated, info

    def close(self):
        """
        Closes the environment and quits the WebDriver.
        """
        
        if self.driver:
            try:
                self.driver.quit()
                print("Driver closed successfully.")
            except Exception as e:
                print(f"Error closing driver: {e}")
            finally:
                self.driver = None