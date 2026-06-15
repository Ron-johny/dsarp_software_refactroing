# src/utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables from a .env file if it exists
load_dotenv()

class Settings:
    # API Configurations
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_LLM_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # # Genetic Algorithm Parameters # Later
    # MAX_GENERATIONS: int = 50
    # POPULATION_SIZE: int = 20
    # MUTATION_PROBABILITY: float = 0.5
    
    # # System Execution Limits # Veto Stuff
    # CBO_THRESHOLD: int = 25
    # WMC_THRESHOLD: int = 40
    # LOC_THRESHOLD: int = 1000

    @classmethod
    def validate_env(cls):
        """Ensures critical security assets exist before processing."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "CRITICAL RUNTIME ERROR: GEMINI_API_KEY environment variable is missing. "
                "Please configure it in your terminal or root .env file."
            )


settings = Settings()