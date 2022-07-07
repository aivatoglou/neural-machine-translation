from pydantic import BaseSettings


class Settings(BaseSettings):

    """Read credentials from environment variables when in production."""

    """import os; os.getenv('SECRET_KEY');"""

    app_name: str = ""
    mongo_usr: str = ""
    mongo_pwd: str = ""
    mongo_clstr: str = ""
    mongo_db: str = ""
    trans_col: str = ""
    users_col: str = ""
    mongo_url: str = ""
    secret_key: str = ""


settings = Settings()
