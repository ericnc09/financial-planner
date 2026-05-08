"""Data models package — ORM tables and dataclass schemas."""

# Import registry so its ORM table is registered with Base.metadata before
# init_db() runs create_all(). Without this, the model_artifacts table would
# only be created when something else imports src.models.registry first.
from src.models import registry  # noqa: F401
