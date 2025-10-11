import pydantic as pdt
import pydantic_settings as pdts

import labelai.jobs as jobs


class Settings(pdts.BaseSettings, strict=True, frozen=True, extra="forbid"):
    """Base class for settings."""

    pass


class JobSettings(Settings):
    """Job settings for the project.

    Args:
        job (jobs.JobKind): The job to run.
    """

    job: jobs.JobKind = pdt.Field(..., discriminator="KIND")
