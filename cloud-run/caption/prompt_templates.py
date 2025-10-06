def build_editorial_prompt(scene_description: str, transcript: str) -> str:
    """
    Constructs a Getty-style editorial prompt for the Qwen model.
    """
    return (
        "Generate a Getty-style editorial caption for a video scene.\n"
        f"Scene: {scene_description}\n"
        f"Transcript: {transcript}\n"
        "Return: title, caption, summary, tags, licensing_flags\n"
        "Tone: editorial, descriptive, non-commercial"
    )


def build_multilingual_prompt(scene_description: str, transcript: str, language: str = "English") -> str:
    """
    Constructs a multilingual editorial prompt for future expansion.
    """
    return (
        f"Generate a Getty-style editorial caption for a video scene in {language}.\n"
        f"Scene: {scene_description}\n"
        f"Transcript: {transcript}\n"
        "Return: title, caption, summary, tags, licensing_flags\n"
        "Tone: editorial, descriptive, non-commercial"
    )


def build_debug_prompt(scene_description: str, transcript: str) -> str:
    """
    Constructs a verbose prompt for debugging model output structure.
    """
    return (
        "You are an editorial captioning assistant. Given a scene and transcript, return structured metadata.\n"
        f"Scene: {scene_description}\n"
        f"Transcript: {transcript}\n"
        "Output format:\n"
        "{\n"
        "  title: ..., \n"
        "  caption: ..., \n"
        "  summary: ..., \n"
        "  tags: [...], \n"
        "  licensing_flags: [...]\n"
        "}"
    )

