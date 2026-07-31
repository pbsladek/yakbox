from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Machine-readable operations and formats supported by a speech backend."""

    name: str
    synthesis: bool
    transformation: bool
    streaming: bool
    hosted: bool
    output_formats: tuple[str, ...]
    max_text_characters: int | None = None
    supports_reference_voice: bool = False
    supports_hd: bool = False
