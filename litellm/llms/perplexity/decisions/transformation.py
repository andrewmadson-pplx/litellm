"""Perplexity Decisions API.

Perplexity reads images in place inside ``state`` as OpenAI-style ``image_url`` parts
(base64 data URLs, at most 8 per question) and rejects a top-level ``images`` field,
so images are folded into ``state`` here. https://docs.perplexity.ai/api-reference/decisions-post
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    image_media_type,
    input_images,
    systemone_state,
)
from litellm.types.decisions import (
    DecisionsBase64Image,
    DecisionsImage,
    DecisionsIRRequest,
    DecisionsIRState,
    DecisionsJSON,
    UnsupportedDecisionsRequest,
)

_DATA_URL_PREFIX: Final = "data:"


def _image_data_url(image: DecisionsImage) -> str:
    if isinstance(image, DecisionsBase64Image):
        return f"{_DATA_URL_PREFIX}{image_media_type(image)};base64,{image.base64}"
    return image


def _image_part(image: DecisionsImage) -> Mapping[str, object]:
    return {"type": "image_url", "image_url": {"url": _image_data_url(image)}}


def _state_parts(state: DecisionsJSON) -> tuple[object, ...]:
    if isinstance(state, str):
        return ({"type": "text", "text": state},)
    if isinstance(state, Mapping):
        return (state,)
    return tuple(state)


class PerplexityDecisionsConfig(BaseDecisionsConfig):
    path = "/v1/decisions"
    api_key_env = ("PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY")
    api_base_env = ("PERPLEXITY_API_BASE",)
    supports_images = True
    max_images = 8
    image_media_types = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})
    inlines_remote_images = True

    def get_default_api_base(self) -> str | None:
        return "https://api.perplexity.ai"

    def transform_decisions_request(
        self, model: str, request: DecisionsIRRequest, custom_llm_provider: str
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        images: Final = input_images(request.input)
        if not images:
            return super().transform_decisions_request(
                model=model, request=request, custom_llm_provider=custom_llm_provider
            )
        state: Final = DecisionsIRState(
            state=(*_state_parts(systemone_state(request.input)), *map(_image_part, images))
        )
        return super().transform_decisions_request(
            model=model, request=replace(request, input=state), custom_llm_provider=custom_llm_provider
        )
