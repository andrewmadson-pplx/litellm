"""Perplexity Decisions API.

Perplexity reads images inside ``state`` as OpenAI-style ``image_url`` parts (base64 PNG, JPEG or WebP data
URLs, at most 8 per question) and rejects a top-level ``images`` field, so images are folded into ``state``
here. Message input keeps each image where the caller placed it among the text parts; images passed next to
a ``state`` value are appended after it. Normalize only separately supplied images and ``input_image`` parts:
lowercase data-URL labels, ``image/jpg`` rewritten as ``image/jpeg``, and standard base64. Leave raw state unchanged.
https://docs.perplexity.ai/api-reference/decisions-post
"""

from collections.abc import Mapping
from dataclasses import replace
from itertools import chain
from types import MappingProxyType
from typing import Final

from litellm.llms.base_llm.chat.transformation import with_attribution_headers
from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    image_media_type,
    input_images,
)
from litellm.types.decisions import (
    DecisionsBase64Image,
    DecisionsImage,
    DecisionsIRMessages,
    DecisionsIRRequest,
    DecisionsIRState,
    DecisionsJSON,
    OpenAIDecisionInputMessage,
    OpenAIDecisionInputText,
    UnsupportedDecisionsRequest,
)

_PERPLEXITY_ATTRIBUTION_HEADERS: Final[Mapping[str, str]] = MappingProxyType({"X-Pplx-Integration": "litellm"})
_DATA_URL_PREFIX: Final = "data:"
_CANONICAL_MEDIA_TYPES: Final[Mapping[str, str]] = MappingProxyType({"image/jpg": "image/jpeg"})
_URL_SAFE_TO_STANDARD_BASE64: Final = str.maketrans("-_", "+/")


def _canonical_data_url(media_type: str, base64_payload: str) -> str:
    canonical_type: Final = _CANONICAL_MEDIA_TYPES.get(media_type, media_type)
    return f"{_DATA_URL_PREFIX}{canonical_type};base64,{base64_payload.translate(_URL_SAFE_TO_STANDARD_BASE64)}"


def _image_data_url(image: DecisionsImage) -> str:
    if isinstance(image, DecisionsBase64Image):
        return _canonical_data_url(image_media_type(image), image.base64)
    media_type: Final = image_media_type(image)
    if media_type is None:
        return image
    return _canonical_data_url(media_type, image.partition(",")[2])


def _image_part(image: DecisionsImage) -> Mapping[str, object]:
    return {"type": "image_url", "image_url": {"url": _image_data_url(image)}}


def _text_part(text: str) -> Mapping[str, object]:
    return {"type": "text", "text": text}


def _message_parts(message: OpenAIDecisionInputMessage) -> tuple[Mapping[str, object], ...]:
    if isinstance(message.content, str):
        return (_text_part(message.content),)
    return tuple(
        _text_part(part.text) if isinstance(part, OpenAIDecisionInputText) else _image_part(part.image_url)
        for part in message.content
    )


def _state_parts(state: DecisionsJSON) -> tuple[object, ...]:
    if isinstance(state, str):
        return (_text_part(state),)
    if isinstance(state, Mapping):
        return (state,)
    return tuple(state)


def _state_with_images(decision_input: DecisionsIRState | DecisionsIRMessages) -> DecisionsIRState:
    if isinstance(decision_input, DecisionsIRMessages):
        return DecisionsIRState(state=tuple(chain.from_iterable(map(_message_parts, decision_input.messages))))
    return DecisionsIRState(state=(*_state_parts(decision_input.state), *map(_image_part, decision_input.images)))


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

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, str]:
        return super().validate_environment(
            headers=with_attribution_headers(_PERPLEXITY_ATTRIBUTION_HEADERS, dict(headers)),
            model=model,
            api_key=api_key,
            litellm_params=litellm_params,
        )

    def transform_decisions_request(
        self, model: str, request: DecisionsIRRequest, custom_llm_provider: str
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        if not input_images(request.input):
            return super().transform_decisions_request(
                model=model, request=request, custom_llm_provider=custom_llm_provider
            )
        return super().transform_decisions_request(
            model=model,
            request=replace(request, input=_state_with_images(request.input)),
            custom_llm_provider=custom_llm_provider,
        )
