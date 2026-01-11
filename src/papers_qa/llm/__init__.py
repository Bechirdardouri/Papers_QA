"""LLM inference module."""

from threading import Thread
from typing import Any, Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)

from papers_qa.config import get_settings
from papers_qa.logging_config import get_logger

logger = get_logger(__name__)


class LLMModel:
    """Wrapper for LLM inference."""

    def __init__(self, model_name: str = "", device: str = "cuda") -> None:
        """Initialize LLM model.

        Args:
            model_name: Model identifier from Hugging Face.
            device: Device to use (cuda, cpu, mps).
        """
        settings = get_settings()
        self.model_name = model_name or settings.model.generation_model
        self.device = device or settings.model.generation_device

        logger.info(
            "loading_llm_model",
            model=self.model_name,
            device=self.device,
        )

        # Load with quantization if enabled
        if settings.model.enable_quantization:
            self._load_quantized()
        else:
            self._load_standard()

        logger.info("llm_model_loaded", model=self.model_name)

    def _load_quantized(self) -> None:
        """Load model with 4-bit quantization."""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("model_loaded_with_4bit_quantization")

    def _load_standard(self) -> None:
        """Load model without quantization."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
        )

    def generate(
        self,
        prompt: str,
        max_length: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text.

        Args:
            prompt: Input prompt.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional generation parameters.

        Returns:
            str: Generated text.
        """
        settings = get_settings()
        max_length = max_length or settings.model.generation_max_length
        temperature = temperature or settings.model.generation_temperature
        top_p = top_p or settings.model.generation_top_p

        logger.debug("generating_text", prompt_length=len(prompt))

        try:
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            )

            outputs = pipe(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs,
            )

            return outputs[0]["generated_text"]

        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise

    def generate_streaming(
        self,
        prompt: str,
        max_length: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Generate text with streaming.

        Args:
            prompt: Input prompt.
            max_length: Maximum length.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional parameters.

        Yields:
            str: Generated text chunks.
        """
        settings = get_settings()
        max_length = max_length or settings.model.generation_max_length
        temperature = temperature or settings.model.generation_temperature
        top_p = top_p or settings.model.generation_top_p

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "streamer": streamer,
            **kwargs,
        }

        # Run generation in separate thread
        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        # Yield generated text
        for text in streamer:
            yield text


class InferencePipeline:
    """Complete inference pipeline."""

    def __init__(self, model_name: str = "") -> None:
        """Initialize inference pipeline.

        Args:
            model_name: Model identifier.
        """
        self.llm = LLMModel(model_name)

    def answer_question(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> str:
        """Answer a question given context.

        Args:
            question: Question text.
            context: Context passage.
            system_prompt: System prompt (optional).

        Returns:
            str: Generated answer.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful medical AI assistant. "
                "Answer the question based on the provided context. "
                "Be concise and accurate."
            )

        prompt = f"""{system_prompt}

Context: {context}

Question: {question}

Answer:"""

        return self.llm.generate(prompt)

    def answer_question_streaming(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """Answer question with streaming.

        Args:
            question: Question text.
            context: Context passage.
            system_prompt: System prompt.

        Yields:
            str: Generated answer chunks.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful medical AI assistant. "
                "Answer the question based on the provided context. "
                "Be concise and accurate."
            )

        prompt = f"""{system_prompt}

Context: {context}

Question: {question}

Answer:"""

        yield from self.llm.generate_streaming(prompt)
