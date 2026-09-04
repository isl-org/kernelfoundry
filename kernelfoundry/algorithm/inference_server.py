"""Inference classes for submitting LLM queries."""

import sys
import os
import threading
import time
import uuid
import warnings
import concurrent
import requests
import getpass
import hydra
import numpy as np
import logging

# API clients
from openai import OpenAI
import openai

try:
    import anthropic
except ImportError:  # optional dependency for Anthropic support
    anthropic = None
import boto3
import botocore

# Treat provider max_tokens stop reasons as truncated completions.
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "length"})

# Fallback models used when a config asks for model_name="default". This is NOT a
# validation list: an explicitly named model is passed through to the provider, so any
# current model works without editing this file. Provider catalogues change often --
# check the provider's own model list rather than treating this as authoritative.
DEFAULT_MODELS = {
    "openai": [
        "gpt-4.1",
        "gpt-4o",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    "anthropic": [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
    ],
}


class InferenceServer:
    """Class for sending queries to an LLM API."""

    def __init__(
        self,
        server_type: str = None,
        model_name: str = "default",
        greedy_sample: bool = False,
        verbose: bool = False,
        timeout: int = 400,
        max_retry: int = 5,
        **kwargs,
    ):
        """Initialize an inference server for LLM queries.

        Args:
            server_type (str): Type of inference server ("openai" or "anthropic").
            model_name (str): ID of the model to use (must be compatible with available models in API).
                Defaults to "default" which is the first model in a hard-coded list.
            greedy_sample (bool): Whether to use greedy sampling (temperature=0). Defaults to False.
            verbose (bool): Whether to print initialization information. Defaults to False.
            timeout (int): Timeout in seconds for inference requests. Defaults to 400.
            max_retry (int): Maximum number of retries for failed requests. Defaults to 5.
            **kwargs: Additional arguments to pass to the server configuration.
        """
        self.server_type = server_type
        self.greedy_sample = greedy_sample
        self.verbose = verbose
        self.kwargs = kwargs
        self.timeout = timeout

        assert server_type in ["openai", "anthropic"], (
            f"Server type {server_type} not available for open source InferenceServer. "
            "Use server_type='openai' or 'anthropic'."
        )
        server_args = SERVER_PRESETS[server_type].copy()

        if kwargs:
            server_args.update(kwargs)
        if model_name == "default":
            model_name = DEFAULT_MODELS[server_type][0]

        if greedy_sample and server_type != "anthropic":
            # Anthropic does not accept sampling knobs in the same way.
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            print(f"Initializing server {server_type} with args: {server_args}")

        # Anthropic handles multiple completions as repeated requests.
        assert server_type == "anthropic" or (
            server_args.get("num_completions", 1) == 1 or server_args.get("temperature", 0) > 0
        ), "num_completions>1 requires temperature>0"

        # The endpoint comes from the config, else from OPENAI_BASE_URL or ANTHROPIC_BASE_URL,
        # which the clients read themselves when given none. Don't allow empty string env vars.
        env_var = f"{server_type.upper()}_BASE_URL"
        if env_var in os.environ and not os.environ[env_var].strip():
            del os.environ[env_var]
        base_url = server_args.get("base_url") or None

        if server_type == "openai":
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        elif server_type == "anthropic":
            if anthropic is None:
                raise ImportError(
                    "anthropic package is required for server_type='anthropic'. "
                    "Install it with: pip install anthropic"
                )
            client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=base_url,
                timeout=self.timeout,
                max_retries=0,
            )

        self.client = client
        self.model = model_name
        self.server_args = server_args

        # variables for retrying requests
        self.max_retry = max_retry
        self.retry_count = 0

    def __call__(self, messages: list[dict], single_question: bool = False, **kwargs):
        if single_question:
            # temperature=0, n=1
            kwargs = {}
        else:
            kwargs = {
                "temperature": self.server_args.get("temperature", 0),
                "n": self.server_args.get("num_completions", 1),
                "max_completion_tokens": self.server_args.get("max_tokens"),
                "top_p": self.server_args.get("top_p", 1),
            }
        if self.server_type == "openai":
            try:
                logging.debug(f"Query server: {self.server_type} with model: {self.model}")
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, stream=False, **kwargs
                )
            except openai.InternalServerError as e:
                logging.warning(f"Error in server call: {e}.\n retrying in 5s...")
                self.retry_count += 1
                if self.retry_count > self.max_retry:
                    logging.error("Max retries reached, raise error")
                    raise RuntimeError("Max retries reached for server call.")
                time.sleep(5)  # wait a bit before retrying
                return self.__call__(
                    messages,
                    single_question=single_question,
                    **kwargs,
                )
            outputs = [choice.message.content for choice in response.choices]
            truncated = [
                i
                for i, choice in enumerate(response.choices)
                if getattr(choice, "finish_reason", None) in TRUNCATION_STOP_REASONS
            ]
            usage_dict = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        elif self.server_type == "anthropic":
            # max_tokens is required by the Anthropic API,
            max_tokens = kwargs.get("max_completion_tokens") or self.server_args.get("max_tokens")
            if not max_tokens:
                raise ValueError(
                    "The Anthropic API requires max_tokens, and none is configured. Set max_tokens "
                    f"in the inference server config for this run (server_type: {self.server_type}, "
                    f"model: {self.model})."
                )
            try:
                system_blocks = [m["content"] for m in messages if m.get("role") == "system"]
                # Omit system kwarg if system block is None
                system_kwarg = {"system": "\n".join([str(x) for x in system_blocks])} if system_blocks else {}
                anthropic_messages = [m for m in messages if m.get("role") != "system"]

                n_completions = kwargs.get("n", 1)
                outputs = []
                truncated = []
                usage_dict = {"input_tokens": 0, "output_tokens": 0}
                # Anthropic does not support N completions in one request.
                for _ in range(n_completions):
                    response = self.client.messages.create(
                        model=self.model,
                        messages=anthropic_messages,
                        **system_kwarg,
                        max_tokens=max_tokens,
                    )
                    text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
                    outputs.append("".join(text_blocks))
                    if getattr(response, "stop_reason", None) in TRUNCATION_STOP_REASONS:
                        truncated.append(len(outputs) - 1)
                    usage_dict["input_tokens"] += response.usage.input_tokens
                    usage_dict["output_tokens"] += response.usage.output_tokens
            except Exception as e:
                # Do not try for specific status codes that will fail again
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    logging.error(f"Anthropic rejected the request ({status}); not retrying: {e}")
                    raise
                logging.warning(f"Error in anthropic server call: {e}.\n retrying in 5s...")
                self.retry_count += 1
                if self.retry_count > self.max_retry:
                    logging.error("Max retries reached, raise error")
                    raise RuntimeError("Max retries reached for anthropic server call.")
                time.sleep(5)
                return self.__call__(
                    messages,
                    single_question=single_question,
                    **kwargs,
                )
        else:
            raise NotImplementedError("Server type not implemented: " + self.server_type)

        # Report cut-off outputs before returning results.
        self._report_truncation(truncated, len(outputs), max_tokens=self.server_args.get("max_tokens"))
        metadata = {
            "model": self.model,
            "input_tokens": usage_dict["input_tokens"],
            "output_tokens": usage_dict["output_tokens"],
            "truncated_completions": truncated,
        }
        return outputs, metadata

    def _report_truncation(self, truncated: list[int], n_outputs: int, max_tokens) -> None:
        """Log when a completion was cut off by the token ceiling."""
        if not truncated:
            return
        logging.error(
            "%d of %d completion(s) from %s hit the max_tokens ceiling (%s) and are cut off "
            "mid-output. The generated code is incomplete, and will fail to compile with a syntax "
            "error at the end of the file rather than anything wrong with the kernel itself. "
            "Raise max_tokens for this server. Note that on models that think before answering, "
            "max_tokens covers the thinking as well as the visible answer.",
            len(truncated),
            n_outputs,
            self.model,
            max_tokens,
        )


class LLMEnsemble:
    """Ensemble of inference servers for LLM queries, allowing for weighted probabilistic
    selection of different servers / models."""

    def __init__(
        self,
        servers: list[InferenceServer],
        weights: list | str = "uniform",
        weights_warmstart: list | None = None,
        trials_warmstart: int = 0,
    ):
        """Initialize an ensemble of inference servers.

        Args:
            servers (list[InferenceServer]): List of inference server instances to ensemble.
            weights (list | str): Weighting scheme for server selection. Can be "uniform" (default) or a list of weights.
            weights_warmstart (list | None): Alternative weights for warmstart trials. Defaults to None.
            trials_warmstart (int): Number of trials to use warmstart weights. Defaults to 0.
        """
        self.servers = servers
        self.n_servers = len(servers)
        self.weights = weights
        if weights == "uniform":
            self.weights = [1.0 / len(servers)] * len(servers)
        self.weights_warmstart = weights_warmstart
        self.trials_warmstart = trials_warmstart
        logging.info(f"Initialized ensemble of models: {[s.model for s in self.servers]}")

    def __call__(self, *args, **kwargs):
        trial = kwargs.get("trial", None)
        if trial is None and self.weights_warmstart is not None:
            warnings.warn("Trial not provided to LLM but weights_warmstart is set. Ignoring warmstart weights.")
        if trial is None or trial >= self.trials_warmstart:
            weights = self.weights
        else:
            weights = self.weights_warmstart
        selected_server = np.random.choice(self.servers, p=weights)
        logging.info(f"Running inference with model {selected_server.model}")
        tic = time.time()
        out = selected_server(*args, **kwargs)
        logging.info(f"Finished inference with model {selected_server.model}, time: {time.time() - tic}")
        return out


# a list of presets for API server configs
SERVER_PRESETS = {
    "deepseek": {"temperature": 1.6, "model_name": "deepseek", "max_tokens": 4096},
    "openai": {"temperature": 0.7, "model_name": "gpt-4o", "max_tokens": 16000},
    "anthropic": {"model_name": "claude-sonnet-5", "max_tokens": 16000},
}
