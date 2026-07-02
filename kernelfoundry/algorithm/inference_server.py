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
from concurrent.futures import ThreadPoolExecutor, as_completed

# API clients
from openai import OpenAI
import openai

try:
    import anthropic
except ImportError:  # optional dependency for Anthropic support
    anthropic = None
import boto3
import botocore

models_avail = {
    "openai": [
        "gpt-4.1",
        "gpt-4o",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    "anthropic": [
        "claude-4-5-sonnet",
        "claude-4-5-haiku",
        "claude-4-opus",
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
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            print(f"Initializing server {server_type} with args: {server_args}")

        if model_name == "default":
            model_name = models_avail[server_type][0]

        assert (
            server_args.get("num_completions", 1) == 1 or server_args.get("temperature", 0) > 0  #
        ), "num_completions>1 requires temperature>0"

        if server_type == "openai":
            url = server_args.get("base_url", "https://api.openai.com/v1/")
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=url,
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
                timeout=self.timeout,
                max_retries=0,
            )

        self.client = client
        self.model = model_name
        self.server_args = server_args

        # variables for retrying requests
        self.max_retry = max_retry
        self.retry_count = 0

    def __call__(self, messages: list[dict], single_question: bool = False, return_model_info: bool = False, **kwargs):
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
                return self.__call__(messages, single_question)
            outputs = [choice.message.content for choice in response.choices]
        elif self.server_type == "anthropic":
            try:
                system_blocks = [m["content"] for m in messages if m.get("role") == "system"]
                system_prompt = "\n".join([str(x) for x in system_blocks]) if system_blocks else None
                anthropic_messages = [m for m in messages if m.get("role") != "system"]

                n_completions = kwargs.get("n", 1)
                outputs = []
                # Anthropic API does not expose n completions in a single call, so we call it repeatedly.
                for _ in range(n_completions):
                    response = self.client.messages.create(
                        model=self.model,
                        messages=anthropic_messages,
                        system=system_prompt,
                        max_tokens=kwargs.get("max_completion_tokens"),
                        temperature=kwargs.get("temperature"),
                        top_p=kwargs.get("top_p"),
                    )
                    text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
                    outputs.append("".join(text_blocks))
            except Exception as e:
                logging.warning(f"Error in anthropic server call: {e}.\n retrying in 5s...")
                self.retry_count += 1
                if self.retry_count > self.max_retry:
                    logging.error("Max retries reached, raise error")
                    raise RuntimeError("Max retries reached for anthropic server call.")
                time.sleep(5)
                return self.__call__(messages, single_question)
        else:
            raise NotImplementedError("Server type not implemented: " + self.server_type)

        # output processing
        if return_model_info:
            model_info = [self.model] * len(outputs)
            return outputs, model_info
        return outputs


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
    "openai": {"temperature": 0.7, "model_name": "gpt-4o", "max_tokens": 4096},
    "anthropic": {"temperature": 0.7, "model_name": "claude-4-5-sonnet", "max_tokens": 4096},
}
