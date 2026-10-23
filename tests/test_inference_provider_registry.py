r"""
Config asks for an LLM by a short provider name (e.g. "openai"); the registry is what
turns that into the class Hydra actually instantiates. Getting this wrong either breaks
every task's config or, worse, silently instantiates the wrong provider.
"""

import types

import pytest

from kernelfoundry.algorithm.inference_server import (
    DisabledServer,
    InferenceServer,
    LLMEnsemble,
    _resolve_inference_target,
    register_inference_provider,
)


class TestDefaults:
    def test_openai_resolves_to_the_public_inference_server(self):
        assert _resolve_inference_target("openai") == f"{InferenceServer.__module__}.InferenceServer"

    def test_anthropic_resolves_to_the_public_inference_server(self):
        assert _resolve_inference_target("anthropic") == f"{InferenceServer.__module__}.InferenceServer"

    def test_claude_cli_resolves_to_the_subscription_backed_server(self):
        assert _resolve_inference_target("claude_cli") == "kernelfoundry.algorithm.claude_cli_server.ClaudeCLIServer"

    def test_unknown_provider_names_the_problem_and_lists_alternatives(self):
        with pytest.raises(ValueError, match="unicorn-llm") as exc_info:
            _resolve_inference_target("unicorn-llm")
        assert "openai" in str(exc_info.value), "the error should help the user fix their config"

    def test_a_null_provider_resolves_to_the_disabled_placeholder(self):
        """An unused ensemble slot (model_2/model_3 by default) has provider: null, and
        Hydra instantiates every slot regardless of weight -- this must not raise."""
        assert _resolve_inference_target(None) == f"{DisabledServer.__module__}.DisabledServer"


class TestEnsembleWeights:
    """model_1/model_2/model_3 are always instantiated, so LLMEnsemble must prune the
    disabled slots (provider: null -> DisabledServer) before np.random.choice ever sees
    them, and normalize whatever weights are left to sum to 1."""

    @staticmethod
    def _fake_server(model="fake-model"):
        return types.SimpleNamespace(model=model, server_type="fake-provider")

    def test_uniform_keyword_splits_evenly(self):
        ensemble = LLMEnsemble(servers=[self._fake_server(), self._fake_server()], weights="uniform")
        assert ensemble.weights == [0.5, 0.5]

    def test_explicit_weights_are_normalized_to_sum_to_one(self):
        servers = [self._fake_server() for _ in range(3)]
        ensemble = LLMEnsemble(servers=servers, weights=[1, 1, 2])
        assert ensemble.weights == pytest.approx([0.25, 0.25, 0.5])

    def test_already_normalized_weights_are_unchanged(self):
        """Existing configs already write weights that sum to 1; normalizing must be a no-op."""
        servers = [self._fake_server() for _ in range(3)]
        ensemble = LLMEnsemble(servers=servers, weights=[0.5, 0.5, 0])
        assert ensemble.weights == pytest.approx([0.5, 0.5, 0.0])

    def test_disabled_slots_are_pruned_from_uniform_ensembles(self):
        """The default model_2/model_3 slots (provider: null) instantiate as
        DisabledServer -- with the default weights="uniform", these must not linger in
        self.servers, and the survivor(s) split the weight uniform would otherwise have
        given to all three slots."""
        real = self._fake_server()
        ensemble = LLMEnsemble(servers=[real, DisabledServer(), DisabledServer()], weights="uniform")
        assert ensemble.servers == [real]
        assert ensemble.weights == [1.0]

    def test_all_slots_disabled_raises(self):
        with pytest.raises(ValueError, match="disabled"):
            LLMEnsemble(servers=[DisabledServer(), DisabledServer()], weights="uniform")

    def test_weights_warmstart_is_normalized_too(self):
        servers = [self._fake_server() for _ in range(2)]
        ensemble = LLMEnsemble(servers=servers, weights="uniform", weights_warmstart=[1, 3])
        assert ensemble.weights_warmstart == pytest.approx([0.25, 0.75])

    def test_extra_config_only_fields_are_accepted_and_ignored(self):
        """Hydra passes every sibling of `_target_` as a kwarg, including provider/model/
        model_1/model_2/model_3 -- LLMEnsemble must tolerate and ignore them."""
        LLMEnsemble(
            servers=[self._fake_server()], weights="uniform", provider="openai", model="gpt-5.4", model_1={}, model_2={}
        )


class TestClaudeCliOnlyFieldsPassThrough:
    """configs/inference/server.yaml passes disallow_tools/max_turns/effort to every provider's
    slot, since only claude_cli uses them -- InferenceServer must silently ignore them rather
    than raising, or every non-claude_cli config would break."""

    def test_openai_ignores_claude_cli_only_fields(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        server = InferenceServer(
            server_type="openai",
            model_name="gpt-5",
            disallow_tools=True,
            max_turns=8,
            effort=None,
        )
        assert server.model == "gpt-5"

    def test_disabled_server_ignores_claude_cli_only_fields(self):
        DisabledServer(disallow_tools=True, max_turns=8, effort="high")


class TestRegistration:
    """This is the hook an internal deployment uses (via an internal plugin module) to add
    providers without kernelfoundry importing or naming them."""

    def test_a_newly_registered_provider_resolves_to_the_given_target(self):
        register_inference_provider("test-provider", "some.module.SomeClass")
        assert _resolve_inference_target("test-provider") == "some.module.SomeClass"

    def test_registering_an_existing_provider_overrides_it(self):
        register_inference_provider("openai", "some.module.Override")
        try:
            assert _resolve_inference_target("openai") == "some.module.Override"
        finally:
            register_inference_provider("openai", f"{InferenceServer.__module__}.InferenceServer")
