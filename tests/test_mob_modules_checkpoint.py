"""A checkpoint restores the experts, heads and coupling it trained, under PEFT and bare (#29).

Before this, a ``--use_lora`` checkpoint held the LoRA attention adapters and the
ledgers and nothing the MoB layers had learned: ``load_mob_state`` put a trained
wealth vector onto zero-initialised experts and called it restored.
"""

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model

from mob import (
    apply_mob_to_model,
    load_mob_modules,
    mob_layers_by_index,
    save_mob_modules,
)
from mob.utils import owned_state

from .conftest import TINY_HIDDEN_DIM, build_tiny_causal_lm

MOB_BLOCKS = [1, 3]


def _wrap(model, peft: bool):
    if not peft:
        return model
    return get_peft_model(
        model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, target_modules=["q_proj", "v_proj"])
    )


def _converted(tiny_mob_config, peft: bool, coupled: bool):
    torch.manual_seed(0)
    model = apply_mob_to_model(build_tiny_causal_lm(4), tiny_mob_config, MOB_BLOCKS)
    if coupled:
        mob_layers_by_index(model)[1].attach_coupling(torch.randn(TINY_HIDDEN_DIM))
    return _wrap(model, peft)


def _perturb(model) -> None:
    """Move every owned tensor off its initial value, as a training step would."""
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for mob in mob_layers_by_index(model).values():
            for name, tensor in mob.state_dict().items():
                if name in owned_state(mob) and tensor.is_floating_point():
                    tensor.add_(torch.randn(tensor.shape, generator=generator))
            coupling = getattr(mob, "coupling", None)
            if coupling is not None:
                coupling.set_coupling_step(37)


def _owned(model) -> dict[int, dict[str, torch.Tensor]]:
    return {index: owned_state(mob) for index, mob in mob_layers_by_index(model).items()}


@pytest.mark.parametrize("peft", [False, True], ids=["bare", "peft"])
def test_the_trained_modules_round_trip_onto_a_fresh_conversion(tiny_mob_config, tmp_path, peft):
    trained = _converted(tiny_mob_config, peft, coupled=True).eval()
    _perturb(trained)
    tokens = torch.randint(0, 16, (2, 8))
    with torch.no_grad():
        expected = trained(input_ids=tokens).logits
    path = tmp_path / "mob_modules.pt"

    assert save_mob_modules(trained, path) == len(MOB_BLOCKS)
    fresh = _converted(tiny_mob_config, peft, coupled=False).eval()
    with torch.no_grad():
        assert not torch.allclose(fresh(input_ids=tokens).logits, expected), (
            "the perturbation must show"
        )
    assert load_mob_modules(fresh, path) == len(MOB_BLOCKS)

    for index, state in _owned(trained).items():
        restored = _owned(fresh)[index]
        assert sorted(restored) == sorted(state)
        for name, tensor in state.items():
            assert torch.equal(restored[name], tensor), f"block {index}: {name}"
    coupling = mob_layers_by_index(fresh)[1].coupling
    assert int(coupling._coupling_step.item()) == 37
    assert coupling.config == mob_layers_by_index(trained)[1].coupling.config
    with torch.no_grad():
        assert torch.equal(fresh(input_ids=tokens).logits, expected)


def test_the_saved_keys_do_not_depend_on_the_wrapper(tiny_mob_config, tmp_path):
    """The same body saved bare or under PEFT is the same file, block for block."""
    bare, wrapped = tmp_path / "bare.pt", tmp_path / "peft.pt"
    save_mob_modules(_converted(tiny_mob_config, peft=False, coupled=True), bare)
    save_mob_modules(_converted(tiny_mob_config, peft=True, coupled=True), wrapped)

    a = torch.load(bare, weights_only=True)
    b = torch.load(wrapped, weights_only=True)

    assert a["_config"] == b["_config"] and sorted(a["blocks"]) == MOB_BLOCKS
    for index in MOB_BLOCKS:
        assert sorted(a["blocks"][index]["state"]) == sorted(b["blocks"][index]["state"])
    saved = a["blocks"][1]["state"]
    assert not any(name.startswith("base_") for name in saved)
    assert "expert_wealth" not in saved, "ledgers live in mob_state.pt"
    for name in (
        "experts.0.gate_adapter_B.weight",
        "experts.2.down_adapter_A.weight",
        "confidence_heads.0.proj.weight",
        "confidence_heads.1.proj.bias",
        "coupling.detector",
        "coupling.steering_direction",
        "coupling._coupling_step",
    ):
        assert name in saved, name


def test_a_partial_or_mismatched_restore_is_refused(tiny_mob_config, tmp_path):
    """Strict by design: the silent partial restore is the defect this replaces."""
    path = tmp_path / "mob_modules.pt"
    save_mob_modules(_converted(tiny_mob_config, peft=False, coupled=False), path)

    other_blocks = apply_mob_to_model(build_tiny_causal_lm(4), tiny_mob_config, [1, 2])
    with pytest.raises(ValueError, match="convert the same layers"):
        load_mob_modules(other_blocks, path)

    coupled = _converted(tiny_mob_config, peft=False, coupled=True)
    with pytest.raises(ValueError, match="coupled and the checkpoint is not"):
        load_mob_modules(coupled, path)

    payload = torch.load(path, weights_only=True)
    del payload["blocks"][1]["state"]["confidence_heads.0.proj.bias"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unrestored"):
        load_mob_modules(_converted(tiny_mob_config, peft=False, coupled=False), path)


def test_a_coupling_configured_differently_is_refused(tiny_mob_config, tmp_path):
    from coupling import SteeringCouplingConfig

    path = tmp_path / "mob_modules.pt"
    save_mob_modules(_converted(tiny_mob_config, peft=False, coupled=True), path)
    other = _converted(tiny_mob_config, peft=False, coupled=False)
    mob_layers_by_index(other)[1].attach_coupling(
        torch.randn(TINY_HIDDEN_DIM),
        SteeringCouplingConfig(hidden_dim=TINY_HIDDEN_DIM, coupling_beta=0.5),
    )

    with pytest.raises(ValueError, match="report a different arm"):
        load_mob_modules(other, path)


def test_a_model_without_mob_layers_saves_nothing(tmp_path):
    assert save_mob_modules(build_tiny_causal_lm(2), tmp_path / "none.pt") == 0
    assert not (tmp_path / "none.pt").exists()
