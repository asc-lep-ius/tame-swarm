"""MoB layers keyed by block index, under whatever wraps the transformer.

The numbers a steering record names are block indices, and a lookup that walks
``model.model.layers`` finds nothing under a PEFT wrapper -- and said nothing.
#24's first ablation arm measured routing against no direction for exactly that
reason, so the lookup reads the block index off the module name instead.
"""

from peft import LoraConfig, TaskType, get_peft_model

from mob import apply_mob_to_model, get_mob_layers, mob_layers_by_index

from .conftest import build_tiny_causal_lm

MOB_BLOCKS = [1, 3]


def test_layers_are_found_by_block_index(tiny_mob_config):
    model = apply_mob_to_model(build_tiny_causal_lm(4), tiny_mob_config, MOB_BLOCKS)

    by_index = mob_layers_by_index(model)

    assert sorted(by_index) == MOB_BLOCKS
    assert list(by_index.values()) == get_mob_layers(model)


def test_layers_are_still_found_under_a_peft_wrapper(tiny_mob_config):
    """The failure that went unnoticed: a LoRA-wrapped model has the blocks two levels deeper."""
    model = apply_mob_to_model(build_tiny_causal_lm(4), tiny_mob_config, MOB_BLOCKS)
    wrapped = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, target_modules=["q_proj", "v_proj"]),
    )

    by_index = mob_layers_by_index(wrapped)

    assert sorted(by_index) == MOB_BLOCKS
    assert list(by_index.values()) == get_mob_layers(wrapped)


def test_a_model_without_mob_layers_has_none(tiny_causal_lm):
    assert mob_layers_by_index(tiny_causal_lm) == {}
