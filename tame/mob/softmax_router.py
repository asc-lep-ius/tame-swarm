"""The learned-gate control arm: the same reports, without the economy.

#12 exists because the null hypothesis a reader reaches for is *"MoB is a learned
router with a wealth scalar"*. Answering it needs an arm that differs from the
auction in exactly one respect, and this is that arm: the **same**
``ConfidenceHead`` modules, the same clamped logits, the same experts and the same
upcycled base weights, with ``softplus x wealth -> VCG top-k -> 1/k share``
replaced by ``softmax -> top-k -> renormalised share``.

What is removed is the economy and only the economy: wealth is never read, no
payment or rebate is computed, and no value objective is added to the loss. What
is deliberately *kept* is the router z-loss, which regularises the same logits in
both arms and so is not part of the difference under test.

Two asymmetries are inherent to the comparison rather than chosen here, and are
recorded so no one has to rediscover them:

1. **The LM loss trains this gate and does not train the auction's.** Under
   ``routing_share="uniform"`` the auction's share is a ``full_like`` constant, so
   the confidence heads receive gradient only from their own value objective. A
   softmax share is differentiable in the logits by construction, so the heads are
   trained by the language-modelling loss here. That difference *is* the
   difference between an auction and a learned router; it cannot be removed
   without removing the baseline.
2. **The backbone is detached in both arms.** ``MixtureOfBidders.forward``
   routes on ``hidden_states.detach()``, so neither gate backpropagates into the
   representation every expert reads. A standard MoE router would not detach, but
   holding it constant is what keeps this a one-variable comparison.

Softmax is taken over the **logits**, not over the softplus reports. The logit is
the analogue of a standard router's pre-softmax score; softplus exists to make a
non-negative *bid* expressible, and a softmax of it would be neither the standard
gate nor a cleaner one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .auction import AuctionOutcome

# Gate selection. The third arm of #12 -- ``dense``, the original FFN -- is not a
# gate at all: it is the absence of a MoB layer, so it is selected in the training
# harness by not converting the layer, not by a value here.
ROUTER_AUCTION = "auction"
ROUTER_SOFTMAX = "softmax"
SUPPORTED_ROUTERS = frozenset({ROUTER_AUCTION, ROUTER_SOFTMAX})


class SoftmaxRouter(nn.Module):
    """Top-*k* softmax gating over expert logits, renormalised across the winners.

    Stateless and parameterless: the parameters it gates with are the
    ``ConfidenceHead`` modules the auction arm also uses, which is the entire point
    of this control. It returns an :class:`AuctionOutcome` so the layer above needs
    no branch for the shape of what came back -- ``payments`` and ``rebates`` are
    ``None`` because there is no auction to have produced them, which is different
    from an auction that produced zeros.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, confidence_logits: torch.Tensor) -> AuctionOutcome:
        """Route on logits alone.

        Softmax over all experts, top *k*, renormalise -- the Mixtral/Switch gate.
        Note that this is *identical* to a softmax over the top-*k* logits alone:
        the losers' probability mass appears in both numerator and denominator of
        the renormalisation and cancels exactly. So a winner's share depends only on
        the gaps between the winners' logits, which is worth knowing before reading
        anything into how confidently the gate rejected the experts it dropped.

        Accumulated in float32 for the reason the auction's log-domain gate is --
        bfloat16 is the training dtype, and the difference between two nearby
        logits is exactly what the gate is trying to represent.
        """
        accumulate_dtype = torch.promote_types(confidence_logits.dtype, torch.float32)
        weights = F.softmax(confidence_logits.to(accumulate_dtype), dim=-1)

        top_weights, selected_experts = torch.topk(weights, self.top_k, dim=-1)

        # No epsilon: the top k of a softmax hold at least k/num_experts of the
        # mass, so the divisor is bounded away from zero by the shape of the tensor.
        routing_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        return AuctionOutcome(
            selected_experts=selected_experts,
            routing_weights=routing_weights.to(confidence_logits.dtype),
            payments=None,
            rebates=None,
        )
