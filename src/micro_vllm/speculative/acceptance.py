from dataclasses import dataclass

@dataclass
class AcceptanceResult:
    accepted: list[int]
    replacement: list[int]
    extra: list[int]

def greedy_accept(proposals: list[int], predictions: list[int]) -> AcceptanceResult:
    if len(predictions) != len(proposals) + 1:
        raise ValueError(
            f"predictions must contain len(proposals) + 1 tokens, "
            f"got {len(proposals)} proposals and {len(predictions)} predictions"
        )

    accepted = []
    for index, proposal in enumerate(proposals):
        if proposal != predictions[index]:
            return AcceptanceResult(
                accepted=accepted,
                replacement=[predictions[index]],
                extra=[],
            )
        accepted.append(proposal)

    return AcceptanceResult(
        accepted=accepted,
        replacement=[],
        extra=[predictions[-1]],
    )
