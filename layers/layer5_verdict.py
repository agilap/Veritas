def generate_verdict(l1, l2, l3, l4, l6) -> dict:
    return {
        "layer": 5,
        "error": None,
        "inputs": {
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "l4": l4,
            "l6": l6,
        },
    }