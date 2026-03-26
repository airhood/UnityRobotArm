def unity_to_rh(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Unity left-hand to right-hand coordinate system."""
    return x, y, -z


def rh_to_unity(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert right-hand to Unity left-hand coordinate system."""
    return x, y, -z
