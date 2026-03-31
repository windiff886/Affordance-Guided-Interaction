from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _FakeQuat:
    def __init__(self, real: float, imag: tuple[float, float, float]) -> None:
        self._real = real
        self._imag = imag

    def GetReal(self) -> float:
        return self._real

    def GetImaginary(self) -> tuple[float, float, float]:
        return self._imag


class _FakeVec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self._values = (x, y, z)

    def __getitem__(self, index: int) -> float:
        return self._values[index]


def test_quat_to_float_components_preserves_values() -> None:
    from affordance_guided_interaction.utils.usd_math import quat_to_float_components

    components = quat_to_float_components(_FakeQuat(0.5, (0.1, 0.2, 0.3)))

    assert components == (0.5, 0.1, 0.2, 0.3)


def test_vec3_to_float_components_preserves_values() -> None:
    from affordance_guided_interaction.utils.usd_math import vec3_to_float_components

    components = vec3_to_float_components(_FakeVec3(1.0, 2.0, 3.0))

    assert components == (1.0, 2.0, 3.0)
