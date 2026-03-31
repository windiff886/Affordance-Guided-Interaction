from __future__ import annotations


def quat_to_float_components(quat: object) -> tuple[float, float, float, float]:
    imag = quat.GetImaginary()
    return (
        float(quat.GetReal()),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
    )


def vec3_to_float_components(vec3: object) -> tuple[float, float, float]:
    return (
        float(vec3[0]),
        float(vec3[1]),
        float(vec3[2]),
    )
