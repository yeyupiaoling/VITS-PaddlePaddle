from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="monotonic_align_paddle.core",
            sources=["src/core.pyx"],
        ),
    ]
)
