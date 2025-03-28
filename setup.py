from setuptools import setup, find_packages

setup(
    name="BEng_Hons_Diss_TMP",
    version="1.0.0",
    packages=find_packages(include=["fig", "suite", "models", "robopianist"]),
    install_requires=[
        "mujoco-py<2.2,>=2.1",
        # other dependencies
    ],
)
