# pcba_project/setup.py

from setuptools import setup, find_packages

setup(
    name="pcba_project",
    version="0.1.0",
    author="Your Name",
    description="Simulation of Polycarboxybetaine polymer using mathematical methods",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20",
        "matplotlib",   # Follow-up visualization dependencies
        "scipy",        # Implicit integrals or advanced function dependencies
    ],
    python_requires=">=3.7",
)

