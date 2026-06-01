from setuptools import setup, find_packages

setup(
    name="data_inspector",
    version="0.1.0",
    description="A Modular Data Sanitization & Exploration Engine",
    author="Ramanayaka R.M.K.D.D.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "plotly",
        "scipy"
    ],
)
