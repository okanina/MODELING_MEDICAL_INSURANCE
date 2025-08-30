from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT="-e ."
def get_requirements(filepath:str)->List[str]:

    with open(filepath, "r") as f:
        requirements=f.readlines()
        requirements=[line.replace('\n', '') for line in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="MODELING_MEDICAL_INSURANCE",
    version="0.0.1",
    author = "Khanyisile C. Jiyane",
    author_email = "kjjkhanyi@gmail.com",
    packages =find_packages(),
    install_requires =get_requirements("requirements.txt")
)