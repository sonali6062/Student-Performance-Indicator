# from setuptools import find_packages, setup
# from typing import List

# HYPEN_E_DOT='-e .'
# def get_requirements(file_path:str)->list[str]:
#     """return a list of packages."""
#     requirements=[]
#     with open(file_path) as file_obj:
#         requirements = file_obj.readlines()
#         requirements = [req.replace("\n","") for req in requirements]
#         if HYPEN_E_DOT in requirements:
#             requirements.remove(HYPEN_E_DOT)
#     return requirements

# setup(
#     name="STUDENT_PERFORMANCE_PREDICTION",
#     version="0.0.1",
#     author="Sonali Kumari",
#     author_email="sonalikumari6062@gmail.com",
#     packages=find_packages(),
#     install_requires=get_requirements('requirements.txt'),
#     description="ML project for student performance prediction",
#     python_requires='>=3.8'
# )
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Return a list of packages from requirements.txt"""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="STUDENT_PERFORMANCE_PREDICTION",
    version="0.0.1",
    author="Sonali Kumari",
    author_email="sonalikumari6062@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description="ML project for student performance prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8'
)
