from setuptools import find_packages, setup

def get_requirements(file_path):
    """Read requirements.txt and return a list of packages."""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name="STUDENT_PERFORMANCE_PREDICTION",
    version="0.0.1",
    author="Sonali Kumari",
    author_email="sonalikumari6062@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
