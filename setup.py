import setuptools

with open("template/README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fr:
    requirements = fr.read().strip().split('\n')

setuptools.setup(
    name="gryphon-data-exploration",
    version="0.0.1",
    author="Daniel Wang",
    author_email="daniel.wang@oliverwyman.com",
    description="Data diagnostics for general data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
)
