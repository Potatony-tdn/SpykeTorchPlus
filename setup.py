import os
import setuptools

with open(os.path.join(os.path.dirname(__file__),"SpykeTorchPlus/VERSION")) as f:
    version = f.read().strip() 


with open("README.md", "r") as fh:
    long_description = fh.read()

def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements

    
setuptools.setup(
    name="SpykeTorchPlus",
    version=version,
    author="Muxiao Liu (forked from Milad Mozafari)",
    author_email="",
    description="Enhanced simulator of spiking convolutional neural networks with realistic event data support, temporal dynamics, and extended STDP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Potatony-tdn/SpykeTorchPlus",
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    python_requires='>=3.6',
)
