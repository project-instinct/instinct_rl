from setuptools import find_packages, setup

setup(
    name="instinct_rl",
    version="1.0.2",
    author="Ziwen Zhuang",
    author_email="",
    license="BSD-3-Clause",
    packages=find_packages(),
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.6",
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "tensorboardX",
        "tensorboard",
        "tabulate",
        "GitPython",
        "onnx",
        "onnxscript",
    ],
)
