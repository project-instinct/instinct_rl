from setuptools import find_packages, setup

setup(
    name="instinct_rl",
    version="1.0.3",
    author="Ziwen Zhuang",
    author_email="",
    license="Modified MIT License",
    packages=find_packages(),
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.12,<3.13",
    install_requires=[
        "torch==2.11.0",
        "torchvision==0.26.0",
        "numpy>=2",
        "tensorboardX",
        "tensorboard",
        "tabulate",
        "GitPython",
        "onnx>=1.18,<1.22",
        "onnxscript>=0.5",
    ],
)
