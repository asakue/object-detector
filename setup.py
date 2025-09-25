from setuptools import setup, find_packages

setup(
    name="object-detector",
    version="1.0.0",
    description="Real-time object detection with distance measurement",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "ultralytics>=8.0.0",
    ],
    python_requires=">=3.7",
)