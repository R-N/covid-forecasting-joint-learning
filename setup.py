import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='covid_forecasting_joint_learning',
    version='0.0.1',
    author="Muhammad Rizqi Nur",
    author_email="rizqinur-2010@gmail.com",
    description="COVID-19 forecasting model for East Java using Joint learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/R-N/covid-forecasting-joint-learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy~=1.19.2",
        "six~=1.15.0",
        "folium==0.2.1",
        "imgaug<0.2.7,>=0.2.5",
        "requests~=2.23.0",
        "pandas~=1.1.0; python_version >= '3.0'",
        "matplotlib",
        "mpld3",
        "seaborn",
        "statsmodels==0.12",
        "scikit-learn==0.24.1",
        "tslearn",
        "optuna",
        "torchinfo"
    ]
)
