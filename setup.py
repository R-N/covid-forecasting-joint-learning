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
)