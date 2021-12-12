import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpsclean",
    version="0.2.0",
    author="Davide Sbetti",
    author_email="davide.sbetti@gmail.com",
    description="An application to correct a GPS trace using machine learning techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sbettid/GPSClean",
    project_urls={
        "Bug Tracker": "https://github.com/sbettid/GPSClean/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "art==5.3",
        "filterpy==1.4.5",
        "geojson==2.5.0",
        "gpxpy==1.4.2",
        "tensorflow==2.7.0",
        "pandas==0.25.3",
        "scipy==1.6.1",
        "pyproj==3.1.0",
        "numpy==1.21.4",
        "pyinstaller==4.5.1",
    ],
    package_data={
         "gpsclean": ["data/*.h5"],
    },
    entry_points={
    'console_scripts': [
        'gpsclean = gpsclean.gpsclean:main',
    ],
},
)