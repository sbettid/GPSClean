import setuptools

pkg_vars  = {}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("src/gpsclean/_version.py") as fp:
    exec(fp.read(), pkg_vars)

setuptools.setup(
    name="gpsclean",
    version=pkg_vars['__version__'],
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
    python_requires=">=3.7, <3.10",
    install_requires=[
        "art==5.3",
        "filterpy==1.4.5",
        "geojson==2.5.0",
        "gpxpy==1.4.2",
        "tflite-runtime>=2.5.0.post1",
        "pandas>=0.25.3",
        "scipy>=1.6.1",
        "pyproj>=3.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.0.0",
    ],
    package_data={
         "gpsclean": ["data/*.tflite"],
    },
    entry_points={
    'console_scripts': [
        'gpsclean = gpsclean.main:main',
    ],
},
)