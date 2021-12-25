import setuptools
import sys
import platform

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
#get version for tflite-runtime    
#cur_version = str(sys.version_info[0]) + "" + str(sys.version_info[1])
#m_char = ''
#if cur_version == '37':
#    m_char = 'm'

#cur_os = "linux_x86_64"
#os_signature = platform.system()
#if "Windows" in os_signature:
#    cur_os = "win_amd64"
#elif "Darwin" in os_signature:
#    cur_os = "macosx_11_0_x86_64"

#tflite = "tflite_runtime>=2.5.0@https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp{}-cp{}{}-{}.whl".format(cur_version,cur_version, m_char, cur_os)

setuptools.setup(
    name="gpsclean",
    version="0.3.0",
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
        "tflite-runtime>=2.5.0",
        "pandas>=0.25.3",
        "scipy>=1.6.1",
        "pyproj>=3.0.0",
        "numpy>=1.20.0",
    ],
    package_data={
         "gpsclean": ["data/*.tflite"],
    },
    entry_points={
    'console_scripts': [
        'gpsclean = gpsclean.gpsclean:main',
    ],
},
)