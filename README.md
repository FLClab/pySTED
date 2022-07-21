# pySTED

The pySTED simulator aims to realistically simulate STED imaging acquisitions. The implementation follows from the analytical descriptions of STED microscopy mechanisms.

The simulator consists in a microscope that acquires images on a datamap describing the structure to image. The microscope is comprised of five objects : the excitation and STED beams, an objective lens, a detector, and the photophysical parameters of the fluorophores. Each object is characterized by adjustable parameters which will affect the image signal, such as the fluorophore's quantum yield, the detector's efficiency, and the background signal. The datamap is represented by a 2D array in which each element of the structure of interest indicates the number of fluorescent molecules at that position.

## Installation

We recommend using a virtual environment in which to install pySTED. [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) facilitates the creation of a virtual environment on most operating system. For exemple, a virtual environment can be created using
```bash
conda create --name venv python=3.8
conda activate venv
```

Otherwise, a virtual environment can be created using the built-in python functionalities. This will create a folder named venv inside the pySTED folder which can then be activated depending on the operating system.
```bash
python -m venv ./venv

# On Windows
venv\Scripts\activate.bat

# On Linux / MAC os
source ./venv/bin/activate
```

### Using pip

We do not provide a pypi installation package (yet). The user can however install pySTED using the url of the current repository
```bash
python -m pip install git+https://github.com/FLClab/pySTED
```

### From source 

I will detail here how to setup the installation of pySTED inside its own virtual environment. This ensures all the
necessary packages are installed and that the C functions are compiled. pySTED uses some C functions for performance 
gains, which need to be compiled before execution. 

Once the repository is cloned on your computer 
```bash
git clone https://github.com/FLClab/pySTED.git
```

To install the necessary public libraries, run
```bash
python -m pip install -r pySTED/requirements.txt
python -m pip install -e pySTED
```

## Example
  
Once the C functions are compiled, try running the script. It shows the basic workings of pySTED.
```bash 
python simple_example_script.py
```
