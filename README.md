# INSTALLATION

<p> I will detail here how to setup the installation of pySTED inside its own virtual environment. This ensures all the
necessary packages are installed and that the C functions are compiled. pySTED uses some C functions for performance 
gains, which need to be compiled before execution. 

Once the repository is cloned on your computer (<code> git clone https://github.com/FLClab/audurand_pysted.git </code>),
create a venv. You can create the venv anywhere, but for simplicity, create the venv inside the audurand_pysted folder.
Open a terminal and cd into the audurand_pysted directory. Then call

<code> python -m venv ./my_venv </code>

This should create a folder named my_venv inside the audurand_pysted folder. The next step will depend on your operating
system. If you are running Windows, run

<code> my_venv\Scripts\activate.bat </code>

If you are running a Linux system (MAC TOO ??? NEED TO VERIFY :) ), run

<code> source ./my_venv/bin/activate </code>

This activates the virtual environment. To install the necessary public libraries, run

<code> python -m pip install -r requirements.txt </code>

To compile the C functions, <code> cd </code> into the directory containing audurand_pysted </p>

<code> cd .. </code>

and run

<code>python -m pip install -e audurand_pysted </code>

<p> Once the C functions are compiled, try running the <code>ex_albert.py</code> script. It can take a while to 
execute, but you should get a figure that looks like </p>

![yo](/examples/ex_albert.png)
