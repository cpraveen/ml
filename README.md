# Deep Learning Tutorial

This repository contains contains scripts and notes for understanding the various aspects of Deep Learning. The scripts are written using TensorFlow, which is an open source machine learning framework.

## Installing TensorFlow
The system requirements for intalling TensorFlow are available [here](https://www.tensorflow.org/install). The instructions below use Python 2.7 and pip. If you prefer using Python 3, or the Anaconda packages instead of the TensorFlow-provided pip package, you can use the alternate instructions detailed [here](https://www.tensorflow.org/install).

1. Check if your Python environment is already configured:
   
        $ python --version
        $ pip --version
        $ virtualenv --version
        
   If any of these packages are not installed, then you can follow the instructions given [here](https://www.tensorflow.org/install) to install them.
   
2. Go the the home directory of this tuturial. Create a new virtual environment by choosing a Python interpreter and making a `./dl_venv` directory to hold it:  

        $ virtualenv --system-site-packages -p python2.7 ./dl_venv   
        
3. Activate the virtual environment:
              
        $ source ./dl_venv/bin/activate
        
   We can now install packages without effecting the host system setup. Note that when the virtual environment is active, your shell prompt is prefixed with `(dl_venv)`

4. Upgrade `pip`:

        (dl_venv) $ pip install --upgrade pip        
        
5. Install TensorFlow in the virtual environment:

		 (dl_venv) $ pip install --upgrade tensorflow
		 
6. We will also need `matplotlib` for plotting in Python. So install this using: 

		 (dl_venv) $ pip install matplotlib		        
7. You can list all the packages installed in the virtual environment using:

		 (dl_venv) $ pip list
		 
8. Test the installation by running a test script:

		 (dl_venv) $ python Test_Install/test_tf.py
		 
	If the code runs and displays `Script was run successfully!!`, then everything has been correctly installed.
	
9. The virtual environment can be exited using:

        (dl_venv) $ deactivate 	
        
### Installing Jupyter

I plan to use Jupyter Notebook for the practical sessions. If you wish to do the same, you can install Jupyter using:

    $ python -m pip install --upgrade pip
    $ python -m pip install jupyter  
You can also install Jupyter using Anaconda by following the instruction [here](https://jupyter.org/install.html).	    
     

To use Jupyter inside `dl_venv`, first activate the virtual environment and install the ipython kernal using `pip`:
    
    (dl_venv) $ pip install ipykernel
   
Now install a new kernal:

    (dl_venv) $ ipython kernel install --user --name=dl_venv
   
Note that we give the kernal the same name as the virtual envirnoment, although this is not necessary. At this point, you can start Jupyter
   
    (dl_venv) $ jupyter notebook

To test whether Jupyter is able to load and use TensorFlow, open and run the script `Test_Install/test_tf.ipynb`.      
        
 
		
		
		

## Deep learning resources

*  [Deep Learning Book](https://www.deeplearningbook.org/) by Goodfellow, Bengio and Courville. This is a good book for beginners and looks are the various components needed to build and train neural networks.
*  Various [keywords](http://www.wildml.com/deep-learning-glossary/) used by people working with machine learning.
*  [TensorFlow tutorial](https://www.tensorflow.org/tutorials/).
*  The 5 part Deep Learning Specialization course offered on [Coursera](https://www.coursera.org/specializations/deep-learning), taught by Andrew NG and other experts in the field.
*  [Notes](https://stats385.github.io/cheat_sheet) and [slides](https://stats385.github.io/lecture_slides) from the Deep Learning course taught at Standford.
*  Many, many other blogs and basic tutorials put up by others ...