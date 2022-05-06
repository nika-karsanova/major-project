# AI Figure Skating Commentator - Major Project

This repository contains files relevant to the technical submission of the BSc project at Aberystwyth University. 

It is primarily a Computer Vision and Machine Learning project put together with Python 3.10 and some third-party 
libraries. The goal was to create a system, which would be able to recognize and identify some basic events in a 
complex sports video. Specifically, figure skating broadcast footage. 

The structure of the project is as follows:


```
├───src
│   ├───classes
│   │   └───data_accumulator.py
│   │   └───pose_estimator.py
│   ├───helpers
│   │   └───constants.py
│   │   └───labeler.py
│   │   └───output_func.py
│   ├───model
│   │   └───eval.py
│   │   └───testing.py
│   │   └───training.py
│   ├───output
│   │   ├───diagrams
│   │   ├───graphs
│   │   ├───labels
│   │   │   └───csv
│   │   ├───ml
│   │   │   ├───fvs
│   │   │   └───models
│   │   ├───pose
│   │   └───testing_video
│   ├───plots
│   │   └───visualisations.py
│   ├───ui
│   │   └───mode_init.py
│   │   └───model_setup.py
│   │   └───stdin_management.py
│   └───main.py
└───tests
│   └───test_da.py
│   └───test_module_main.py
└───requirements.txt
└───README.md
```

The directories in the projects are as follows:

*src* - contains all the code files, the system generated files and models trained with the extracted data.

*tests* - contains the test files from unit testing. 

The directory, into which all the generated files are saved is called *output* and it's contained within *src*.

To run the program, first, make sure you have Python 3.6+ installed. If there are any errors, try upgrading to 3.10, using pipenv or the like Python manager.

Then, run `pip install requirements.txt`. It should install all the third-party libraries for you.

From there, navigate to *src* folder, if you want, and run `python main.py` or equivalent. A CLI will appear. 
Follow the menu to explore the functionality of the program. 

If you want to get access to the 500 Figure Skating videos, which were being used throughout this dissertation, please visit
the following [GitHub repo](https://github.com/loadder/MS_LSTM).



