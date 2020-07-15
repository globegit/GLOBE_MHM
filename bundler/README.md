### Mosquito Bundler

The purpose of this notebook is to allow the user to select a data range and polygon on a map to retrieve all the data from the mosquito bundle protocols for the selection.

It is available for distribution in three different ways:


### MyBinder:

https://mybinder.org/v2/git/https%3A%2F%2Fmattbandel%40bitbucket.org%2Fmattbandel%2Fglobe-mosquitoes-bundler.git/master?filepath=bundler.ipynb

This link will open a copy of the notebook in your browser.  It is hosted by a free service called MyBinder.org.  The advantage is that the Python environment is already established for you.  The disadvantage is that resources are limited, and so the connection is lost after 10 minutes of inactivity.  

You may save any changes you made, even after the connection is lost, by clicking the leftmost cloud icon in the toolbar.  You may then reopen using the above link (not by simply reloading the page, as it turns out), and import your saved changes with the rightmost cloud icon.  Often this results in opening all the code blocks, which you may close with the eye icon in the toolbar.


### Docker:

Another way to access the notebook is to download a containerized version that packages all the requirements but allows you to host it yourself.  This requires installing an application called Docker.  You may do so at the following site:

https://www.docker.com/get-started

You will need to create a username and password to access the DockerHub.  Install the application, launch it, and sign in.  Open a Terminal window and login with:

	docker login

Pull the notebook image from DockerHub with:

	docker pull matthewbandel/bundler:latest

Then start up the notebook with:

	docker run -p 8888:8888 matthewbandel/bundler:latest 

You will get links to the browser looking something like this:
 
    The Jupyter Notebook is running at:
    http://cecb97358251:8888/?token=f418eb5ae31a84affd7b44e96926ab91521d8b1c96d51b9c
    or http://127.0.0.1:8888/?token=f418eb5ae31a84affd7b44e96926ab91521d8b1c96d51b9c
    
The second link will open up the home page from which you may access the notebook by clicking on 'regressions.ipynb.'  

Save any changes you make under File menu, Save and Checkpoint.  At the homepage, you may download the notebook, using the token given in the link above.

When starting a new session, you may reupload this saved copy to reinstate any changes you made.


### Python:

Alternatively, if you already have a Python environment, you may work directly with a copy of the notebook, using these four files:

	•	bundler.ipynb, the notebook itself
	•	requirements.txt, a list of Python modules required to run the code
	•	jupyter_notebook_config.py, a Python file with some configuration settings
	•	protocols.txt, a list of mosquito bundle protocols

If you’re on a Mac, you may follow these steps to install:

Assuming you are in a Terminal window in the same directory with these files, running:

	pip install -r requirements.txt

will install all the required Python modules.  Then running:

	jupyter notebook

will fire up the Jupyter home page in your default browser.  You should be able to click on regressions.ipynb to open the notebook.  You may click on “Save and Checkpoint” from the File menu to save any changes.  Typing Control-C from Terminal will shut it down.  

A fifth file:

	•	bundler.py
	
is provided as a working copy of the notebook's python code.


Questions?

This was developed using Python version 3.6.5 on MacOS with Chrome as the browser.  I am hoping it will install correctly on most versions of Python 3, and will run on most browsers, but this has by no means been rigorously tested yet.  As with the first notebook, we are expecting some issues across browsers and platforms. 

I’ve tried to put some tips in the “Notes on Navigation” section.  Otherwise, feel free to direct any feedback, issues, or questions my way at matthew.bandel@ssaihq.com.  Thank you so much!

	•	Matt