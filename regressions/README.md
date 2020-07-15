## Mosquito Regressions

The idea behind this notebook is to combine data from the Mosquito Habitat Mapper with data from another GLOBE protocol, such as air temperature or precipitation.  The goal is to provide tools for examining the relationship between the two protocols using Weighted Least Squares Regression.  

It is available for distribution in three different ways:


### MyBinder:

https://mybinder.org/v2/git/https%3A%2F%2Fmattbandel%40bitbucket.org%2Fmattbandel%2Fglobe-mosquitoes-regressions.git/master?filepath=regressions.ipynb

This link will open a copy of the notebook in your browser.  It is hosted by a free service called MyBinder.org.  The advantage is that the Python environment is already established for you.  The disadvantage is that resources are limited, and so the connection is lost after 10 minutes of inactivity.  

You may save any changes you made, even after the connection is lost, by clicking the leftmost cloud icon in the toolbar.  You may then reopen using the above link (not by simply reloading the page, as it turns out), and import your saved changes with the rightmost cloud icon.  Often this results in opening all the code blocks, which you may close with the eye icon in the toolbar.


### Docker:

Another way to access the notebook is to download a containerized version that packages all the requirements but allows you to host it yourself.  This requires installing an application called Docker.  You may do so at the following site:

https://www.docker.com/get-started

You will need to create a username and password to access the DockerHub.  Install the application, launch it, and sign in.  Open a Terminal window and login with:

	docker login

Pull the notebook image from DockerHub with:

	docker pull matthewbandel/regressions:latest

Then start up the notebook with:

	docker run -p 8888:8888 matthewbandel/regressions:latest 

You will get a link to browser window and will be able to access the notebook.  There is a token field in the link that may be used to login again as long as the server is still running.  Unfortunately, it is not yet capable to save changes in a more permanent way.  Once the server has stopped, the changes will be gone.  This is being looked into.


### Python:

Alternatively, if you already have a Python environment, you may work directly with a copy of the notebook, using these four files:

	•	regressions.ipynb, the notebook itself
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

	•	regressions.py
	
is provided as a working copy of the notebook's python code.


Questions?

This was developed using Python version 3.6.5 on MacOS with Chrome as the browser.  I am hoping it will install correctly on most versions of Python 3, and will run on most browsers, but this has by no means been rigorously tested yet.  As with the first notebook, we are expecting some issues across browsers and platforms. 

I’ve tried to put some tips in the “Notes on Navigation” section.  Otherwise, feel free to direct any feedback, issues, or questions my way at matthew.bandel@ssaihq.com.  Thank you so much!

	•	Matt