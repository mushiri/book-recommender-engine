# Book Recommender Engine

The following project is about recommendation systems. More specifically, recommendation systems
used for books. The following steps are used for running the application. In addition, there is a PythonAnywhere website that can be accessed to anyone.
I have used Pycharm (Python IDE) when developing the web app. All the commands used here are for macOS.

**Running the Code Locally**
- Make sure that the `pip` command is installed
- Install the virtualenv package using `pip install virtualenv`  
- Create a virtual environment by typing `virtualenv <choose a name>` I used venv for the name which will then be `virtualenv venv` 
- Activate venv using `source venv/bin/activate` (you may want to change venv to the name you have chosen)
- Install requirements using `pip install -r requirements.txt`
- Export Flask using `export FLASK_APP=flask_app.py`
- Run the app using `flask run`
- Proceed to http://127.0.0.1:5000/ to view the application (it takes some time to load)

**PythonAnywhere Website**

The code is also available for the public and can be accessed through https://bookrecommenderengine.pythonanywhere.com/
(Note that the website is a bit slow compared to the local version)