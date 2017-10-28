= Virtual Environment =
We use venv to setup virtual environment. To create a venv or update it, run
venv.sh. To activate the venv from the command line, run source env/bin/activate.

= Visdom =
We use visdom to output learning diagnostics. To run it, active the environment,
and use `python3 -m visdom.server` command. After this, you will be to access it
via http://localhost:8097