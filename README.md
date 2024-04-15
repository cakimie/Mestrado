# Python environment
This is my Python/Jupyter environment.

# Executing the code
It's strongly suggested to utilize a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to install project dependencies in a sandbox. The following command may be used to do that:

```
python3 -m venv .venv
```

Then, to activate the virtual environment in `bash` for the current shell session, run the following command:

```
source .venv/bin/activate
```

With this environment actived, it is possible to automatically install all dependencies with the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

One can also list and operate over modules using the following commands:

```
pip install somepackage             # Installs new package.
pip install --upgrade somepackage   # Upgrades existing packages.
pip freeze > requirements.txt       # Updates the `requirements.txt` file.
```

After which the `requirements.txt` file must be committed.

While using `pip` (especially `pip list`) a *WARNING* may be issue regarding `pip` version. One can simply use the following command to upgrade it in order for the *WARNING* to stop appearing:

```
pip install --upgrade pip
```

Finally, to deactivate this environment (restoring the previous shell session one), the following command may be used:

```
deactivate
```

At which point the environment folder can be destroyed if necessary:

```
rm -rf .venv/
```

## Setting up Visual Studio Code

The activation (and "deactivation") of the virtual environment can be dealt automatically by Visual Studio Code's `Python` extension. All you have to do is accept the installation of the suggested extension (or search for it in the "Extensions" tab), then open the project folder (or restart VS Code).

Also, you should install the `Jupyter` extension.