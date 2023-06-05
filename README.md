# rl-excavator-planning
Install the Terra environment with:
```
pip install terra
```
or if not available, clone the Terra repo and install it from the Terra folder with:
```
pip install -e .
```

## Visualization
If you want to render the environment, you are going to need the following:

on Ubuntu:
~~~
sudo apt install libcairo2-dev
sudo apt install libgirepository1.0-dev
~~~

~~~
matplotlib
PyGObject (you could use another backend like PyQt5 but it will clash with cv2 as it's QT-based)
~~~
