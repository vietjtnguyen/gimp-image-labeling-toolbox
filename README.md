GIMP Image Labeling Toolbox
===========================

Image labeling is a common task in computer vision where humans are used to generate a ground truth data set to act both as training data for computer vision algorithms and testing benchmarks for methods that perform semantic segmentation algorithms. The label here is an assignment of a value (or possibly multiple values) to each pixel in the image. These values are usually integers which map to semantic categories such as "train" and "person". Since labels are assigned to each pixel the task is inherently a *painting* task. It then makes sense to use a painting program to perform hand labeling. This toolbox seeks to facilitate this by working with [the GNU Image Manipulation Program (GIMP)](http://www.gimp.org/) thus providing access to the following features and more:

* Canvas zoom and pan
* Pencil tool with multiple brushes and variable brush sizes
* Free form and polygonal select tool
* Color wand select tool
* Color select tool
* Selection morphology operators
* Selection color fill
* Foreground extraction tool
* Multiple blend modes and blend control

The toolbox is written as a Python plugin for GIMP. Its interface is built using [PyGTK](http://www.pygtk.org/) which is a Python wrapper for the [Gnome Toolkit (GTK)](http://www.gtk.org/). It currently exists as a [single Python script](https://github.com/vietjtnguyen/gimp-image-labeling-toolbox/blob/master/gimp/label-toolbox.py).

Dependencies
------------

* [NumPy](http://www.numpy.org/) for array manipulation (required)
* [SciPy](http://www.scipy.org/) for [MATLAB `.mat` file I/O](http://docs.scipy.org/doc/scipy/reference/tutorial/io.html) (required)
* [Scikit-Image](http://scikit-image.org/) for [SLIC segmentation algorithm](http://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=slic#skimage.segmentation.slic) (optional)

The plugin was developed and tested on Ubuntu 13.10 and 12.04 with GIMP 2.8. Installation of NumPy and SciPy can be taken care of easy on Ubuntu using the following:

```
sudo apt-get install python-pip python-setuptools python-numpy python-scipy
sudo pip install numpy scipy
```

For Mac OS X and Windows consider using [the binary package installers provided by the maintainers](http://www.scipy.org/install.html#individual-binary-and-source-packages).

Also uses `appdirs`, specifically the file https://raw.githubusercontent.com/ActiveState/appdirs/2727a1b0444405a8728052512f02f26884528d64/appdirs.py included directly. Thus I need to honor the [MIT License](https://github.com/ActiveState/appdirs/blob/master/LICENSE.txt) appropriately.

Installation
------------

Assuming GIMP 2.8 is installed, installation of the toolbox on Ubuntu is just a matter of creating a symlink to or copying both `label-toolbox.py` and `appdirs.py` to `$HOME/.gimp-2.8/plug-ins`. *On Linux and Mac OS X remember to give `label-toolbox.py` executable permissions (e.g. `chmod ugo+x label-toolbox.py`).*

### Ubuntu

The following instructions are "from memory" and should work but have not been tested on a fresh machine.

1. Python 2.7 should already be available.
2. Install `pip` and `gcc`.
    - `sudo apt-get install -y python-pip python-dev python-setuptools`
    - `sudo apt-get install -y gcc g++ gfortran make`
3. Install NumPy and SciPy.
    - `sudo apt-get install -y cython python-numpy python-scipy`
    - `sudo pip install --upgrade cython`
    - `sudo pip install --upgrade numpy`
    - `sudo pip install --upgrade scipy`
4. *[OPTIONAL]* Install SciKit-Image.
    - `sudo pip install scikit-image`
5. Install GIMP 2.8.
    - <http://www.gimp.org/downloads/>
    - If you're on Ubuntu 12.04 you can get an updated PPA [here](http://www.webupd8.org/2013/06/install-gimp-286-in-ubuntu-ppa.html).
6. Run GIMP 2.8 so that it can initialize itself and the appropriate plug-in folder.
7. Install plug-in.
    - `cd $HOME/.gimp-2.8/plug-ins/`
    - `git clone https://github.com/vietjtnguyen/gimp-image-labeling-toolbox.git`
    - `ln -s gimp-image-labeling-toolbox/gimp/appdirs.py .`
    - `ln -s gimp-image-labeling-toolbox/gimp/label-toolbox.py .`
8. Close GIMP and reopen it.
9. Create a new, blank image (size doesn't matter, an active image is required to open the toolbox).
10. Open the toolbox via the file menu `Toolbox > Labeling`.

### Windows

The following instructions worked in my testing for Windows 7 64-bit and Windows 8 64-bit.

1. Install Python 2.7 **32-bit**.
    - <https://www.python.org/download/releases/2.7.6>
    - It is important that you install the 32-bit version (e.g. <https://www.python.org/ftp/python/2.7.6/python-2.7.6.msi>) because the Python that GIMP installs on its own is 32-bit.
2. Install NumPy for Python 2.7 32-bit.
    - <http://sourceforge.net/projects/numpy/files/NumPy/1.7.2/>
    - <http://sourceforge.net/projects/numpy/files/NumPy/1.7.2/numpy-1.7.2-win32-superpack-python2.7.exe/download>
3. Install SciPy for Python 2.7 32-bit.
    - <http://sourceforge.net/projects/scipy/files/scipy/0.14.0/>
    - <http://sourceforge.net/projects/scipy/files/scipy/0.14.0/scipy-0.14.0-win32-superpack-python2.7.exe/download>
4. *[OPTIONAL]* Install SciKit-Image for Python 2.7 32-bit.
    - <http://scikit-image.org/download>
    - <http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image>
5. Install GIMP 2.8.
    - <http://www.gimp.org/downloads/>
    - <http://download.gimp.org/pub/gimp/v2.8/windows/gimp-2.8.10-setup.exe>
6. Run GIMP 2.8 so that it can initialize itself and the appropriate plug-in folder.
7. Download the following source files to `C:\Users\<your user name>\.gimp-2.8\plug-ins\`.
    - [label-toolbox.py](https://raw.githubusercontent.com/vietjtnguyen/gimp-image-labeling-toolbox/master/gimp/label-toolbox.py)
    - [appdirs.py](https://raw.githubusercontent.com/vietjtnguyen/gimp-image-labeling-toolbox/master/gimp/appdirs.py)
8. Close GIMP and reopen it.
9. Create a new, blank image (size doesn't matter, an active image is required to open the toolbox).
10. Open the toolbox via the file menu `Toolbox > Labeling`.

What's going on here is GIMP 2.8 installs *its own Python binary*. As a result any dependencies (e.g. NumPy, SciPy) you install on your machine are available to the Python 2.7 you install (`C:\Python27` by default) but not immediately available to GIMP and the Python plug-ins it runs. What we do here is install Python 2.7 as an install target for our dependencies. The toolbox will then update its `sys.path` to look for the dependencies in the normally installed Python 2.7 (see [commit `c910e15dbb`](https://github.com/vietjtnguyen/gimp-image-labeling-toolbox/blob/7f8e68ae67546b09e87f2ccfd988338b2dfd93f3/gimp/label-toolbox.py#L27)). Since we match the Python binary that GIMP installs (v2.7, 32-bit) the dependencies will also work for the GIMP Python install.

### Mac OS X

Installation on Mac OS X still requires further testing. GIMP appears to also install its own Python binary on Mac OS X.

Directory Structure
-------------------

When the toolbox opens an image, say at `/path/to/image/my-image.jpg`, it does four things (see `openImageButtonClicked`):

1. Loads the label name to label integer mapping file (see `loadMetaData`). It looks for this file at `/path/to/image/../map.txt`.
2. Loads the original image (see `loadImage`). This is the image that is opened so the toolbox already knows where it is.
3. Loads the label `.mat` file (see `loadLabelMat`). The toolbox looks for this at `/path/to/image/../label/my-image.mat`.
4. Loads the comment associated with the image (see `loadComment`). The toolbox finds this file at `/path/to/image/../comment/my-image.txt`.

The `/label` folder contains the `.mat` array files that store the labels themselves. The `/comment` folder contains text files (simple UTF-8 text files) that have the per-image comments. A text file per-image is *optional*; the folder could be empty to begin with.

### Example

```
/path/to/map.txt
/path/to/image/2010_002080.jpg
/path/to/comment/2010_002080.txt
/path/to/label/2010_002080.mat
```

Workflow
--------

1. Open GIMP
2. Create a new blank image
3. Open `Toolbox > Labeling` in the menu
4. Open image using toolbox's `Open Image` button
5. Edit label image *on label layer*, avoiding anti-aliasing everywhere
6. Save label using toolbox's `Save MAT Label` button
7. Repeat step 4 for more images, optionally using the `Previous` and `Next` buttons.

Usage
-----

The open/save state of an image when using the toolbox exists *independently* of GIMP's open/save states. When using the toolbox use only the `Open Image` and `Save MAT Label` buttons on the toolbox.

When the toolbox loads an image it will load the original image into a layer named `Original` and load all of the label layers on top of the `Original` layer. The color map is randomized on load and can be shuffled further using the `Shuffle Colors` button. *The only changes that are saved are edits to layers whose name starts with `Label` (e.g. `LabelGrass`, `Label_wabalabaDUBDUB`.* Changes to any other layer are discarded/ignored.

Remember that ***tools will only operate on the currently selected layer***. The list of layers should be in the interface by default. If not you can bring it up using `Ctrl+L`.

I personally suggest using [GIMP in single-window mode](http://docs.gimp.org/2.8/en/gimp-concepts-main-windows.html). This can be activated at `Windows > Single-Window Mode`.

***Users must pay attention to whether or not their selection is anti-aliased.*** Anti-aliased selections will result in corrupted labels since blended label colors have no semantic meaning and will most likely fall outside the label color map. Be wary of the anti-aliased options on your selection tool or use the `Un-Antialias` button in the toolbox under `Selection Helper`.

The toolbox window will automatically float on top of the GIMP window ***but will still capture focus***. It is an entirely separate application from GIMP. This means if the toolbox has window focus ***then GIMP keyboard shortcuts will not work***. To use keyboard shortcuts you must bring the GIMP window back in focus by clicking on it or Alt-Tabbing to it.

### Miscellany

* The toolbox is currently very tall. Fully expanded it is nearly 1000 pixels tall. Sections can be collapsed to help it fit on smaller screens.
* I would have put buttons on the toolbox for the common tools but the GIMP plugin API does not provide functionality for selecting tools.
* The comment text field is *saved on the fly*. Each edit updates the text file ***immediately***.

Screenshots
-----------

![Fully expanded toolbox](https://raw.githubusercontent.com/vietjtnguyen/gimp-image-labeling-toolbox/master/docs/expanded-toolbox.png)

TODO
----

* Write a full tutorial
* Document `.mat` label array format
* Document Super Pixel Helper
* Reduce toolbox height, maybe use two columns?
* Support alternative label formats (e.g. `.ipy`, packed binary integers, etc.)
* Add configuration file for supporting options such as default blend mode, etc.

