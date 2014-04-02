# GIMP Image Labeling Toolbox

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

## Dependencies

* [NumPy](http://www.numpy.org/) for array manipulation (required)
* [SciPy](http://www.scipy.org/) for [MATLAB `.mat` file I/O](http://docs.scipy.org/doc/scipy/reference/tutorial/io.html) (required)
* [Scikit-Image](http://scikit-image.org/) for [SLIC segmentation algorithm](http://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=slic#skimage.segmentation.slic) (optional)

Installation of NumPy and SciPy can be taken care of easy on Ubuntu using the following:

```
sudo apt-get install python-pip python-setuptools python-numpy python-scipy
sudo pip install numpy scipy
```

For Mac OS X and Windows consider using [the binary package installers provided by the maintainers](http://www.scipy.org/install.html#individual-binary-and-source-packages).

## Cross Platform

The plugin was developed and tested on Ubuntu 13.10 with GIMP 2.8. On this system installation is just a matter of creating a symlink to or copying `label-toolbox.py` to `$HOME/.gimp-2.8/plug-ins`.

Assuming the required dependencies of NumPy and SciPy are taken care of the plugin should theoretically work on both Mac OS X and Windows since GIMP and Python plugins for GIMP are supported on both of those platforms. This cross platform capability has not been tested yet and will likely require some finesse and further development. For generic GIMP plugin installation instructions for other platforms see [here](http://en.wikibooks.org/wiki/GIMP/Installing_Plugins#Copying_the_plugin_to_the_GIMP_plugin_directory).

## Usage

1. Open GIMP
2. Create a new blank image
3. Open `Toolbox > Labeling` in the menu
4. Open image using toolbox's `Open Image` button
5. Edit label image *on label layer*
6. Save label using toolbox's `Save MAT Label` button
7. Repeat step 4 for more images

## Directory Structure

When the toolbox opens an image, say at `/path/to/my-image.jpg`, it does four things (see `openImageButtonClicked`):

1. Loads the label name to label integer mapping file (see `loadMetaData`). It looks for this file at `/path/to/label-mat/map.txt`.
2. Loads the original image (see `loadImage`). This is the image that is opened so the toolbox already knows where it is.
3. Loads the label `.mat` file (see `loadLabelMat`). The toolbox looks for this at `/path/to/label-mat/my-image.mat`.
4. Loads the comment associated with the image (see `loadComment`). The toolbox finds this file at `/path/to/comment-txt/my-image.txt`.

### Example

```
/path/to/2010_002080.jpg
/path/to/comment-txt/2010_002080.txt
/path/to/label-mat/2010_002080.mat
/path/to/label-mat/map.txt
```

## Caveats

* I personally suggest using [GIMP in single-window mode](http://docs.gimp.org/2.8/en/gimp-concepts-main-windows.html).
* The toolbox window will automatically float on top of the GIMP window ***but will still capture focus***.
* The toolbox window is an entirely separate application from GIMP. This means if the toolbox has window focus ***then GIMP keyboard shortcuts will not work***. To use keyboard shortcuts you must bring the GIMP window back in focus by clicking on it or Alt-Tabbing to it.
* ***Users must pay attention to whether or not their selection is anti-aliased.*** Anti-aliased selections will result in corrupted labels since blended label colors have no semantic meaning and will most likely fall outside the label color map. Be wary of the anti-aliased options on your selection tool or use the `Un-Antialias` button in the toolbox under `Selection Helper`.
* The toolbox is currently very tall. Fully expanded it is nearly 1000 pixels tall. Sections can be collapsed to help it fit on smaller screens.
* I would have put buttons on the toolbox for the common tools but the GIMP plugin API does not provide functionality for selecting tools.
* The comment text field is *saved on the fly*. Each edit updates the text file ***immediately***.

## Screenshots

![Fully expanded toolbox](https://raw.githubusercontent.com/vietjtnguyen/gimp-image-labeling-toolbox/master/docs/expanded-toolbox.png)

## TODO

* Cross-platform testing
* Support multiple layers of labels
* Write a full tutorial
* Document `.mat` label array format
* Document Super Pixel Helper
* Reduce toolbox height, maybe use two columns?
* Support alternative label formats (e.g. `.ipynb`, packed binary integers, etc.)
* Add previous/next buttons for quick navigation
* Add configuration file for supporting options such as default blend mode, etc.
* Add an invalid label sanity check

