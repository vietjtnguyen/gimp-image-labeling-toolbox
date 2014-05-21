Directory Structure
-------------------

When an image is opened the toolbox will automatically infer the location of the comment file and label file according to a predefined directory structure.

Say our image is located at `/my/path/image/image01.jpg`. The toolbox will then look for the following:

* Comment file at `/my/path/comment/image01.txt`
* Label file at `/my/path/label/image01.mat`
* Label mapping file at `/my/path/map.txt`

Another way to think of it is the toolbox treats `/my/path` as a root folder for the dataset where the root folder contains the following three folders: `image`, `comment`, and `label`.
