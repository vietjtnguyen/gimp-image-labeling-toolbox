#!/usr/bin/env python

import gimp
import gimpfu
from gimpfu import pdb
import gimpui
import gimpenums

import gtk
import gobject

import array
import colorsys
import datetime
import os.path
import sys

try:
  import numpy as np
  from scipy.io import savemat, loadmat
  imports_succeeded = True
except ImportError:
  imports_succeeded = False

try:
  from skimage.segmentation import slic
  slic_available = True
except ImportError:
  slic_available = False

# for debugging
#sys.stderr = open('/home/vnguyen/.widget-toolbox-err.log', 'w', buffering=0)
#sys.stdout = open('/home/vnguyen/.widget-toolbox-log.log', 'w', buffering=0)

def make_colormap(n):
  i = np.arange(n)
  iii = np.array(np.vstack((i, i, i)).T, dtype='float64')
  fff = iii / n
  # TODO: Explain where this 800 comes from (rough discretization threshold)
  if n <= 800:
    hsv = np.apply_along_axis(lambda x: (x[0], 1.0, 0.8), 1, fff)
  else:
    # TODO: Explain what's going on here
    # can produce 2204 unique colors
    period = 512.0 / float(n)
    freq = float(n) / 512.0
    hsv = np.apply_along_axis(lambda x: (np.fmod(x[0], period) / period,
                                        1.0,
                                        ( 1.0 - np.floor(x[0] / period) * period ) * 0.6 + 0.4),
                              1,
                              fff)
  rgbf = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), 1, hsv)
  rgb = np.array(rgbf * 255, dtype='uint32')
  # verify that the mapping is unique
  if not len(np.unique(rgb[:, 0] * 256 * 256 + rgb[:, 1] * 256 + rgb[:, 2])) == n:
    gimp.message('Cannot produce a unique color map for {:} values!'.format(n))
  return np.array(rgb, dtype='uint8')

class LabelToolbox(gtk.Window):

  def __init__ (self, image, *args):
    self.is_image_open = False
    self.image_full_path = ''
    self.working_path = ''
    self.image_filename = ''
    self.image_name = ''
    self.image_extension = ''
    self.image = image
    self.original_layer = None
    self.label_layer = None
    self.last_foreground_color = None
    self.only_available_with_open_image = []
    self.selection_interface = []
    self.slic_interface = []

    # TODO: Support multiple label layers
    # TODO: Change this according to number of labels
    self.num_of_labels = 1024
    self.colormap = make_colormap(self.num_of_labels)
    self.shufflemap = np.arange(self.num_of_labels)
    self.shuffle()

    self.label_int_to_name_map = {}
    self.label_name_to_int_map = {}

    window = gtk.Window.__init__(self, *args)
    self.show()
    self.set_border_width(4)
    self.set_keep_above(True)
    self.set_resizable(False)
    self.connect('destroy', gtk.main_quit)

    container = [self]

    widget = gtk.VBox(spacing=4, homogeneous=False)
    widget.show()
    widget.set_size_request(200, -1)
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.Label('Current Image')
    widget.show()
    container[-1].add(widget)

    widget = self.image_name_box = gtk.Entry()
    widget.set_editable(False)
    widget.show()
    container[-1].add(widget)

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.open_image_button = gtk.Button('Open Image')
    widget.show()
    widget.connect('clicked', self.openImageButtonClicked)
    container[-1].add(widget)

    widget = self.save_label_png_button = gtk.Button('Save PNG Label')
    widget.show()
    widget.connect('clicked', self.saveLabelPngButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    container.pop()

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.load_label_mat_button = gtk.Button('Load MAT Label')
    widget.show()
    widget.connect('clicked', self.loadLabelMatButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    widget = self.save_label_mat_button = gtk.Button('Save MAT Label')
    widget.show()
    widget.connect('clicked', self.saveLabelMatButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    container.pop()

    widget = gtk.HSeparator()
    widget.show()
    container[-1].add(widget)

    widget = gtk.Label('Current Foreground Color Label')
    widget.show()
    container[-1].add(widget)

    widget = self.current_label = gtk.Label()
    widget.set_text('Hello world!')
    widget.show()
    container[-1].add(widget)

    def completion_match_func(widget, key, tree_iter):
      model = widget.get_model()
      text = model.get_value(tree_iter, 0)
      return text.startswith(key) or text.find(' ' + key) > -1 or text.find('_' + key) > -1

    self.completion = gtk.EntryCompletion()
    self.liststore = gtk.ListStore(str)
    self.completion.set_model(self.liststore)
    self.completion.set_text_column(0)
    self.completion.set_match_func(completion_match_func)
    self.completion.connect('match-selected', self.completionMatchSelected)

    widget = gtk.Expander('Select/Shuffle Label')
    widget.show()
    widget.set_expanded(True)
    widget.set_resize_mode(gtk.RESIZE_PARENT)
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.VBox(spacing=4, homogeneous=False)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.label_name = gtk.Entry()
    widget.set_completion(self.completion)
    widget.connect('activate', self.labelNameActivated)
    widget.show()
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.select_label_button = gtk.Button('Select Label')
    widget.show()
    widget.connect('clicked', self.selectLabelButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    widget = self.shuffle_colors_button = gtk.Button('Shuffle Colors')
    widget.show()
    widget.connect('clicked', self.shuffleColorsButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    container.pop()

    container.pop()

    container.pop()

    widget = gtk.HSeparator()
    widget.show()
    container[-1].add(widget)

    widget = gtk.Expander('Visibility/Opacity')
    widget.show()
    widget.set_expanded(True)
    widget.set_resize_mode(gtk.RESIZE_PARENT)
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.VBox(spacing=4, homogeneous=False)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.label_opacity_slider = gtk.HScale()
    widget.set_range(0.0, 100.0)
    widget.set_value(100.0)
    widget.connect('change-value', self.labelOpacitySliderChange)
    widget.show()
    container[-1].add(widget)

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.toggle_label_button = gtk.Button('Toggle Label')
    widget.show()
    widget.connect('clicked', self.toggleLabelButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    widget = self.normal_blend_button = gtk.Button('Normal Blend')
    widget.show()
    widget.connect('clicked', self.normalBlendButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    container.pop()

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.grain_blend_button = gtk.Button('Grain Blend')
    widget.show()
    widget.connect('clicked', self.grainBlendButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    widget = self.color_blend_button = gtk.Button('Color Blend')
    widget.show()
    widget.connect('clicked', self.colorBlendButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    container.pop()

    container.pop()

    container.pop()

    widget = gtk.HSeparator()
    widget.show()
    container[-1].add(widget)

    widget = gtk.Expander('Selection Helper')
    widget.show()
    widget.set_expanded(True)
    widget.set_resize_mode(gtk.RESIZE_PARENT)
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.VBox(spacing=4, homogeneous=False)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.label_selection_button = gtk.Button('Fill Label')
    widget.show()
    widget.connect('clicked', self.labelSelectionButtonClicked)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)
    self.selection_interface.append(widget)

    widget = self.clear_selection_button = gtk.Button('Clear')
    widget.show()
    widget.connect('clicked', self.clearSelectionButtonClicked)
    container[-1].add(widget)
    self.selection_interface.append(widget)

    container.pop()

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.harden_selection_button = gtk.Button('Un-Antialias')
    widget.show()
    widget.connect('clicked', self.hardenSelectionButtonClicked)
    container[-1].add(widget)
    self.selection_interface.append(widget)

    widget = self.smooth_selection_button = gtk.Button('Smooth')
    widget.show()
    widget.connect('clicked', self.smoothSelectionButtonClicked)
    container[-1].add(widget)
    self.selection_interface.append(widget)

    container.pop()

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.invert_selection_button = gtk.Button('Invert')
    widget.show()
    widget.connect('clicked', self.invertSelectionButtonClicked)
    container[-1].add(widget)
    self.selection_interface.append(widget)

    widget = self.grow_selection_button = gtk.Button('Grow')
    widget.show()
    widget.connect('clicked', self.growSelectionButtonClicked)
    container[-1].add(widget)
    self.selection_interface.append(widget)

    widget = self.shrink_selection_button = gtk.Button('Shrink')
    widget.show()
    widget.connect('clicked', self.shrinkSelectionButtonClicked)
    container[-1].add(widget)
    self.selection_interface.append(widget)

    container.pop()

    container.pop()

    container.pop()

    widget = gtk.HSeparator()
    widget.show()
    container[-1].add(widget)

    widget = gtk.Expander('Super Pixel Helper')
    widget.show()
    widget.set_expanded(True)
    widget.set_resize_mode(gtk.RESIZE_PARENT)
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.VBox(spacing=4, homogeneous=False)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.VBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.Label('# of regions')
    widget.show()
    container[-1].add(widget)

    widget = self.slic_n = gtk.Entry()
    widget.show()
    widget.set_text('100')
    widget.set_sensitive(slic_available)
    container[-1].add(widget)
    self.slic_interface.append(widget)

    container.pop()

    widget = gtk.VBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.Label('Compactness')
    widget.show()
    container[-1].add(widget)

    widget = self.slic_compactness = gtk.Entry()
    widget.show()
    widget.set_text('1.0')
    widget.set_sensitive(slic_available)
    container[-1].add(widget)
    self.slic_interface.append(widget)

    container.pop()

    container.pop()

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.VBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.Label('Smoothing Sigma')
    widget.show()
    container[-1].add(widget)

    widget = self.slic_sigma = gtk.Entry()
    widget.show()
    widget.set_text('0.0')
    widget.set_sensitive(slic_available)
    container[-1].add(widget)
    self.slic_interface.append(widget)

    container.pop()

    widget = gtk.VBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.Label('Colorspace')
    widget.show()
    container[-1].add(widget)

    widget = self.slic_lab = gtk.ToggleButton('Using LAB')
    widget.show()
    widget.set_active(True)
    widget.set_sensitive(slic_available)
    widget.connect('toggled', self.slicColorSpaceButtonToggled)
    container[-1].add(widget)
    self.slic_interface.append(widget)

    container.pop()

    container.pop()

    widget = gtk.HBox(spacing=4, homogeneous=True)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.create_segmentation_button = gtk.Button('Create')
    widget.show()
    widget.set_sensitive(slic_available)
    widget.connect('clicked', self.createSegmentationButtonClicked)
    container[-1].add(widget)
    self.slic_interface.append(widget)

    widget = self.clear_all_segmentations_button = gtk.Button('Clear All')
    widget.show()
    widget.set_sensitive(slic_available)
    widget.connect('clicked', self.clearAllSegmentationsButtonClicked)
    container[-1].add(widget)
    self.slic_interface.append(widget)

    container.pop()

    container.pop()

    container.pop()

    widget = gtk.HSeparator()
    widget.show()
    container[-1].add(widget)

    widget = gtk.Expander('Comment')
    widget.show()
    widget.set_expanded(True)
    widget.set_resize_mode(gtk.RESIZE_PARENT)
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.VBox(spacing=4, homogeneous=False)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    widget = self.comment = gtk.Entry()
    widget.show()
    widget.connect('changed', self.commentChanged)
    container[-1].add(widget)
    self.only_available_with_open_image.append(widget)

    container.pop()

    container.pop()

    widget = gtk.HSeparator()
    widget.show()
    container[-1].add(widget)

    widget = gtk.Expander('Shortcuts Reference')
    widget.show()
    widget.set_expanded(True)
    widget.set_resize_mode(gtk.RESIZE_PARENT)
    container[-1].add(widget)
    container.append(widget)

    widget = gtk.VBox(spacing=4, homogeneous=False)
    widget.show()
    container[-1].add(widget)
    container.append(widget)

    shortcuts = [
      'Keyboard:',
      '  N: Pencil',
      '  Shift+E: Eraser',
      '  O: Color Picker',
      '  U: Fuzzy Select (Color Wand)',
      '  F: Poly/Free Select',
      '  1, 2, 3, 4: Zoom',
      '  Selection+Shift: Union',
      '  Selection+Ctrl: Subtract',
      '  Selection+Ctrl+Shift: Intersection',
      '',
      'Hints:',
      '  Avoid all anti-aliasing',
      '  Hold Ctrl with Pencil to color pick',
      '  Hold Shift with Pencil for lines',
      '  Use [ and ] to change brush size',
      '  Use middle click to pan',
      '  When focused, hold Space to pan',
      '  Use Pencil instead of Brush',
      '  Turn on Hard Edge for Eraser',
    ]

    widget = gtk.Label('\n'.join(shortcuts))
    widget.set_alignment(0.0, 0.0)
    widget.justify = gtk.JUSTIFY_LEFT
    widget.show()
    container[-1].add(widget)

    container.pop()

    container.pop()

    self.show()

    gobject.timeout_add(100, self.update, self)  

    return window
  
  #### METHODS ####

  def alertDialog(self, message):
      alert = gtk.MessageDialog(self, gtk.DIALOG_DESTROY_WITH_PARENT, gtk.MESSAGE_ERROR, gtk.BUTTONS_CLOSE, message)
      alert.run()
      alert.destroy()

  def updateImagePaths(self, image_full_path):
      self.image_full_path = image_full_path
      self.working_path, self.image_filename = os.path.split(self.image_full_path)
      self.image_name, self.image_extension = os.path.splitext(self.image_filename)
      self.image_name_box.set_text(self.image_filename)

  def loadMetaData(self):
    label_map_filename = os.path.join(self.working_path, 'label-mat', 'map.txt')
    self.liststore.clear()
    self.label_int_to_name_map = {}
    self.label_name_to_int_map = {}
    with open(label_map_filename, 'r') as f:
      for line in f.readlines():
        num, name = line[:-1].split(': ')
        self.liststore.append([name])
        self.label_int_to_name_map[int(num)] = name
        self.label_name_to_int_map[name] = int(num)

  def loadImage(self):
    # get paths
    # get original image file names
    original_filename = os.path.join(self.working_path, self.image_filename)
    # load the original image
    try:
      original_image = pdb.gimp_file_load(original_filename, original_filename)
    except:
      self.alertDialog('Could not load file "{:}".'.format(original_filename))
      self.is_image_open = False
      self.resetInterface()
      return
    # clear layers
    while len(self.image.layers) > 1:
      pdb.gimp_image_remove_layer(self.image, self.image.layers[0])
    # resize image to fit original image
    pdb.gimp_image_resize(self.image, original_image.width, original_image.height, 0, 0)
    # copy from the original image
    pdb.gimp_edit_copy(original_image.layers[0])
    # paste original image as a floating selection over the "background"
    background = self.image.layers[0]
    floating_selection = pdb.gimp_edit_paste(background, True)
    # convert floating selection to a layer
    pdb.gimp_floating_sel_to_layer(floating_selection)
    # align the layer with the image
    new_layer = self.image.layers[0]
    new_layer.set_offsets(0, 0)
    # merge the new layer to the "background"
    pdb.gimp_selection_none(self.image)
    pdb.gimp_image_merge_down(self.image, new_layer, 0)
    # update the new "background" layer
    self.original_layer = self.image.layers[0]
    self.original_layer.name = 'Original'
    self.original_layer.merge_shadow(True)
    pdb.gimp_layer_resize_to_image_size(self.original_layer)
    self.original_layer.update(0, 0, self.image.width, self.image.height)
    self.original_layer.flush()
    # delete original image
    pdb.gimp_image_delete(original_image)
    # create a new label layer
    self.label_layer = pdb.gimp_layer_new(self.image, self.image.width, self.image.height, gimpenums.RGBA_IMAGE, 'Label', 100, gimpenums.NORMAL_MODE)
    pdb.gimp_image_insert_layer(self.image, self.label_layer, None, 0)
    # update gimp
    pdb.gimp_displays_flush()
    self.is_image_open = True

  def loadLabelMat(self):
    # get paths
    try:
      mat_filename = os.path.join(self.working_path, 'label-mat', self.image_name+'.mat')
    except:
      self.alertDialog('Could not load file "{:}".'.format(mat_filename))
      self.is_image_open = False
      self.resetInterface()
    else:
      self.int_label_image = loadmat(mat_filename)['LabelMap']
      self.updateInternalRgbLabelImage()
      self.pushInternalRgbLabelImageToLayer()
      # update gimp
      pdb.gimp_displays_flush()

  def saveLabelMat(self):
    # get paths
    mat_filename = os.path.join(self.working_path, 'label-mat', self.image_name+'.mat')
    gimp.progress_init('Saving labels as "{:}"...'.format(mat_filename))
    self.pullInternalRgbLabeImageFromLayer()
    gimp.progress_update(20)
    self.updateInternalIntLabelImage()
    gimp.progress_update(50)
    savemat(mat_filename, {'LabelMap': self.int_label_image}, do_compression=True)
    gimp.progress_update(100)
    pdb.gimp_progress_set_text('Saved labels as "{:}"!'.format(mat_filename))
    pdb.gimp_progress_end()

  def loadLabelPng(self):
    '''
    This method is not used because there is no color mapping information stored with PNG label images.
    '''
    # construct label filename
    label_filename = os.path.join(self.working_path, 'label-img', self.image_name+'.png')
    # load the label image
    try:
      label_image = pdb.gimp_file_load(label_filename, label_filename)
    except:
      self.alertDialog('Could not load file "{:}".'.format(label_filename))
      self.is_image_open = False
      self.resetInterface()
    else:
      # copy from the label image
      pdb.gimp_edit_copy(label_image.layers[0])
      # paste label image as a floating selection over the "background"
      floating_selection = pdb.gimp_edit_paste(self.original_layer, True)
      # convert floating selection to a layer
      pdb.gimp_floating_sel_to_layer(floating_selection)
      # align the layer with the image
      new_layer = self.image.layers[0]
      new_layer.set_offsets(0, 0)
      # merge the new layer to the label layer
      pdb.gimp_selection_none(self.image)
      pdb.gimp_image_merge_down(self.image, new_layer, 0)
      # update label layer
      self.label_layer = self.image.layers[0]
      self.label_layer.name = 'Label'
      # delete label image
      pdb.gimp_image_delete(label_image)
      # update gimp
      pdb.gimp_displays_flush()

  def saveLabelPng(self):
    label_filename = os.path.join(self.working_path, 'label-img', self.image_name+'.png')
    pdb.file_png_save(self.image,
                      self.label_layer,
                      label_filename,
                      label_filename,
                      False,
                      9,
                      False,
                      False,
                      False,
                      False,
                      True)

  def updateInternalRgbLabelImage(self):
    self.rgb_label_image = self.colormap[self.shufflemap[self.int_label_image]]

  def updateInternalIntLabelImage(self):
    try:
      for i in range(self.int_label_image.shape[0]):
        for j in range(self.int_label_image.shape[1]):
          self.int_label_image[i, j] = self.reversemap[tuple(self.rgb_label_image[i, j])]
    except KeyError:
      self.alertDialog('An unknown color was found in the label image. This most likely occured because some operation were performed with anti-aliasing. Could not proceed with operation.')

  def pushInternalRgbLabelImageToLayer(self):
    pdb.gimp_selection_none(self.image)
    pixel_region = self.label_layer.get_pixel_rgn(0, 0,
                                                  self.image.width, self.image.height,
                                                  True, True)
    rgba_label_image = np.dstack((self.rgb_label_image,
                                  np.ones((self.image.height, self.image.width),
                                          dtype='uint8') * 255))
    pixel_region[0:self.image.width, 0:self.image.height] = array.array('B', rgba_label_image.ravel()).tostring()
    self.label_layer.merge_shadow(True)
    self.label_layer.update(0, 0, self.image.width, self.image.height)
    self.label_layer.flush()
    pdb.gimp_displays_flush()

  def pullInternalRgbLabeImageFromLayer(self):
    pdb.gimp_selection_none(self.image)
    pixel_region = self.label_layer.get_pixel_rgn(0, 0,
                                                  self.image.width, self.image.height,
                                                  False, False)
    byte_array = array.array('B', pixel_region[0:self.image.width, 0:self.image.height])
    byte_array = np.array(byte_array, dtype='uint8')
    byte_array = byte_array.reshape(len(byte_array)/4, 4)
    # NOTE THE SWITCH IN INDEX ORDER
    rgba_label_image = byte_array.reshape(self.image.height, self.image.width, 4)
    self.rgb_label_image = rgba_label_image[:, :, :3]

  def shuffle(self):
    np.random.shuffle(self.shufflemap)
    self.reversemap = {}
    for i in range(self.num_of_labels):
      self.reversemap[tuple(self.colormap[self.shufflemap[i]])] = i

  def setForegroundColorFromLabelName(self):
    label_name = self.label_name.get_text()
    if self.label_name_to_int_map.has_key(label_name):
      color = self.colormap[self.shufflemap[self.label_name_to_int_map[label_name]]]
      gimp.set_foreground(tuple(map(lambda x: int(x), color)))

  def loadComment(self):
      comment_filename = os.path.join(self.working_path, 'comment-txt', self.image_name+'.txt')
      try:
        with open(comment_filename, 'r') as f:
          self.comment.set_text(f.read())
      except IOError:
        pass
  
  def resetInterface(self):
    if self.is_image_open:
      self.label_opacity_slider.set_value(100.0)

  def updateInterface(self):
    is_selection_active = not pdb.gimp_selection_is_empty(self.image)
    for widget in self.selection_interface:
      widget.set_sensitive(is_selection_active)
    if self.is_image_open:
      for widget in self.only_available_with_open_image:
        widget.set_sensitive(True)
      for widget in self.slic_interface:
        widget.set_sensitive(slic_available)
      self.label_opacity_slider.set_value(self.label_layer.opacity)
      foreground_color = tuple(gimp.get_foreground())[:3]
      if foreground_color != self.last_foreground_color:
        self.last_foreground_color = foreground_color
        if self.reversemap.has_key(foreground_color):
          foreground_name = self.label_int_to_name_map[self.reversemap[foreground_color]]
          self.current_label.set_text(foreground_name)
        else:
          self.current_label.set_text('{:} not found'.format(str(foreground_color)))
    else:
      for widget in self.only_available_with_open_image:
        widget.set_sensitive(False)
      for widget in self.slic_interface:
        widget.set_sensitive(False)

  #### GUI CALLBACKS ####

  def openImageButtonClicked(self, widget):
    dialog = gtk.FileChooserDialog(
        'Open Image...',
        None, 
        gtk.FILE_CHOOSER_ACTION_OPEN,
        (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))
    dialog.set_default_response(gtk.RESPONSE_OK)
    response = dialog.run()
    if response == gtk.RESPONSE_OK:
      self.updateImagePaths(dialog.get_filename())
      self.loadMetaData()
      self.loadImage()
      self.loadLabelMat()
      self.loadComment()
    dialog.destroy()

  def loadLabelMatButtonClicked(self, widget):
    self.loadLabelMat()

  def saveLabelMatButtonClicked(self, widget):
    self.saveLabelMat()

  # NOTE: This method no longer exists because a PNG does not store any color mapping information.
  #def loadLabelPngButtonClicked(self, widget):
  #  self.loadLabelPng()

  def saveLabelPngButtonClicked(self, widget):
    self.saveLabelPng()

  def completionMatchSelected(self, completion, model, iterator):
    self.setForegroundColorFromLabelName()

  def labelNameActivated(self, widget):
    self.setForegroundColorFromLabelName()

  def selectLabelButtonClicked(self, widget):
    self.setForegroundColorFromLabelName()
  
  def pickLabelButtonClicked(self, widget):
    #sys.stdout.write(datetime.datetime.isoformat(datetime.datetime.utcnow()) + '\n')
    pass

  def shuffleColorsButtonClicked(self, widget):
    self.pullInternalRgbLabeImageFromLayer()
    self.updateInternalIntLabelImage()
    self.shuffle()
    self.updateInternalRgbLabelImage()
    self.pushInternalRgbLabelImageToLayer()

  def labelOpacitySliderChange(self, widget, scroll, value):
    self.label_layer.opacity = min(100.0, max(0.0, value))
    self.label_layer.flush()
    pdb.gimp_displays_flush()

  def toggleLabelButtonClicked(self, widget):
    self.label_layer.visible = not self.label_layer.visible
    if self.label_layer.visible:
      pdb.gimp_image_set_active_layer(self.image, self.label_layer)
    else:
      pdb.gimp_image_set_active_layer(self.image, self.original_layer)
    #if self.label_layer.opacity == 0.0:
    #  if self.last_label_layer_opacity == 0.0:
    #    self.label_layer_opacity = 100.0
    #  else:
    #    self.label_layer.opacity = self.last_label_layer_opacity
    #else:
    #  self.last_label_layer_opacity = self.label_layer.opacity
    #  self.label_layer.visible = False
    #  self.label_layer.opacity = 0.0
    self.label_layer.flush()
    pdb.gimp_displays_flush()

  def normalBlendButtonClicked(self, widget):
    self.label_layer.mode = gimpenums.NORMAL_MODE
    self.label_layer.flush()
    pdb.gimp_displays_flush()

  def grainBlendButtonClicked(self, widget):
    self.label_layer.mode = gimpenums.GRAIN_MERGE_MODE
    self.label_layer.flush()
    pdb.gimp_displays_flush()

  def colorBlendButtonClicked(self, widget):
    self.label_layer.mode = gimpenums.COLOR_MODE
    self.label_layer.flush()
    pdb.gimp_displays_flush()

  def labelSelectionButtonClicked(self, widget):
    pdb.gimp_selection_sharpen(self.image)
    pdb.gimp_edit_fill(self.label_layer, gimpenums.FOREGROUND_FILL)
    self.label_layer.flush()
    pdb.gimp_displays_flush()

  def clearSelectionButtonClicked(self, widget):
    pdb.gimp_selection_none(self.image)
    pdb.gimp_displays_flush()

  def hardenSelectionButtonClicked(self, widget):
    pdb.gimp_selection_sharpen(self.image)
    pdb.gimp_displays_flush()

  def smoothSelectionButtonClicked(self, widget):
    pdb.gimp_selection_grow(self.image, 3)
    pdb.gimp_selection_shrink(self.image, 3)
    pdb.gimp_selection_sharpen(self.image)
    pdb.gimp_displays_flush()

  def invertSelectionButtonClicked(self, widget):
    pdb.gimp_selection_invert(self.image)
    pdb.gimp_selection_sharpen(self.image)
    pdb.gimp_displays_flush()

  def growSelectionButtonClicked(self, widget):
    pdb.gimp_selection_grow(self.image, 1)
    pdb.gimp_selection_sharpen(self.image)
    pdb.gimp_displays_flush()

  def shrinkSelectionButtonClicked(self, widget):
    pdb.gimp_selection_shrink(self.image, 1)
    pdb.gimp_selection_sharpen(self.image)
    pdb.gimp_displays_flush()

  def slicColorSpaceButtonToggled(self, widget):
    widget.set_label('Using LAB' if widget.get_active() else 'Using RGB')

  def createSegmentationButtonClicked(self, widget):
    try:
      slic_n = max(2, min(self.num_of_labels, int(self.slic_n.get_text())))
    except ValueError:
      self.alertDialog('# of regions is not a valid integer.')
    try:
      slic_sigma = float(self.slic_sigma.get_text())
      if slic_sigma < 0.0:
        raise ValueError()
    except ValueError:
      self.alertDialog('Smooth sigma is not a valid, non-negative floating point number.')
    try:
      slic_compactness = float(self.slic_compactness.get_text())
      if slic_compactness <= 0.0:
        raise ValueError()
    except ValueError:
      self.alertDialog('Compactness is not a valid, positive, non-zero floating point number.')
    slic_lab = self.slic_lab.get_active()
    pixel_region = self.original_layer.get_pixel_rgn(0, 0,
                                                     self.image.width, self.image.height,
                                                     False, False)
    byte_array = array.array('B', pixel_region[0:self.image.width, 0:self.image.height])
    byte_array = np.array(byte_array, dtype='uint8')
    byte_array = byte_array.reshape(len(byte_array)/4, 4)
    rgba_original_image = byte_array.reshape(self.image.height, self.image.width, 4)
    rgb_original_image = rgba_original_image[:, :, :3]
    segments_int_image = slic(rgb_original_image,
                              n_segments=slic_n,
                              compactness=slic_compactness,
                              sigma=slic_sigma,
                              convert2lab=slic_lab)
    # turn into rgb image
    segments_rgb_image = self.colormap[self.shufflemap[segments_int_image]]
    # create a new label layer
    new_layer = pdb.gimp_layer_new(self.image, self.image.width, self.image.height, gimpenums.RGBA_IMAGE, 'Superpixel Helper', 100, gimpenums.NORMAL_MODE)
    pdb.gimp_image_insert_layer(self.image, new_layer, None, 0)
    pixel_region = new_layer.get_pixel_rgn(0, 0,
                                           self.image.width, self.image.height,
                                           True, True)
    segments_rgba_image = np.dstack((segments_rgb_image,
                                     np.ones((self.image.height, self.image.width),
                                             dtype='uint8') * 255))
    pixel_region[0:self.image.width, 0:self.image.height] = array.array('B', segments_rgba_image.ravel()).tostring()
    new_layer.merge_shadow(True)
    new_layer.update(0, 0, self.image.width, self.image.height)
    new_layer.flush()
    pdb.gimp_displays_flush()

  def clearAllSegmentationsButtonClicked(self, widget):
    segmentation_layer = None
    while True:
      for layer in self.image.layers:
        if layer.name.startswith('Superpixel Helper'):
          segmentation_layer = layer
          break
      if segmentation_layer == None:
        break
      else:
        pdb.gimp_image_remove_layer(self.image, segmentation_layer)
        segmentation_layer = None
    pdb.gimp_displays_flush()

  def commentChanged(self, widget):
      comment_filename = os.path.join(self.working_path, 'comment-txt', self.image_name+'.txt')
      with open(comment_filename, 'w') as f:
        f.write(widget.get_text())

  def update(self, *args):
    self.updateInterface()
    gobject.timeout_add(200, self.update, self)

def toolbox_main(image, drawable):
  window = LabelToolbox(image)
  if not imports_succeeded:
    window.alertDialog('This plugin requires NumPy and SciPy. For segmentation, Scikit-Image is required.')
    window.destroy()
  else:
    gtk.main()

gimpfu.register(
     'label_toolbox',
     'Toolbox for labeling images.',
     'Toolbox for labeling images.',
     'Viet Nguyen',
     'Viet Nguyen',
     '2014',
     '<Image>/Toolbox/Labeling',
     '*',
     [],
     [],
     toolbox_main)

gimpfu.main()
