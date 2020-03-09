# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tools to analyse and plot TFL models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import tempfile

from . import model_info
import matplotlib.pyplot as plt
# Needed for pyplot 3d projections.
from mpl_toolkits.mplot3d import Axes3D as _  # pylint: disable=unused-import
import numpy as np


def draw_model_graph(model_graph, calibrator_dpi=30):
  """Draws the model graph.

  This function requires IPython and graphviz packages.

  ```
  model_graph = estimators.get_model_graph(saved_model_path)
  visualization.draw_model_graph(model_graph)
  ```

  Args:
    model_graph: a `model_info.ModelInfo` objects to plot.
    calibrator_dpi: The DPI for calibrator plots inside the graph nodes.
  """
  import graphviz  # pylint: disable=g-import-not-at-top
  import IPython.display  # pylint: disable=g-import-not-at-top

  dot = graphviz.Digraph(format='png', engine='dot')
  dot.graph_attr['ranksep'] = '0.75'

  # Check if we need split nodes for shared calibration
  model_has_shared_calibration = False
  for node in model_graph.nodes:
    model_has_shared_calibration |= (
        (isinstance(node, model_info.PWLCalibrationNode) or
         isinstance(node, model_info.CategoricalCalibrationNode)) and
        (len(_output_nodes(model_graph, node)) > 1))

  split_nodes = {}
  for node in model_graph.nodes:
    node_id = _node_id(node)
    if (isinstance(node, model_info.PWLCalibrationNode) or
        isinstance(node, model_info.CategoricalCalibrationNode)):
      # Add node for calibrator with calibrator plot inside.
      fig = plot_calibrator_nodes([node])
      filename = os.path.join(tempfile.tempdir, 'i{}.png'.format(node_id))
      plt.savefig(filename, dpi=calibrator_dpi)
      plt.close(fig)
      dot.node(node_id, '', image=filename, shape='box')

      # Add input feature node.
      node_is_feature_calibration = isinstance(node.input_node,
                                               model_info.InputFeatureNode)
      if node_is_feature_calibration:
        input_node_id = node_id + 'input'
        dot.node(input_node_id, node.input_node.name)
        dot.edge(input_node_id + ':s', node_id + ':n')

        # Add split node for shared calibration.
        if model_has_shared_calibration:
          split_node_id = node_id + 'calibrated'
          split_node_name = 'calibrated {}'.format(node.input_node.name)
          dot.node(split_node_id, split_node_name)
          dot.edge(node_id + ':s', split_node_id + ':n')
          split_nodes[node_id] = (split_node_id, split_node_name)

    elif not isinstance(node, model_info.InputFeatureNode):
      dot.node(node_id, _node_name(node), shape='box', margin='0.3')

    if node is model_graph.output_node:
      output_node_id = node_id + 'output'
      dot.node(output_node_id, 'output')
      dot.edge(node_id + ':s', output_node_id)

  for node in model_graph.nodes:
    node_id = _node_id(node)
    for input_node in _input_nodes(node):
      if isinstance(input_node, model_info.InputFeatureNode):
        continue
      input_node_id = _node_id(input_node)
      if input_node_id in split_nodes:
        split_node_id, split_node_name = split_nodes[input_node_id]
        input_node_id = split_node_id + node_id
        dot.node(input_node_id, split_node_name)

      dot.edge(input_node_id + ':s', node_id)  # + ':n')

  filename = os.path.join(tempfile.tempdir, 'dot')
  try:
    dot.render(filename)
    IPython.display.display(IPython.display.Image('{}.png'.format(filename)))
  except graphviz.backend.ExecutableNotFound as e:
    if 'IPython.core.magics.namespace' in sys.modules:
      # Similar to Keras visualization lib, we don't raise an exception here to
      # avoid crashing notebooks during tests.
      print(
          'dot binaries were not found or not in PATH. The system running the '
          'colab binary might not have graphviz package installed: format({})'
          .format(e))
    else:
      raise e


def plot_calibrator_nodes(nodes,
                          plot_submodel_calibration=True,
                          font_size=12,
                          axis_label_font_size=14,
                          figsize=None):
  """Plots feature calibrator(s) extracted from a TFL canned estimator.

  Args:
    nodes: List of calibrator nodes to be plotted.
    plot_submodel_calibration: If submodel calibrators should be included in the
      output plot, when more than one calibration node is provided. These are
      individual calibration layers for each lattice in a lattice ensemble
      constructed from `configs.CalibratedLatticeEnsembleConfig`.
    font_size: Font size for values and labels on the plot.
    axis_label_font_size: Font size for axis labels.
    figsize: The figsize parameter passed to `pyplot.figure()`.

  Returns:
    Pyplot figure object containing the visualisation.
  """

  with plt.style.context('seaborn-whitegrid'):
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('axes', labelsize=axis_label_font_size)
    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(1, 1, 1)
    if isinstance(nodes[0], model_info.PWLCalibrationNode):
      _plot_pwl_calibrator(nodes, axes, plot_submodel_calibration)
    elif isinstance(nodes[0], model_info.CategoricalCalibrationNode):
      _plot_categorical_calibrator(nodes, axes, plot_submodel_calibration)
    else:
      raise ValueError('Unknown calibrator type: {}'.format(nodes[0]))
    plt.tight_layout()

  return fig


def plot_feature_calibrator(model_graph,
                            feature_name,
                            plot_submodel_calibration=True,
                            font_size=12,
                            axis_label_font_size=14,
                            figsize=None):
  """Plots feature calibrator(s) extracted from a TFL canned estimator.

  ```
  model_graph = estimators.get_model_graph(saved_model_path)
  visualization.plot_feature_calibrator(model_graph, "feature_name")
  ```

  Args:
    model_graph: `model_info.ModelGraph` object that includes model nodes.
    feature_name: Name of the feature to plot the calibrator for.
    plot_submodel_calibration: If submodel calibrators should be included in the
      output plot, when more than one calibration node is provided. These are
      individual calibration layers for each lattice in a lattice ensemble
      constructed from `configs.CalibratedLatticeEnsembleConfig`.
    font_size: Font size for values and labels on the plot.
    axis_label_font_size: Font size for axis labels.
    figsize: The figsize parameter passed to `pyplot.figure()`.

  Returns:
    Pyplot figure object containing the visualisation.
  """

  input_feature_node = [
      input_feature_node
      for input_feature_node in _input_feature_nodes(model_graph)
      if input_feature_node.name == feature_name
  ]
  if not input_feature_node:
    raise ValueError(
        'Feature "{}" not found in the model_graph.'.format(feature_name))

  input_feature_node = input_feature_node[0]
  calibrator_nodes = _output_nodes(model_graph, input_feature_node)
  return plot_calibrator_nodes(calibrator_nodes, plot_submodel_calibration,
                               font_size, axis_label_font_size, figsize)


def plot_all_calibrators(model_graph, num_cols=4, **kwargs):
  """Plots all feature calibrator(s) extracted from a TFL canned estimator.

  The generated plots are arranged in a grid.
  This function requires IPython and colabtools packages.

  ```
  model_graph = estimators.get_model_graph(saved_model_path)
  visualization.plot_all_calibrators(model_graph)
  ```

  Args:
    model_graph: a `model_info.ModelGraph` objects to plot.
    num_cols: Number of columns in the grid view.
    **kwargs: args passed to `analysis.plot_calibrators`.
  """
  import google.colab.widgets  # pylint: disable=g-import-not-at-top
  import IPython.display  # pylint: disable=g-import-not-at-top

  feature_infos = _input_feature_nodes(model_graph)
  feature_names = sorted([feature_info.name for feature_info in feature_infos])

  output_calibrator_node = (
      model_graph.output_node if isinstance(
          model_graph.output_node, model_info.PWLCalibrationNode) else None)

  num_feature_calibrators = len(feature_names)
  num_output_calibrators = 1 if output_calibrator_node else 0

  # Calibrator plots are organized in a grid. We first plot all the feature
  # calibrators, followed by any existing output calibrator.
  num_rows = int(
      math.ceil(
          float(num_feature_calibrators + num_output_calibrators) / num_cols))
  for index, _ in enumerate(
      google.colab.widgets.Grid(
          num_rows, num_cols, style='border-top: 0; border-bottom: 0;')):
    if index >= num_feature_calibrators + num_output_calibrators:
      continue  # Empty cells

    if index < num_feature_calibrators:
      feature_name = feature_names[index]
      tb = google.colab.widgets.TabBar(
          ['Calibrator for "{}"'.format(feature_name), 'Large Plot'])
    else:
      feature_name = 'output'
      tb = google.colab.widgets.TabBar(['Output calibration', 'Large Plot'])

    with tb.output_to(0, select=True):
      if index < len(feature_names):
        plot_feature_calibrator(model_graph, feature_name, **kwargs)
      else:
        plot_calibrator_nodes([output_calibrator_node])
      filename = os.path.join(tempfile.tempdir, '{}.png'.format(feature_name))
      # Save a larger temporary copy to be shown in a second tab.
      plt.savefig(filename, dpi=200)
      plt.show()
    with tb.output_to(1, select=False):
      IPython.display.display(IPython.display.Image(filename))


def _input_feature_nodes(model_graph):
  return [
      node for node in model_graph.nodes
      if isinstance(node, model_info.InputFeatureNode)
  ]


def _node_id(node):
  return str(id(node))


def _node_name(node):
  if isinstance(node, model_info.LinearNode):
    return 'Linear'
  if isinstance(node, model_info.LatticeNode):
    return 'Lattice'
  if isinstance(node, model_info.MeanNode):
    return 'Average'
  return str(type(node))


def _contains(nodes, node):
  return any(other_node is node for other_node in nodes)


def _input_nodes(node):
  if hasattr(node, 'input_nodes'):
    return node.input_nodes
  if hasattr(node, 'input_node'):
    return [node.input_node]
  return []


def _output_nodes(model_graph, node):
  return [
      other_node for other_node in model_graph.nodes
      if _contains(_input_nodes(other_node), node)
  ]


_MISSING_NAME = 'missing'
_CALIBRATOR_COLOR = 'tab:blue'
_MISSING_COLOR = 'tab:orange'


def _plot_categorical_calibrator(categorical_calibrator_nodes, axes,
                                 plot_submodel_calibration):
  """Plots a categorical calibrator.


  Creates a categorical calibraiton plot combining the passed in calibration
  nodes. You can select to also show individual calibrator nodes in the plot.

  Args:
    categorical_calibrator_nodes: a list of
      `model_info.CategoricalCalibrationNode` objects in a model graph. If more
      that one node is provided, they must be for the same input feature.
    axes: Pyplot axes object.
    plot_submodel_calibration: If submodel calibrators should be included in the
      output plot, when more than one calibration node is provided. These are
      individual calibration layers for each lattice in a lattice ensemble
      constructed from `configs.CalibratedLatticeEnsembleConfig`.
  """
  feature_info = categorical_calibrator_nodes[0].input_node
  assert feature_info.is_categorical

  # Adding missing category to input values.
  # Note that there might be more than one out-of-vocabulary value
  # (i.e. (num_oov_buckets + (default_value is not none)) > 1), in which case
  # we name all of them missing.
  input_values = list(feature_info.vocabulary_list)
  while len(input_values) < len(categorical_calibrator_nodes[0].output_values):
    input_values.append(_MISSING_NAME)

  submodels_output_values = [
      node.output_values for node in categorical_calibrator_nodes
  ]
  mean_output_values = np.mean(submodels_output_values, axis=0)

  # Submodels categorical outputs are plotted in grouped form inside the
  # average calibration bar.
  bar_width = 0.8
  sub_width = bar_width / len(submodels_output_values)

  # Bar colors for each category.
  color = [
      _MISSING_COLOR if v == _MISSING_NAME else _CALIBRATOR_COLOR
      for v in input_values
  ]

  # Plot submodel calibrations fitting inside the average calibration bar.
  x = np.arange(len(input_values))
  if plot_submodel_calibration:
    for sub_index, output_values in enumerate(submodels_output_values):
      plt.bar(
          x - bar_width / 2 + sub_width / 2 + sub_index * sub_width,
          output_values,
          width=sub_width,
          alpha=0.1,
          color=color,
          linewidth=0.5)

  # Plot average category output.
  plt.bar(
      x,
      mean_output_values,
      color=color,
      linewidth=2,
      alpha=0.2,
      width=bar_width)
  plt.bar(
      x,
      mean_output_values,
      fill=False,
      edgecolor=color,
      linewidth=3,
      width=bar_width)

  # Set axes labels and tick values.
  plt.xlabel(feature_info.name)
  plt.ylabel('calibrated {}'.format(feature_info.name))
  axes.set_xticks(x)
  axes.set_xticklabels(input_values)
  axes.yaxis.grid(True, linewidth=0.25)
  axes.xaxis.grid(False)


def _plot_pwl_calibrator(pwl_calibrator_nodes, axes, plot_submodel_calibration):
  """Plots a PWL calibrator.

  Creates a pwl plot combining the passed in calibration nodes. You can select
  to also show individual calibrator nodes in the plot.

  Args:
    pwl_calibrator_nodes: a list of `model_info.PWLCalibrationNode` objects in a
      model graph. If more that one node is provided, they must be for the same
      input feature.
    axes: Pyplot axes object.
    plot_submodel_calibration: If submodel calibrators should be included in the
      output plot, when more than one calibration node is provided. These are
      individual calibration layers for each lattice in a lattice ensemble
      constructed from `configs.CalibratedLatticeEnsembleConfig`.
  """

  pwl_calibrator_node = pwl_calibrator_nodes[0]
  if isinstance(pwl_calibrator_node.input_node, model_info.InputFeatureNode):
    assert not pwl_calibrator_node.input_node.is_categorical
    input_name = pwl_calibrator_node.input_node.name
    output_name = 'calibrated {}'.format(input_name)
  else:
    # Output PWL calibration.
    input_name = 'input'
    output_name = 'output'

  # Average output_keypoints and (any) default_output across all the nodes.
  mean_output_keypoints = np.mean(
      [
          pwl_calibrator_node.output_keypoints
          for pwl_calibrator_node in pwl_calibrator_nodes
      ],
      axis=0,
  )
  if pwl_calibrator_node.default_output:
    mean_default_output = np.mean([
        pwl_calibrator_node.default_output
        for pwl_calibrator_node in pwl_calibrator_nodes
    ])
  else:
    mean_default_output = None

  if plot_submodel_calibration:
    for pwl_calibrator_node in pwl_calibrator_nodes:
      plt.plot(
          pwl_calibrator_node.input_keypoints,
          pwl_calibrator_node.output_keypoints,
          '--',
          linewidth=0.25,
          color=_CALIBRATOR_COLOR)
      if pwl_calibrator_node.default_output is not None:
        plt.plot(
            pwl_calibrator_node.input_keypoints,
            [pwl_calibrator_node.default_output] *
            len(pwl_calibrator_node.input_keypoints),
            '--',
            color=_MISSING_COLOR,
            linewidth=0.25)

  plt.plot(
      pwl_calibrator_node.input_keypoints,
      mean_output_keypoints,
      _CALIBRATOR_COLOR,
      linewidth=3,
      label='calibrated')
  if mean_default_output is not None:
    plt.plot(
        pwl_calibrator_node.input_keypoints,
        [mean_default_output] * len(pwl_calibrator_node.input_keypoints),
        color=_MISSING_COLOR,
        linewidth=3,
        label=_MISSING_NAME)

  plt.xlabel(input_name)
  plt.ylabel(output_name)
  axes.yaxis.grid(True, linewidth=0.25)
  axes.xaxis.grid(True, linewidth=0.25)
  axes.legend()


def plot_outputs(inputs, outputs_map, file_path=None, figsize=(20, 20)):
  """Visualises several outputs for same set of inputs.

  This is generic plotting helper not tied to any layer.
  Can visualize either:
    - 2-d graphs: 1-d input, 1-d output.
    - 3-d surfaces: 2-d input, 1-d output.

  Args:
    inputs: one of:
      - ordered list of 1-d points
      - tuple of exactly 2 elements which represent X and Y coordinates of 2-d
        mesh grid for pyplot 3-d surface visualization. See
        `test_utils.two_dim_mesh_grid` for more details.
    outputs_map: dictionary {name: outputs} where "outputs" is a list of 1-d
      points which correspond to "inputs". "name" is an arbitrary string used as
      legend.
    file_path: if set - visualisation will be saved as png at specified
      location.
    figsize: The figsize parameter passed to `pyplot.figure()`.

  Raises:
    ValueError: if configured to visualise more than 4 3-d plots.

  Returns:
    Pyplot object containing visualisation.
  """
  legend = []
  if isinstance(inputs, tuple):
    figure = plt.figure(figsize=figsize)
    axes = figure.gca(projection='3d')
    # 4 colors is enough because no one would ever think of drawing 5 or more
    # 3-d surfaces on same graph due to them looking like fabulous mess anyway.
    colors = ['dodgerblue', 'forestgreen', 'saddiebrown', 'lightsalmon']
    if len(outputs_map) > 4:
      raise ValueError('Cannot visualize more than 4 3-d plots.')

    x_inputs, y_inputs = inputs
    for i, (name, outputs) in enumerate(outputs_map.items()):
      legend.append(name)
      z_outputs = np.reshape(
          np.asarray(outputs), newshape=(len(x_inputs), len(x_inputs[0])))

      axes.plot_wireframe(x_inputs, y_inputs, z_outputs, color=colors[i])
  else:
    for name, outputs in sorted(outputs_map.items()):
      legend.append(name)
      plt.plot(inputs, outputs)

    plt.ylabel('y')
    plt.xlabel('x')

  plt.legend(legend)
  if file_path:
    plt.savefig(file_path)
  return plt
