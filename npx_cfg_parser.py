from pathlib import *
import math
import copy
import os
import re
import ast
import json
from enum import Enum
from typing import NamedTuple

from npx_neuron_type import *


class NpxCfgSection(dict):
  def __init__(self, name: str = None, node_name: str = None):
    self.name = name
    self.node_name = node_name



# an app file is <app name>.snn.jsonc, an SNN definition (preprocess, train, network), not just any JSONC file
APP_CFG_SUFFIX = '.snn.jsonc'
# the same content written by code (no comments) is plain JSON
APP_JSON_SUFFIX = '.snn.json'


def app_stem(path: Path) -> str:
  """the app name of an app file: mnist_lenet5.snn.jsonc or .snn.json -> mnist_lenet5 (any other file: its stem)"""
  path = Path(path)
  for suffix in (APP_CFG_SUFFIX, APP_JSON_SUFFIX):
    if path.name.endswith(suffix):
      return path.name[:-len(suffix)]
  return path.stem


def is_app_json(path: Path) -> bool:
  return Path(path).suffix in ('.jsonc', '.json')
JSON_TOP_KEYS = ('preprocess', 'train', 'global', 'nodes')
# "name" and "type" belong to the node, so a section's own option called type (Shortcut) is "op" in JSON
JSON_RENAMED_KEYS = {'type': 'op'}
JSON_RENAMED_KEYS_BACK = {value: key for key, value in JSON_RENAMED_KEYS.items()}


def strip_jsonc(text: str) -> str:
  """JSON with comments: drop // and /* */ comments and trailing commas, keep strings as they are."""
  result = []
  i = 0
  n = len(text)
  while i < n:
    c = text[i]
    if c == '"':
      j = i + 1
      while j < n and text[j] != '"':
        j += 2 if text[j] == '\\' else 1
      result.append(text[i:j+1])
      i = j + 1
    elif text.startswith('//', i):
      while i < n and text[i] != '\n':
        i += 1
    elif text.startswith('/*', i):
      j = text.find('*/', i + 2)
      if j < 0:
        raise json.JSONDecodeError('unterminated /* comment', text, i)
      result.append('\n' * text.count('\n', i, j))  # keep line numbers for error messages
      i = j + 2
    elif c == ',':
      j = i + 1
      while j < n and text[j] in ' \t\r\n':
        j += 1
      if j < n and text[j] in '}]':
        i += 1  # trailing comma
      else:
        result.append(c)
        i += 1
    else:
      result.append(c)
      i += 1
  return ''.join(result)


def json_to_python(value):
  # a cfg value like 28,28 was a tuple, and the code relies on that (e.g. (c,) + input_size)
  if isinstance(value, list):
    return tuple(json_to_python(each) for each in value)
  if isinstance(value, dict):
    return {key: json_to_python(each) for key, each in value.items()}
  return value


def python_to_json(value):
  if isinstance(value, (tuple, list)):
    return [python_to_json(each) for each in value]
  if isinstance(value, dict):
    return {key: python_to_json(each) for key, each in value.items()}
  if isinstance(value, Enum):
    return value.name
  if hasattr(value, 'item') and callable(value.item):  # numpy/torch scalar
    return value.item()
  return value


def default_node_name(section_name: str, index: int, section_list) -> str:
  count = sum(1 for each in section_list[:index+1] if each.name == section_name)
  return f'{section_name.lower()}_{count}'


def format_json(value, indent: int = 0) -> str:
  """json.dumps with indent, but arrays of scalars and small objects stay on one line"""
  pad = '  ' * indent
  if isinstance(value, dict):
    if not value:
      return '{}'
    item_list = [f'{pad}  {json.dumps(key)}: {format_json(each, indent+1)}' for key, each in value.items()]
    return '{\n' + ',\n'.join(item_list) + f'\n{pad}}}'
  if isinstance(value, list):
    if all(not isinstance(each, (dict, list)) for each in value):
      return '[' + ', '.join(json.dumps(each) for each in value) + ']'
    item_list = [f'{pad}  {format_json(each, indent+1)}' for each in value]
    return '[\n' + ',\n'.join(item_list) + f'\n{pad}]'
  return json.dumps(value)


class SignedType(Enum):
  ERROR = -1
  UNSIGNED = 0
  SIGNED = 1

  def __mul__(self, other):
    assert (self != SignedType.ERROR)
    assert (other != SignedType.ERROR)
    if self == SignedType.SIGNED or other == SignedType.SIGNED:
      result = SignedType.SIGNED
    else:
      result = SignedType.UNSIGNED
    return result

  @property
  def is_signed(self):
    return self.value


class NumberType(Enum):
  ERROR = -1
  CONTI = 0
  DISCR = 1

  def __mul__(self, other):
    assert (self != NumberType.ERROR)
    assert (other != NumberType.ERROR)
    if self == NumberType.CONTI or other == NumberType.CONTI:
      result = NumberType.CONTI
    else:
      result = NumberType.DISCR
    return result

  @property
  def is_quantized(self):
    return self.value


class DataType(NamedTuple):
  signedtype: SignedType
  numbertype: NumberType
  is_binary: bool
  maxvalue: int

  def __str__(self):
    return str((self.name, self.numbertype.name, self.is_binary, self.maxvalue))

  def __mul__(self, other):
    if type(other) == DataType:
      other_datatype = other
    elif type(other) == int:
      is_binary = (other == 0) or (other == 1) or (other == (-1))
      other_datatype = DataType(
        SignedType.SIGNED, NumberType.DISCR, is_binary, other)
    elif type(other) == float:
      is_binary = (other == 0) or (other == 1) or (other == (-1))
      other_datatype = DataType(
        SignedType.SIGNED, NumberType.CONTI, is_binary, other)
    else:
      assert 0
    signedtype = self.signedtype*other_datatype.signedtype
    numbertype = self.numbertype*other_datatype.numbertype
    is_binary = self.is_binary and other_datatype.is_binary
    maxvalue = self.maxvalue*other_datatype.maxvalue
    return DataType(signedtype, numbertype, is_binary, maxvalue)

  @property
  def bitwidth(self):
    return max(1, math.ceil(math.log2(self.maxvalue)))

  @staticmethod
  def convert_neuron_type(neuron_type: NpxNeuronType):
    signedtype = SignedType.SIGNED if neuron_type.is_signed_weight else SignedType.UNSIGNED
    numbertype = NumberType.DISCR if neuron_type.is_quantized else NumberType.CONTI
    return DataType(signedtype, numbertype, 0, neuron_type.qcode_max)


class LayerIoInfo(NamedTuple):
  scale: int
  datatype: DataType
  channels: int
  size: tuple

  def __str__(self):
    return str((self.scale, self.datatype.signedtype.name, self.datatype.numbertype.name, self.datatype.is_binary, self.datatype.maxvalue, self.datatype.bitwidth, self.channels, self.size))

  @property
  def dims(self):
    size = len(self.size)
    for each_size in self.size:
      if each_size == 1:
        size -= 1
    return size


RISCV_GLOBAL_KEYS = ('input_size', 'input_channels', 'output_classes')
RISCV_IO_KEYS = ('in_channels', 'out_channels', 'in_size', 'out_size',
                 'in_is_quantized', 'out_is_quantized', 'in_is_binary', 'out_is_binary', 'out_maxvalue')
RISCV_WEIGHT_KEYS = ('neuron_type',)
RISCV_CONV_KEYS = ('kernel_size', 'stride', 'padding', 'groups') + RISCV_WEIGHT_KEYS
RISCV_POOL_KEYS = ('kernel_size', 'stride', 'padding')
RISCV_NEURON_KEYS = ('neuron_type', 'reset_mechanism', 'reset_delay')
RISCV_LAYER_KEYS = {
  'Conv2d': RISCV_CONV_KEYS,
  'Linear': ('in_features', 'out_features') + RISCV_WEIGHT_KEYS,
  'MaxPool2d': RISCV_POOL_KEYS,
  'AvgPool2d': RISCV_POOL_KEYS,
  'Shortcut': ('skip_from', 'type') + RISCV_CONV_KEYS,
  'Flatten': (),
  'Leaky': RISCV_NEURON_KEYS,
  'Synaptic': RISCV_NEURON_KEYS,
  'Alpha': RISCV_NEURON_KEYS,
}

def keep_only(section, key_list):
  for key in [key for key in section if key not in key_list]:
    del section[key]


class NpxCfgParser():
  def __init__(self, path: Path = None):
    super().__init__()
    self.train_info = None
    self.global_info = None
    self.preprocess_info = None
    self.layer_info_list = []
    if path and path.exists():
      assert path.is_file(), path
      self.parse_file(path)

  def parse_file(self, path: Path):
    assert path.is_file(), path
    if path.suffix == '.cfg':
      raise ValueError(f'{path}: the .cfg format is no longer read, convert it first: python3 npx_cfg_convert.py {path}')
    self.parse_json_text(path.read_text(), str(path))

  def parse_json_text(self, text: str, source: str = '<string>'):
    try:
      root = json.loads(strip_jsonc(text))
    except json.JSONDecodeError as error:
      raise ValueError(f'{source}: {error}') from None
    if not isinstance(root, dict):
      raise ValueError(f'{source}: the top level must be an object')
    unknown_keys = [key for key in root if key not in JSON_TOP_KEYS]
    if unknown_keys:
      raise ValueError(f'{source}: unknown top-level keys {unknown_keys}, expected {list(JSON_TOP_KEYS)}')
    for key in ('preprocess', 'train', 'global'):
      if key in root:
        section = NpxCfgSection(key)
        section.update(json_to_python(root[key]))
        setattr(self, f'{key}_info', section)

    node_list = root.get('nodes', [])
    if not isinstance(node_list, list):
      raise ValueError(f'{source}: "nodes" must be an array (the order of the nodes is the order of the layers)')
    name_to_index = {}
    for index, node in enumerate(node_list):
      if not isinstance(node, dict) or 'type' not in node:
        raise ValueError(f'{source}: node {index} must be an object with a "type"')
      node = json_to_python(node)
      node_name = node.pop('name', None) or f'layer{index}'
      if node_name in name_to_index:
        raise ValueError(f'{source}: node name {node_name!r} is used twice')
      name_to_index[node_name] = index
      section = NpxCfgSection(node.pop('type'), node_name)
      for key, value in node.items():
        section[JSON_RENAMED_KEYS_BACK.get(key, key)] = value
      self.layer_info_list.append(section)

    # a shortcut names its source node; inside, skip_from stays the relative index the cfg used
    for index, section in enumerate(self.layer_info_list):
      skip_from = section.get('skip_from')
      if isinstance(skip_from, str):
        if skip_from not in name_to_index:
          raise ValueError(f'{source}: node {section.node_name!r} skips from unknown node {skip_from!r}')
        if name_to_index[skip_from] >= index:
          raise ValueError(f'{source}: node {section.node_name!r} must skip from an earlier node, not {skip_from!r}')
        section['skip_from'] = name_to_index[skip_from] - index

  def node_name_list(self):
    result = []
    for index, section in enumerate(self.layer_info_list):
      result.append(section.node_name or default_node_name(section.name, index, self.layer_info_list))
    return result

  def to_json_dict(self) -> dict:
    root = {}
    for key in ('preprocess', 'train', 'global'):
      section = getattr(self, f'{key}_info')
      if section is not None:
        root[key] = python_to_json(dict(section))
    node_name_list = self.node_name_list()
    node_list = []
    for index, section in enumerate(self.layer_info_list):
      node = {'name': node_name_list[index], 'type': section.name}
      for key, value in section.items():
        if key == 'skip_from' and isinstance(value, int):
          value = node_name_list[index + value]
        node[JSON_RENAMED_KEYS.get(key, key)] = python_to_json(value)
      node_list.append(node)
    root['nodes'] = node_list
    return root

  def to_json_text(self) -> str:
    return format_json(self.to_json_dict()) + '\n'

  def __str__(self) -> str:
    return self.to_json_text()

  def __repr__(self) -> str:
    return str(self)

  def write_file(self, path: Path):
    path.write_text(self.to_json_text())

  def add_train_info(self):
    assert self.train_info
    self.train_info = NpxCfgSection('train')
    return self.train_info

  def add_preprocess_info(self):
    assert self.preprocess_info
    self.preprocess_info = NpxCfgSection('preprocess')
    return self.preprocess_info

  def add_network(self, section_name: str):
    self.layer_info_list.append(NpxCfgSection(section_name))
    return self.layer_info_list[-1]

  def generate_preprocess_output_info(self):
    assert self.preprocess_info
    assert self.train_info
    assert self.global_info
    if self.preprocess_info['input']=='mnist_opendataset' or self.preprocess_info['input']=='kmnist_opendataset' or self.preprocess_info['input']=='fmnist_opendataset' or self.preprocess_info['input']=='cifar10_opendataset' or self.preprocess_info['input']=='gtsrb_opendataset':
      if self.preprocess_info['step_generation'] == 'direct':
        scale = 65535
        datatype = DataType(SignedType.UNSIGNED,
                  NumberType.DISCR, 0, scale)
      else:
        assert 0
    elif self.preprocess_info['input'] == 'dvsgesture_opendataset':
      scale = 65535
      datatype = DataType(SignedType.UNSIGNED, NumberType.DISCR, 0, scale)
    elif self.preprocess_info['input'].endswith('dataset'):
      if self.preprocess_info['step_generation'] == 'direct':
        scale = 65535
        datatype = DataType(SignedType.UNSIGNED,
                  NumberType.DISCR, 0, scale)
      else:
        assert 0
    else:
      assert 0
    return LayerIoInfo(scale, datatype, self.global_info['input_channels'], self.global_info['input_size'])

  def elaborate_for_riscv(self):
    output_info = self.generate_preprocess_output_info()
    output_info_list = []
    for layer_info in self.layer_info_list:
      layer_info['input_info'] = output_info
      neuron_type = NpxNeuronType(layer_info['neuron_type'])
      weight_datatype = DataType.convert_neuron_type(neuron_type)
      output_scale = layer_info['input_info'].scale
      output_datatype = layer_info['input_info'].datatype
      if layer_info.name == 'Conv2d':
        in_channels = layer_info['in_channels']
        assert layer_info['input_info'].dims == 2, layer_info['input_info'].size
        assert layer_info['input_info'].channels == in_channels, in_channels
        output_datatype *= (layer_info['kernel_size']
                  * layer_info['kernel_size']
                  * (in_channels // layer_info.get('groups', 1)))
        output_datatype *= weight_datatype
        output_size = []
        for each_size in layer_info['input_info'].size:
          new_size = int((each_size + (2*layer_info['padding']) - (
            layer_info['kernel_size']-1)) / layer_info['stride'])
          output_size.append(new_size)
        out_channels = layer_info['out_channels']

        layer_info['weight_bitwidth'] = neuron_type.num_bits

      elif layer_info.name == 'Shortcut':
        layer_info['input_info'] = output_info_list[len(output_info_list) + layer_info['skip_from']]

        if layer_info['type']=='Conv2d':
          in_channels = layer_info['in_channels']
          assert layer_info['input_info'].dims == 2, layer_info['input_info'].size
          assert layer_info['input_info'].channels == in_channels, in_channels
          output_datatype *= (layer_info['kernel_size']
                    * layer_info['kernel_size']
                    * in_channels)
          output_datatype *= weight_datatype
          output_size = []
          for each_size in layer_info['input_info'].size:
            new_size = int((each_size + (2*layer_info['padding']) - (
              layer_info['kernel_size']-1)) / layer_info['stride'])
            output_size.append(new_size)
          out_channels = layer_info['out_channels']

          layer_info['weight_bitwidth'] = neuron_type.num_bits

        elif layer_info['type']=='Identity':
          output_scale = 1
          output_datatype = layer_info['input_info'].datatype
          out_channels = layer_info['input_info'].channels
          output_size = layer_info['input_info'].size

        elif layer_info['type'] in ('MaxPool2d', 'AvgPool2d'):
          assert layer_info['input_info'].dims == 2, layer_info['input_info'].size
          output_scale = 1
          output_datatype = layer_info['input_info'].datatype
          out_channels = layer_info['input_info'].channels
          output_size = []
          for each_size in layer_info['input_info'].size:
            output_size.append(int(each_size/layer_info['stride']))

        else:
          assert 0, layer_info['type']

      elif layer_info.name == 'Linear':
        assert layer_info['input_info'].dims == 1, layer_info['input_info'].size
        assert layer_info['in_features'] == layer_info['input_info'].size[0]
        output_datatype *= layer_info['in_features']
        output_datatype *= weight_datatype
        out_channels = layer_info['input_info'].channels
        output_size = (layer_info['out_features'], 1) 
        layer_info['weight_bitwidth'] = neuron_type.num_bits

      elif layer_info.name == 'AvgPool2d' or layer_info.name == 'MaxPool2d':
        assert layer_info['input_info'].dims == 2, layer_info['input_info'].size
        out_channels = layer_info['input_info'].channels
        output_size = []
        for each_size in layer_info['input_info'].size:
          new_size = int(each_size/layer_info['stride'])
          output_size.append(new_size)
      elif layer_info.name in ('Leaky', 'Synaptic', 'Alpha'):
        output_scale = 1
        output_datatype = DataType(
          SignedType.UNSIGNED, NumberType.DISCR, 1, 1)
        out_channels = layer_info['input_info'].channels
        output_size = layer_info['input_info'].size

      elif layer_info.name == 'Flatten':
        out_channels = 1
        output_size = layer_info['input_info'].channels
        for each_size in layer_info['input_info'].size:
          output_size *= each_size
        output_size = (output_size, 1)
      else:
        assert 0, layer_info.name
      output_info = LayerIoInfo(
        output_scale, output_datatype, out_channels, tuple(output_size))
      layer_info['output_info'] = output_info
      output_info_list.append(output_info)

    for layer_info in self.layer_info_list:
      for io_tpye in ('in', 'out'):
        layer_info[f'{io_tpye}_is_signed'] = layer_info[f'{io_tpye}put_info'].datatype.signedtype.is_signed
        layer_info[f'{io_tpye}_is_quantized'] = layer_info[f'{io_tpye}put_info'].datatype.numbertype.is_quantized
        layer_info[f'{io_tpye}_is_binary'] = int(layer_info[f'{io_tpye}put_info'].datatype.is_binary)
        layer_info[f'{io_tpye}_maxvalue'] = layer_info[f'{io_tpye}put_info'].datatype.maxvalue
        layer_info[f'{io_tpye}_channels'] = layer_info[f'{io_tpye}put_info'].channels
        layer_info[f'{io_tpye}_dims'] = layer_info[f'{io_tpye}put_info'].dims
        layer_info[f'{io_tpye}_size'] = layer_info[f'{io_tpye}put_info'].size

      print(layer_info.name)
      print(layer_info['input_info'])
      print(layer_info['output_info'])

      if (layer_info.name in ('Leaky', 'Synaptic', 'Alpha')) or ('weight_bitwidth' in layer_info):
        layer_info['neuron_type'] = NpxNeuronType(layer_info['neuron_type']).full_name
      else:
        layer_info.pop('neuron_type', None)
      keep_only(layer_info, RISCV_IO_KEYS + RISCV_LAYER_KEYS[layer_info.name])

    keep_only(self.global_info, RISCV_GLOBAL_KEYS)
    self.preprocess_info = None
    self.train_info = None
