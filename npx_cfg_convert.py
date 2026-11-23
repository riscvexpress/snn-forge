"""Convert app files between formats.

  python3 npx_cfg_convert.py app/*.cfg              # legacy .cfg -> .snn.jsonc, comments kept
  python3 npx_cfg_convert.py net.nir                # NIR -> net.snn.jsonc
  python3 npx_cfg_convert.py app/x.snn.jsonc        # .snn.jsonc -> x.nir (zero weights)
  python3 npx_cfg_convert.py net.nir -t snn -o app  # pick the output format and directory
  python3 npx_cfg_convert.py --list                 # the formats and what each can do

The input format comes from the file name; the output format is -t, or the input
format's default target. The output goes next to each input, or into -o <dir>.

Structure: every conversion goes through one in-memory form, AppData (the parsed
app plus, when there is one, its .snn.jsonc text with comments). A format is a
Format entry in FORMAT_LIST with a reader (file -> AppData) and/or a writer
(AppData -> file), so a format only needs to be converted to and from AppData,
never to every other format. To add one: write read_x(path) -> AppData and/or
write_x(data, out_path), then add a Format(...) to FORMAT_LIST.

Formats (sections below):
- snn         .snn.jsonc, the app format (npx_cfg_parser.py); the hub
- cfg         legacy .cfg, read only. This file holds the only .cfg reader:
              [preprocess], [train], [global] become objects of the same name;
              every other section becomes one entry of "nodes", in order, with a
              generated "name" (conv2d_1, leaky_2, ...) and its section name as
              "type"; a Shortcut's skip_from=-4 becomes the name of the node it
              points at and its type=Conv2d becomes "op": "Conv2d"; "# text"
              becomes "// text" and #bias=False becomes // "bias": false
- preprocess  <app>.preprocess.jsonc, a preprocess section only (from <app>_preprocess.cfg)
- nir         NIR graph (.nir, HDF5), the network only, zero weights
"""
import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from npx_cfg_parser import (NpxCfgParser, NpxCfgSection, APP_CFG_SUFFIX, JSON_RENAMED_KEYS,
                            default_node_name, format_json, python_to_json, json_to_python)


###############################################################################
## Format registry
###############################################################################

@dataclass
class AppData:
  """an app between a reader and a writer"""
  cfg_parser: NpxCfgParser
  text: Optional[str] = None   # .snn.jsonc text with comments, when the reader made one


@dataclass
class Format:
  name: str
  suffix_list: tuple                 # the first one is what the writer writes
  description: str
  read: Optional[Callable] = None    # read(path: Path) -> AppData
  write: Optional[Callable] = None   # write(data: AppData, out_path: Path)
  default_target: Optional[str] = None

  def matched_suffix(self, path: Path) -> Optional[str]:
    return max((each for each in self.suffix_list if path.name.endswith(each)), key=len, default=None)

  def stem(self, path: Path) -> str:
    return path.name[:-len(self.matched_suffix(path))]


FORMAT_LIST = []   # filled at the end of this file, once every reader and writer exists


def get_format(name: str) -> Format:
  for each in FORMAT_LIST:
    if each.name == name:
      return each
  raise SystemExit(f'unknown format {name!r} (known: {", ".join(each.name for each in FORMAT_LIST)})')


def find_format(path: Path) -> Format:
  """the readable format whose suffix matches the file name best (.preprocess.jsonc before .jsonc)"""
  matched = [(len(each.matched_suffix(path)), each) for each in FORMAT_LIST if each.read and each.matched_suffix(path)]
  if not matched:
    suffixes = ', '.join(suffix for each in FORMAT_LIST if each.read for suffix in each.suffix_list)
    raise SystemExit(f'{path}: unknown input format (readable: {suffixes})')
  return max(matched, key=lambda item: item[0])[1]


def convert_file(in_path: Path, target_name: Optional[str] = None, out_dir: Optional[Path] = None,
                 force: bool = False) -> Optional[Path]:
  source = find_format(in_path)
  target = get_format(target_name or source.default_target or '')
  if not target.write:
    raise SystemExit(f'{target.name}: this format is read only')
  out_dir = out_dir or in_path.parent
  out_path = out_dir / (source.stem(in_path) + target.suffix_list[0])
  if out_path.resolve() == in_path.resolve():
    raise SystemExit(f'{in_path}: the output would overwrite the input')
  if out_path.exists() and not force:
    print(f'[skip] {out_path} exists (use -f to overwrite)')
    return None
  data = source.read(in_path)
  out_dir.mkdir(parents=True, exist_ok=True)
  target.write(data, out_path)
  print(f'{in_path} -> {out_path}')
  return out_path


###############################################################################
## snn: .snn.jsonc, the hub
###############################################################################

def read_snn(path: Path) -> AppData:
  return AppData(NpxCfgParser(path), path.read_text())


def write_snn(data: AppData, out_path: Path):
  out_path.write_text(data.text if data.text is not None else data.cfg_parser.to_json_text())


def write_preprocess(data: AppData, out_path: Path):
  if data.cfg_parser.preprocess_info is None:
    raise ValueError(f'{out_path}: the input has no preprocess section')
  if data.text is not None and not data.cfg_parser.layer_info_list and data.cfg_parser.train_info is None \
     and data.cfg_parser.global_info is None:
    out_path.write_text(data.text)   # already preprocess only: keep its comments
    return
  out_path.write_text(format_json({'preprocess': python_to_json(dict(data.cfg_parser.preprocess_info))}) + '\n')


###############################################################################
## cfg: legacy .cfg, read only
###############################################################################

def convert_legacy_value(text: str):
  try:
    return ast.literal_eval(text)
  except (ValueError, SyntaxError):
    return text


def parse_legacy_cfg(text: str) -> NpxCfgParser:
  """the .cfg reader as it was: [section] headers, key=value options, # or ; comments"""
  cfg_parser = NpxCfgParser()
  find_section_name = re.compile(r'\[([^]]+)\]').findall
  current_section = None
  for line in text.split('\n'):
    if line == '':
      continue
    if '[' in line and ']' in line:
      section_name = find_section_name(line)[0]
      current_section = NpxCfgSection(section_name)
      if section_name == 'train':
        cfg_parser.train_info = current_section
      elif section_name == 'preprocess':
        cfg_parser.preprocess_info = current_section
      elif section_name == 'global':
        cfg_parser.global_info = current_section
      else:
        cfg_parser.layer_info_list.append(current_section)
    elif line.startswith('\0') or line.startswith('#') or line.startswith(';') or line.startswith('\n'):
      pass
    else:
      key, value = line.split('=')
      current_section[key.strip()] = convert_legacy_value(value.strip())
  return cfg_parser

TOP_SECTIONS = ('preprocess', 'train', 'global')
_COMMENTED_OPTION = re.compile(r'^[#;]\s*([A-Za-z_][\w.]*)\s*=(.*)$')


def to_comment(line: str) -> str:
  body = line.lstrip('#;')
  if not body.strip():
    return '// ' + '-' * max(len(line) - 3, 10)  # a line of only '#' is a separator
  if set(body.strip()) <= set('#=-*'):
    return '// ' + body.strip()
  return '//' + (body if body.startswith(' ') else ' ' + body)


def json_value(text: str) -> str:
  return json.dumps(python_to_json(convert_legacy_value(text.strip())))


class Block:
  def __init__(self, name: str):
    self.name = name
    self.comment_before = []   # comment lines above the section header
    self.entry_list = []       # ('option', key, value_text) or ('comment', text)


def scan(text: str):
  header_comment_list = []
  block_list = []
  pending = []          # comments not yet given to a section
  current = None
  blank_since_option = True
  for raw_line in text.split('\n'):
    line = raw_line.strip()
    if not line:
      blank_since_option = True
      continue
    if line.startswith('[') and ']' in line:
      current = Block(line[1:line.index(']')])
      if not block_list:
        header_comment_list, pending = pending, []
      current.comment_before, pending = pending, []
      block_list.append(current)
      blank_since_option = False
    elif line[0] in '#;':
      match = _COMMENTED_OPTION.match(line)
      if current and match and not blank_since_option and not pending:
        current.entry_list.append(('commented', match.group(1), match.group(2)))
      else:
        pending.append(to_comment(line))
    else:
      assert current, f'option before any section: {line!r}'
      # comments between two options of a section stay above the option they precede
      current.entry_list += [('comment', each) for each in pending]
      pending = []
      key, value = line.split('=', 1)
      current.entry_list.append(('option', key.strip(), value))
      blank_since_option = False
  return header_comment_list, block_list, pending


def render_object(entry_list, head_item_list, indent: str, rename: bool = False) -> list:
  """head_item_list: '"key": value' strings placed first; entry_list: options and comments"""
  item_list = [('item', each) for each in head_item_list]
  for entry in entry_list:
    if entry[0] in ('option', 'commented'):
      key = json.dumps(JSON_RENAMED_KEYS.get(entry[1], entry[1]) if rename else entry[1])
    if entry[0] == 'option':
      item_list.append(('item', f'{key}: {entry[2]}'))
    elif entry[0] == 'commented':
      # "#reset_delay=False // bug": the value stops at an inner remark, which stays a remark
      remark = re.search(r'\s+((?://|#).*)$', entry[2])
      value_text = entry[2][:remark.start()] if remark else entry[2]
      tail = f'  // {remark.group(1).lstrip("/#").strip()}' if remark else ''
      item_list.append(('comment', f'// {key}: {json_value(value_text)}{tail}'))
    else:
      item_list.append(('comment', entry[1]))
  last_item = max([i for i, (kind, _) in enumerate(item_list) if kind == 'item'], default=-1)
  line_list = []
  for i, (kind, text) in enumerate(item_list):
    comma = ',' if (kind == 'item' and i < last_item) else ''
    line_list.append(f'{indent}{text}{comma}')
  return line_list


def convert_text(text: str) -> str:
  header_comment_list, block_list, trailing_comment_list = scan(text)

  # the values come from the real parser, so the .jsonc means exactly what the .cfg meant
  cfg_parser = parse_legacy_cfg(text)
  layer_list = cfg_parser.layer_info_list
  node_name_list = [default_node_name(section.name, i, layer_list) for i, section in enumerate(layer_list)]

  top_part = []   # (comment lines, rendered lines) per top-level key
  node_part = []
  layer_index = 0
  for block in block_list:
    if block.name in TOP_SECTIONS:
      section = getattr(cfg_parser, f'{block.name}_info')
      entry_list = [e if e[0] != 'option' else ('option', e[1], json.dumps(python_to_json(section[e[1]])))
                    for e in block.entry_list]
      body = render_object(entry_list, [], '    ')
      top_part.append((block.comment_before, [f'  {json.dumps(block.name)}: {{'] + body + ['  }']))
    else:
      section = layer_list[layer_index]
      assert section.name == block.name, (section.name, block.name)
      entry_list = []
      for entry in block.entry_list:
        if entry[0] == 'option':
          value = section[entry[1]]
          if entry[1] == 'skip_from' and isinstance(value, int):
            value = node_name_list[layer_index + value]
          entry = ('option', entry[1], json.dumps(python_to_json(value)))
        entry_list.append(entry)
      head = [f'"name": {json.dumps(node_name_list[layer_index])}', f'"type": {json.dumps(block.name)}']
      if not entry_list:
        lines = ['    {' + ', '.join(head) + '}']
      else:
        lines = ['    {'] + render_object(entry_list, head, '      ', rename=True) + ['    }']
      node_part.append((block.comment_before, lines))
      layer_index += 1

  out = ['{']
  out += ['  ' + each for each in header_comment_list]
  # None marks "nodes"; a preprocess-only file (no layers) has no "nodes" at all
  part_list = top_part + ([([], None)] if node_part else [])
  for i, (comment_list, lines) in enumerate(part_list):
    if out[-1] != '{':
      out.append('')
    if lines is None:
      out.append('  "nodes": [')
      for j, (node_comment_list, node_lines) in enumerate(node_part):
        if node_comment_list:
          if j > 0:
            out.append('')
          out += ['    ' + each for each in node_comment_list]
        node_lines = list(node_lines)
        if j < len(node_part) - 1:
          node_lines[-1] += ','
        out += node_lines
      out += ['    ' + each for each in trailing_comment_list]
      out.append('  ]')
    else:
      out += ['  ' + each for each in comment_list]
      lines = list(lines)
      if i < len(part_list) - 1:
        lines[-1] += ','
      out += lines
  if not node_part:
    out += ['  ' + each for each in trailing_comment_list]
  out.append('}')
  return '\n'.join(out) + '\n'


def check_same(cfg_text: str, jsonc_text: str):
  legacy = parse_legacy_cfg(cfg_text)
  converted = NpxCfgParser()
  converted.parse_json_text(jsonc_text)
  for key in TOP_SECTIONS:
    a = getattr(legacy, f'{key}_info')
    b = getattr(converted, f'{key}_info')
    assert (a is None) == (b is None) and (a is None or dict(a) == dict(b)), key
  assert len(legacy.layer_info_list) == len(converted.layer_info_list)
  for a, b in zip(legacy.layer_info_list, converted.layer_info_list):
    assert a.name == b.name and dict(a) == dict(b), (a.name, dict(a), dict(b))


def read_legacy_cfg(path: Path) -> AppData:
  cfg_text = path.read_text()
  jsonc_text = convert_text(cfg_text)
  check_same(cfg_text, jsonc_text)
  converted = NpxCfgParser()
  converted.parse_json_text(jsonc_text, str(path))
  return AppData(converted, jsonc_text)


###############################################################################
## nir: NIR graph <-> app
##
## Only what the network itself supports converts: a chain of Conv2d, Linear,
## Flatten, AvgPool2d/MaxPool2d and Leaky nodes, plus residual adds (Shortcut).
## NIR has no field for what it cannot express, so that goes to metadata:
##   graph metadata["npx"]          preprocess / train / global sections
##   node  metadata["npx_options"]  node options NIR has no parameter for
##                                  (neuron_type, reset_mechanism, ...)
##   node  metadata["npx_type"]     MaxPool2d (written as AvgPool2d, NIR has no
##                                  max pooling) or Shortcut (a projection node)
## A NIR file from another tool has no such metadata; the defaults that fill the
## gaps are listed as comments at the top of the .jsonc.
###############################################################################

NIR_DT = 1e-4  # snnTorch's NIR convention: beta = 1 - dt/tau, r = tau/dt
NIR_RESERVED_NAMES = ('input', 'output')
# options a NIR node holds as parameters; every other option goes to npx_options
NIR_PARAMETER_KEYS = {
  'Conv2d': ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'groups'),
  'Linear': ('in_features', 'out_features'),
  'AvgPool2d': ('kernel_size', 'stride', 'padding'),
  'MaxPool2d': ('kernel_size', 'stride', 'padding'),
  'Flatten': (),
  'Leaky': ('beta', 'threshold'),
}
SHORTCUT_OPS = ('Identity', 'Conv2d', 'AvgPool2d', 'MaxPool2d')


def _import_nir():
  try:
    import nir
    import numpy
  except ImportError:
    raise SystemExit('the nir package is needed: python3 -m pip install nir')
  return nir, numpy


def _out_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
  return (size + 2*padding - kernel_size) // stride + 1


## .jsonc -> NIR

def _nir_layer(nir, np, section, shape):
  """the NIR node for one layer (zero weights: the .jsonc has no trained weights), and its output shape"""
  name = section.name
  if name == 'Conv2d':
    k, s, p, g = section['kernel_size'], section.get('stride', 1), section.get('padding', 0), section.get('groups', 1)
    out_channels = section['out_channels']
    node = nir.Conv2d(input_shape=np.array(shape[1:]),
                      weight=np.zeros((out_channels, section['in_channels'] // g, k, k), dtype=np.float32),
                      stride=(s, s), padding=(p, p), dilation=(1, 1), groups=g,
                      bias=np.zeros(out_channels, dtype=np.float32))
    return node, (out_channels, _out_size(shape[1], k, s, p), _out_size(shape[2], k, s, p))
  if name in ('AvgPool2d', 'MaxPool2d'):
    k = section['kernel_size']
    s, p = section.get('stride', k), section.get('padding', 0)
    node = nir.AvgPool2d(kernel_size=np.array([k, k]), stride=np.array([s, s]), padding=np.array([p, p]))
    return node, (shape[0], _out_size(shape[1], k, s, p), _out_size(shape[2], k, s, p))
  if name == 'Linear':
    node = nir.Linear(weight=np.zeros((section['out_features'], section['in_features']), dtype=np.float32))
    return node, (section['out_features'],)
  if name == 'Flatten':
    size = int(np.prod(shape))
    node = nir.Flatten(input_type={'input': np.array(shape)}, start_dim=0, end_dim=-1)
    return node, (size,)
  if name == 'Leaky':
    beta = float(section.get('beta', 1.0))
    threshold = np.full(shape, float(section.get('threshold', 1.0)))
    if beta >= 1.0:
      node = nir.IF(r=np.ones(shape), v_threshold=threshold, v_reset=np.zeros(shape))
    else:
      tau = NIR_DT / (1.0 - beta)
      node = nir.LIF(tau=np.full(shape, tau), r=np.full(shape, tau / NIR_DT),
                     v_leak=np.zeros(shape), v_threshold=threshold, v_reset=np.zeros(shape))
    return node, tuple(shape)
  raise ValueError(f'{name} has no NIR node (supported: {", ".join(NIR_PARAMETER_KEYS)}, Shortcut)')


def _set_npx_metadata(node, section, npx_type=None, extra_skip_keys=()):
  options = {key: python_to_json(value) for key, value in section.items()
             if key not in NIR_PARAMETER_KEYS.get(section.name, ()) and key not in extra_skip_keys}
  metadata = {}
  if options:
    metadata['npx_options'] = options
  if npx_type:
    metadata['npx_type'] = npx_type
  if metadata:
    node.metadata = metadata


def jsonc_to_nir(cfg_parser: NpxCfgParser):
  nir, np = _import_nir()
  global_info = cfg_parser.global_info
  assert global_info, 'the network needs a "global" section (input_size, input_channels)'
  shape = (global_info['input_channels'],) + tuple(global_info['input_size'])
  node_name_list = cfg_parser.node_name_list()
  for each in node_name_list:
    if each in NIR_RESERVED_NAMES:
      raise ValueError(f'node name {each!r} is reserved for the NIR input/output nodes')

  nodes = {'input': nir.Input(input_type={'input': np.array(shape)})}
  edges = []
  shape_of = {'input': shape}
  previous = 'input'
  pending = []           # shortcut inputs waiting for the next node, which adds them
  identity_names = {}    # identity shortcuts make no node: keep their names on the node they feed
  for index, section in enumerate(cfg_parser.layer_info_list):
    name = node_name_list[index]
    if section.name == 'Shortcut':
      source = node_name_list[index + section['skip_from']]
      op = section.get('type', 'Conv2d')
      if op not in SHORTCUT_OPS:
        raise ValueError(f'{name}: unknown shortcut op {op!r}')
      if op == 'Identity':
        pending.append(source)
        identity_names[source] = name
      else:
        projection = NpxCfgSection(op, name)
        projection.update({key: value for key, value in section.items() if key not in ('skip_from', 'type')})
        node, shape_of[name] = _nir_layer(nir, np, projection, shape_of[source])
        _set_npx_metadata(node, projection, 'Shortcut')
        nodes[name] = node
        edges.append((source, name))
        pending.append(name)
      continue
    node, shape_of[name] = _nir_layer(nir, np, section, shape_of[previous])
    _set_npx_metadata(node, section, 'MaxPool2d' if section.name == 'MaxPool2d' else None)
    if identity_names:
      node.metadata = dict(node.metadata or {}, npx_identity_shortcuts=identity_names)
      identity_names = {}
    nodes[name] = node
    edges += [(previous, name)] + [(each, name) for each in pending]
    pending = []
    previous = name
  nodes['output'] = nir.Output(output_type={'output': np.array(shape_of[previous])})
  if identity_names:
    nodes['output'].metadata = {'npx_identity_shortcuts': identity_names}
  edges += [(previous, 'output')] + [(each, 'output') for each in pending]

  metadata = {'npx': {key: python_to_json(dict(getattr(cfg_parser, f'{key}_info')))
                      for key in TOP_SECTIONS if getattr(cfg_parser, f'{key}_info')}}
  # NIR's own type check gets depthwise convolutions (groups > 1) wrong, so it is skipped
  return nir.NIRGraph(nodes=nodes, edges=edges, metadata=metadata, type_check=False)


## NIR -> .jsonc

def _from_h5(value):
  """metadata read back from HDF5: numpy arrays/scalars and bytes to plain python"""
  if isinstance(value, dict):
    return {key: _from_h5(each) for key, each in value.items()}
  if isinstance(value, bytes):
    return value.decode()
  if hasattr(value, 'tolist'):
    return _from_h5(value.tolist())
  if isinstance(value, list):
    return [_from_h5(each) for each in value]
  return value


def _single(np, value, what: str, node_name: str):
  value = np.asarray(value).reshape(-1)
  if value.size == 0 or not np.allclose(value, value[0]):
    raise ValueError(f'{node_name}: {what} differs across the layer, but a layer here has one {what}')
  return float(value[0])


def _square(np, value, what: str, node_name: str) -> int:
  value = np.asarray(value).reshape(-1)
  if value.size == 0 or not np.all(value == value[0]):
    raise ValueError(f'{node_name}: {what} {value.tolist()} is not the same in both directions')
  return int(value[0])


def _nir_to_section(nir, np, node, node_name: str, note_list: list):
  """one NIR node to (section name, options); options NIR cannot hold come from npx_options"""
  metadata = _from_h5(dict(node.metadata or {}))
  npx_type = metadata.get('npx_type')
  options = {}
  if isinstance(node, nir.Conv2d):
    weight = np.asarray(node.weight)
    groups = int(node.groups)
    if weight.shape[2] != weight.shape[3]:
      raise ValueError(f'{node_name}: kernel {weight.shape[2]}x{weight.shape[3]} is not square')
    if np.any(np.asarray(node.dilation) != 1):
      raise ValueError(f'{node_name}: dilation {np.asarray(node.dilation).tolist()} is not supported')
    if node.bias is not None and np.any(np.asarray(node.bias) != 0):
      raise ValueError(f'{node_name}: a convolution bias is not supported (layers have bias=False)')
    k = int(weight.shape[2])
    stride = _square(np, node.stride, 'stride', node_name)
    if isinstance(node.padding, str):
      if node.padding == 'valid':
        padding = 0
      elif stride == 1 and k % 2 == 1:
        padding = (k - 1) // 2
      else:
        raise ValueError(f'{node_name}: padding="same" needs stride 1 and an odd kernel')
    else:
      padding = _square(np, node.padding, 'padding', node_name)
    section_name = 'Conv2d'
    options.update(in_channels=int(weight.shape[1]) * groups, out_channels=int(weight.shape[0]),
                   kernel_size=k, stride=stride, padding=padding, groups=groups)
  elif isinstance(node, (nir.Linear, nir.Affine)):
    if isinstance(node, nir.Affine) and np.any(np.asarray(node.bias) != 0):
      raise ValueError(f'{node_name}: a linear bias is not supported (layers have bias=False)')
    weight = np.asarray(node.weight)
    section_name = 'Linear'
    options.update(in_features=int(weight.shape[1]), out_features=int(weight.shape[0]))
  elif isinstance(node, nir.Flatten):
    section_name = 'Flatten'
  elif isinstance(node, nir.AvgPool2d):
    section_name = 'MaxPool2d' if npx_type == 'MaxPool2d' else 'AvgPool2d'
    options.update(kernel_size=_square(np, node.kernel_size, 'kernel_size', node_name),
                   stride=_square(np, node.stride, 'stride', node_name),
                   padding=_square(np, node.padding, 'padding', node_name))
  elif isinstance(node, (nir.LIF, nir.IF)):
    if node.v_reset is not None and not np.allclose(node.v_reset, 0):
      raise ValueError(f'{node_name}: v_reset other than 0 is not supported')
    if isinstance(node, nir.LIF):
      if not np.allclose(node.v_leak, 0):
        raise ValueError(f'{node_name}: v_leak other than 0 is not supported')
      tau = _single(np, node.tau, 'tau', node_name)
      input_scale = _single(np, node.r, 'r', node_name) * NIR_DT / tau
      beta = round(1.0 - NIR_DT / tau, 12)
    else:
      input_scale = _single(np, node.r, 'r', node_name)
      beta = 1.0
    if not np.isclose(input_scale, 1.0):
      raise ValueError(f'{node_name}: the input is scaled by {input_scale:g} (r*dt/tau for LIF, r for IF), '
                       f'only 1 is supported (dt={NIR_DT:g}, the snnTorch convention)')
    section_name = 'Leaky'
    options.update(beta=beta, threshold=_single(np, node.v_threshold, 'v_threshold', node_name))
    if 'reset_mechanism' not in metadata.get('npx_options', {}):
      options['reset_mechanism'] = 'zero'
      note_list.append(f'{node_name}: reset_mechanism "zero" (NIR resets the membrane to v_reset)')
  else:
    supported = 'Conv2d, Linear, Affine (zero bias), Flatten, AvgPool2d, LIF, IF'
    raise ValueError(f'{node_name}: NIR {type(node).__name__} is not supported (supported: {supported})')
  options.update(metadata.get('npx_options', {}))
  return section_name, options


def _main_path(np, nodes, edges, is_shortcut):
  """the longest input->output path; shortcut projection nodes stay off it"""
  pred = {name: [] for name in nodes}
  succ = {name: [] for name in nodes}
  for source, target in edges:
    pred[target].append(source)
    succ[source].append(target)
  order = []
  in_degree = {name: len(pred[name]) for name in nodes}
  ready = [name for name in nodes if in_degree[name] == 0]
  while ready:
    name = ready.pop(0)
    order.append(name)
    for target in succ[name]:
      in_degree[target] -= 1
      if in_degree[target] == 0:
        ready.append(target)
  if len(order) != len(nodes):
    raise ValueError('the NIR graph has a cycle (recurrent connections are not supported)')
  score = {}
  best_pred = {}
  file_order = {name: i for i, name in enumerate(nodes)}
  for name in order:
    gain = -1_000_000 if is_shortcut(name) else 1
    # a tie (two branches of the same length into one add) only arises in a NIR file from another
    # tool: the sum is the same either way, so the node first in the graph's node order (alphabetical
    # after an HDF5 round trip) is the main one. Our own shortcut projections are tagged and never tie.
    candidates = [(score[each], -file_order[each], each) for each in pred[name] if each in score]
    if nodes[name].__class__.__name__ == 'Input':
      score[name] = 0
    elif candidates:
      best = max(candidates)
      score[name] = best[0] + gain
      best_pred[name] = best[2]
  path = ['output']
  while path[-1] in best_pred:
    path.append(best_pred[path[-1]])
  path.reverse()
  if path[0] != 'input':
    raise ValueError('there is no path from the input node to the output node')
  return path, pred, succ


def nir_to_jsonc(graph, source: str = '<nir>') -> str:
  nir, np = _import_nir()
  nodes = dict(graph.nodes)
  input_names = [name for name, node in nodes.items() if isinstance(node, nir.Input)]
  output_names = [name for name, node in nodes.items() if isinstance(node, nir.Output)]
  if len(input_names) != 1 or len(output_names) != 1:
    raise ValueError(f'one input and one output node are needed, found {input_names} and {output_names}')
  for name, node in nodes.items():
    if isinstance(node, nir.NIRGraph):
      raise ValueError(f'{name}: nested NIR graphs (e.g. recurrent layers) are not supported')
  # name the input/output nodes "input"/"output" internally
  rename = {input_names[0]: 'input', output_names[0]: 'output'}
  nodes = {rename.get(name, name): node for name, node in nodes.items()}
  edges = [(rename.get(a, a), rename.get(b, b)) for a, b in graph.edges]

  def is_shortcut(name):
    return _from_h5(dict(nodes[name].metadata or {})).get('npx_type') == 'Shortcut'

  path, pred, succ = _main_path(np, nodes, edges, is_shortcut)
  position = {name: i for i, name in enumerate(path)}
  for name in nodes:
    if name in position:
      continue
    if len(pred[name]) != 1 or len(succ[name]) != 1 or pred[name][0] not in position \
       or succ[name][0] not in position or position[pred[name][0]] >= position[succ[name][0]]:
      raise ValueError(f'{name}: only a chain with residual adds is supported; this node is neither on '
                       f'the main path nor a shortcut from one main node to a later one')

  note_list = []
  layer_list = []
  shortcut_count = 0
  for i, name in enumerate(path[1:], start=1):
    node = nodes[name]
    identity_names = _from_h5(dict(node.metadata or {})).get('npx_identity_shortcuts', {})
    for skip in pred[name]:
      if skip == path[i - 1]:
        continue
      if skip in position:
        skip_source, projection = skip, None
      else:
        skip_source, projection = pred[skip][0], skip
      if skip_source == 'input':
        raise ValueError(f'{name}: a shortcut from the input node is not supported, it must start at a layer')
      if projection is None:
        shortcut_count += 1
        section = NpxCfgSection('Shortcut', identity_names.get(skip_source, f'shortcut_{shortcut_count}'))
        section['skip_from'] = skip_source  # a node name until every layer has its index
        section['type'] = 'Identity'
      else:
        op, options = _nir_to_section(nir, np, nodes[projection], projection, note_list)
        if op not in SHORTCUT_OPS:
          raise ValueError(f'{projection}: a shortcut cannot be a {op}')
        if op == 'Conv2d':
          # a shortcut convolution has no groups option (npx_module.Shortcut ignores it)
          if options.pop('groups', 1) != 1:
            raise ValueError(f'{projection}: a grouped convolution is not supported in a shortcut')
        section = NpxCfgSection('Shortcut', projection)
        section['skip_from'] = skip_source
        section['type'] = op
        section.update(options)
      layer_list.append(section)
    if name == 'output':
      break
    section_name, options = _nir_to_section(nir, np, node, name, note_list)
    section = NpxCfgSection(section_name, name)
    section.update(options)
    layer_list.append(section)

  index_of = {section.node_name: i for i, section in enumerate(layer_list)}
  for i, section in enumerate(layer_list):
    if section.name == 'Shortcut':
      section['skip_from'] = index_of[section['skip_from']] - i

  cfg_parser = NpxCfgParser()
  cfg_parser.layer_info_list = layer_list
  npx = _from_h5(dict(graph.metadata or {})).get('npx', {})
  input_shape = [int(each) for each in np.asarray(nodes['input'].input_type['input']).reshape(-1)]
  output_shape = [int(each) for each in np.asarray(nodes['output'].output_type['output']).reshape(-1)]
  section_note_list = []
  for key in TOP_SECTIONS:
    section = NpxCfgSection(key)
    if key in npx:
      section.update(json_to_python(npx[key]))
    elif key == 'global':
      if len(input_shape) == 3:
        channels, size = input_shape[0], tuple(input_shape[1:])
      elif len(input_shape) == 2:
        channels, size = 1, tuple(input_shape)
      else:
        raise ValueError(f'input shape {input_shape} is not an image (C,H,W or H,W)')
      section.update(input_size=size, input_channels=channels,
                     output_classes=int(np.prod(output_shape)), neuron_type='ws8')
      section_note_list.append('global: from the NIR input/output shapes; neuron_type "ws8" is a default')
    elif key == 'preprocess':
      section.update(input='TODO_opendataset', timesteps=4, step_generation='direct')
      section_note_list.append('preprocess: TODO, NIR has no preprocessing (set "input" to the dataset)')
    elif key == 'train':
      section.update(epoch=10, kfold=5, repeat=1)
      section_note_list.append('train: defaults, NIR has no training options')
    setattr(cfg_parser, f'{key}_info', section)

  text = cfg_parser.to_json_text()
  header = [f'  // converted from {source} by npx_cfg_convert.py (NIR weights are not imported)']
  header += [f'  // {each}' for each in section_note_list + note_list]
  return '{\n' + '\n'.join(header) + '\n' + text[2:]


def read_nir(path: Path) -> AppData:
  nir, _ = _import_nir()
  # NIR's type check gets depthwise convolutions (groups > 1) wrong, as when writing
  text = nir_to_jsonc(nir.read(str(path), type_check=False), path.name)
  cfg_parser = NpxCfgParser()
  cfg_parser.parse_json_text(text, str(path))
  return AppData(cfg_parser, text)


def write_nir(data: AppData, out_path: Path):
  nir, _ = _import_nir()
  nir.write(str(out_path), jsonc_to_nir(data.cfg_parser))


###############################################################################
## Registered formats: add a new one here
###############################################################################

FORMAT_LIST += [
  Format('snn', (APP_CFG_SUFFIX, '.snn.json', '.jsonc', '.json'), 'the app format (preprocess, train, network)',
         read=read_snn, write=write_snn, default_target='nir'),
  Format('cfg', ('.cfg',), 'legacy app format, read only',
         read=read_legacy_cfg, default_target='snn'),
  Format('preprocess', ('.preprocess.jsonc', '_preprocess.cfg'), 'a preprocess section only',
         read=lambda path: read_legacy_cfg(path) if path.suffix == '.cfg' else read_snn(path),
         write=write_preprocess, default_target='preprocess'),
  Format('nir', ('.nir',), 'NIR graph: the network only, zero weights, the rest in metadata',
         read=read_nir, write=write_nir, default_target='snn'),
]


def format_table() -> str:
  line_list = []
  for each in FORMAT_LIST:
    mode = ('r' if each.read else '-') + ('w' if each.write else '-')
    default = f'  (default -> {each.default_target})' if each.default_target else ''
    line_list.append(f'  {each.name:11s} {mode}  {", ".join(each.suffix_list):40s} {each.description}{default}')
  return '\n'.join(line_list)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert app files between formats.',
                                   epilog='formats (r: read, w: write):\n' + format_table(),
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('input', nargs='*', help='input files; the format comes from the file name')
  parser.add_argument('-t', '--to', help='output format (default: the input format\'s default target)')
  parser.add_argument('-o', '--output', help='output directory (default: next to each input)')
  parser.add_argument('-f', '--force', action='store_true', help='overwrite an existing output')
  parser.add_argument('--list', action='store_true', help='list the formats and exit')
  args = parser.parse_args()
  if args.list:
    print(format_table())
    raise SystemExit(0)
  if not args.input:
    parser.error('no input file')
  for each in args.input:
    convert_file(Path(each), args.to, Path(args.output) if args.output else None, args.force)
