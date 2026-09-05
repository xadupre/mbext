# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Conversions between the internal IR objects and ONNX protos.

The protos are created with :mod:`onnx_light.onnx`.  Deserialization is written
with duck typing so that protos coming from another implementation of the ONNX
Python API (for example the ones returned by ``onnxruntime.quantization``) are
supported as well.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modelbuilder.helpers.onnx_helper import onnx

from ._core import (
    Attr,
    ExternalTensor,
    Function,
    Graph,
    Model,
    Node,
    PackedTensor,
    Shape,
    StringTensor,
    Tensor,
    TensorBase,
    TensorType,
    Value,
)
from ._enums import AttributeType, DataType

#: Signature of the callback writing a tensor into an external data file.
#: It returns ``(location, offset, length)`` or ``None`` when the tensor must be
#: kept inside the ONNX proto.
ExternalDataWriter = Callable[[TensorBase], Optional[Tuple[str, int, int]]]

_SUB_BYTE_TYPES = frozenset({DataType.INT4, DataType.UINT4, DataType.FLOAT4E2M1, DataType.INT2, DataType.UINT2})

# ONNX data types whose values are stored in the ``int32_data`` field.
_INT32_DATA_TYPES = frozenset(
    {
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.UINT8,
        DataType.UINT16,
        DataType.BOOL,
        DataType.FLOAT16,
        DataType.BFLOAT16,
        DataType.FLOAT8E4M3FN,
        DataType.FLOAT8E4M3FNUZ,
        DataType.FLOAT8E5M2,
        DataType.FLOAT8E5M2FNUZ,
        DataType.FLOAT8E8M0,
        DataType.INT4,
        DataType.UINT4,
        DataType.FLOAT4E2M1,
        DataType.INT2,
        DataType.UINT2,
    }
)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def serialize_tensor_into(tensor_proto, from_: TensorBase, *, external_data_writer: Optional[ExternalDataWriter] = None) -> None:
    """Serializes a tensor into a ``TensorProto``."""
    if from_.name:
        tensor_proto.name = from_.name
    if from_.doc_string:
        tensor_proto.doc_string = from_.doc_string
    tensor_proto.data_type = int(from_.dtype)
    for dim in from_.shape.numpy():
        tensor_proto.dims.append(int(dim))

    if isinstance(from_, StringTensor):
        for value in from_.raw:
            tensor_proto.string_data.append(value)
        return

    if external_data_writer is not None:
        written = external_data_writer(from_)
        if written is not None:
            _set_external_data(tensor_proto, *written)
            return

    if isinstance(from_, ExternalTensor):
        _set_external_data(tensor_proto, from_.location, from_.offset, from_.length)
        return

    tensor_proto.raw_data = from_.tobytes()


def _set_external_data(tensor_proto, location: str, offset: int, length: int) -> None:
    """Fills the external data fields of a ``TensorProto``."""
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        entry = tensor_proto.external_data.add()
        entry.key = key
        entry.value = value


def serialize_tensor(from_: TensorBase):
    """Serializes a tensor into a new ``TensorProto``."""
    tensor_proto = onnx.TensorProto()
    serialize_tensor_into(tensor_proto, from_)
    return tensor_proto


def serialize_type_into(type_proto, from_: TensorType, shape: Optional[Shape]) -> None:
    """Serializes a type and a shape into a ``TypeProto``."""
    tensor_type = type_proto.tensor_type
    if from_ is not None:
        tensor_type.elem_type = int(from_.dtype)
        if from_.denotation:
            type_proto.denotation = from_.denotation
    if shape is not None:
        # Touching the field is required so that a scalar (rank 0) is
        # serialized as an empty shape instead of an unknown one.
        tensor_type.shape.ClearField("dim")
        for dim in shape:
            dim_proto = tensor_type.shape.dim.add()
            if isinstance(dim, int):
                dim_proto.dim_value = dim
            elif isinstance(dim, str):
                dim_proto.dim_param = dim


def serialize_value_into(value_info_proto, from_: Value, *, name: str = "") -> None:
    """Serializes a value into a ``ValueInfoProto``."""
    value_info_proto.name = name or (from_.name or "")
    if from_.type is not None or from_.shape is not None:
        serialize_type_into(value_info_proto.type, from_.type, from_.shape)
    if from_.doc_string:
        value_info_proto.doc_string = from_.doc_string
    _serialize_metadata_props_into(value_info_proto.metadata_props, from_.metadata_props)


def _serialize_metadata_props_into(container, metadata_props: Optional[Dict[str, str]]) -> None:
    if not metadata_props:
        return
    for key, value in metadata_props.items():
        entry = container.add()
        entry.key = key
        entry.value = value


def serialize_attribute_into(attribute_proto, from_: Attr, *, external_data_writer: Optional[ExternalDataWriter] = None) -> None:
    """Serializes an attribute into an ``AttributeProto``."""
    attribute_proto.name = from_.name
    if from_.doc_string:
        attribute_proto.doc_string = from_.doc_string
    if from_.is_ref():
        attribute_proto.ref_attr_name = from_.ref_attr_name
        attribute_proto.type = int(from_.type)
        return
    attribute_proto.type = int(from_.type)
    kind = from_.type
    value = from_.value
    if kind == AttributeType.INT:
        attribute_proto.i = int(value)
    elif kind == AttributeType.FLOAT:
        attribute_proto.f = float(value)
    elif kind == AttributeType.STRING:
        attribute_proto.s = value.encode("utf-8") if isinstance(value, str) else value
    elif kind == AttributeType.TENSOR:
        serialize_tensor_into(attribute_proto.t, value)
    elif kind == AttributeType.GRAPH:
        serialize_graph_into(attribute_proto.g, value, external_data_writer=external_data_writer)
    elif kind == AttributeType.INTS:
        for item in value:
            attribute_proto.ints.append(int(item))
    elif kind == AttributeType.FLOATS:
        for item in value:
            attribute_proto.floats.append(float(item))
    elif kind == AttributeType.STRINGS:
        for item in value:
            attribute_proto.strings.append(item.encode("utf-8") if isinstance(item, str) else item)
    elif kind == AttributeType.TENSORS:
        for item in value:
            serialize_tensor_into(attribute_proto.tensors.add(), item)
    elif kind == AttributeType.GRAPHS:
        for item in value:
            serialize_graph_into(attribute_proto.graphs.add(), item, external_data_writer=external_data_writer)
    elif kind == AttributeType.TYPE_PROTO:
        serialize_type_into(attribute_proto.tp, value, None)
    else:
        raise NotImplementedError(f"Unable to serialize an attribute of type {kind!r}.")


def _remove_trailing_outputs(outputs: Sequence[Value]) -> List[Value]:
    """Removes the trailing optional (unnamed) outputs of a node."""
    kept = list(outputs)
    while kept and not kept[-1].name:
        kept.pop()
    return kept


def serialize_node_into(node_proto, from_: Node, *, external_data_writer: Optional[ExternalDataWriter] = None) -> None:
    """Serializes a node into a ``NodeProto``."""
    node_proto.op_type = from_.op_type
    if from_.domain:
        node_proto.domain = from_.domain
    if from_.name:
        node_proto.name = from_.name
    if from_.overload:
        node_proto.overload = from_.overload
    if from_.doc_string:
        node_proto.doc_string = from_.doc_string
    _serialize_metadata_props_into(node_proto.metadata_props, from_.metadata_props)
    for input_ in from_.inputs:
        node_proto.input.append("" if input_ is None else (input_.name or ""))
    for output in _remove_trailing_outputs(from_.outputs):
        node_proto.output.append(output.name or "")
    for attr in from_.attributes.values():
        serialize_attribute_into(node_proto.attribute.add(), attr, external_data_writer=external_data_writer)


def _should_create_value_info_for_value(value: Value) -> bool:
    """Returns ``True`` when a ``ValueInfoProto`` should be created for a value."""
    if value.shape is None and value.type is None and not value.metadata_props and not value.doc_string:
        return False
    return bool(value.name)


def serialize_graph_into(graph_proto, from_: Graph, *, external_data_writer: Optional[ExternalDataWriter] = None) -> None:
    """Serializes a graph into a ``GraphProto``."""
    if from_.name:
        graph_proto.name = from_.name
    if from_.doc_string:
        graph_proto.doc_string = from_.doc_string
    for input_ in from_.inputs:
        serialize_value_into(graph_proto.input.add(), input_)
    input_names = {input_.name for input_ in from_.inputs}
    for value in from_.initializers.values():
        if _should_create_value_info_for_value(value) and value.name not in input_names:
            serialize_value_into(graph_proto.value_info.add(), value)
        if value.const_value is None:
            continue
        value.const_value.name = value.name
        serialize_tensor_into(graph_proto.initializer.add(), value.const_value, external_data_writer=external_data_writer)
    for node in from_:
        serialize_node_into(graph_proto.node.add(), node, external_data_writer=external_data_writer)
        for node_output in node.outputs:
            if node_output.is_graph_output():
                continue
            if not _should_create_value_info_for_value(node_output):
                continue
            serialize_value_into(graph_proto.value_info.add(), node_output)
    for output in from_.outputs:
        serialize_value_into(graph_proto.output.add(), output)
    _serialize_metadata_props_into(graph_proto.metadata_props, from_.metadata_props)


def serialize_graph(from_: Graph):
    """Serializes a graph into a new ``GraphProto``."""
    graph_proto = onnx.GraphProto()
    serialize_graph_into(graph_proto, from_)
    return graph_proto


def serialize_function_into(function_proto, from_: Function, *, external_data_writer: Optional[ExternalDataWriter] = None) -> None:
    """Serializes a local function into a ``FunctionProto``."""
    function_proto.domain = from_.domain
    function_proto.name = from_.name
    if from_.overload:
        function_proto.overload = from_.overload
    if from_.doc_string:
        function_proto.doc_string = from_.doc_string
    for name in from_.attributes:
        function_proto.attribute.append(name)
    for input_ in from_.inputs:
        function_proto.input.append(input_.name or "")
    for output in from_.outputs:
        function_proto.output.append(output.name or "")
    for domain, version in from_.opset_imports.items():
        opset = function_proto.opset_import.add()
        opset.domain = domain
        opset.version = version
    for node in from_.graph:
        serialize_node_into(function_proto.node.add(), node, external_data_writer=external_data_writer)
    _serialize_metadata_props_into(function_proto.metadata_props, from_.metadata_props)


def serialize_function(from_: Function):
    """Serializes a local function into a new ``FunctionProto``."""
    function_proto = onnx.FunctionProto()
    serialize_function_into(function_proto, from_)
    return function_proto


def serialize_model_into(model_proto, from_: Model, *, external_data_writer: Optional[ExternalDataWriter] = None) -> None:
    """Serializes a model into a ``ModelProto``."""
    model_proto.ir_version = from_.ir_version
    if from_.producer_name:
        model_proto.producer_name = from_.producer_name
    if from_.producer_version:
        model_proto.producer_version = from_.producer_version
    if from_.domain:
        model_proto.domain = from_.domain
    if from_.model_version is not None:
        model_proto.model_version = from_.model_version
    if from_.doc_string:
        model_proto.doc_string = from_.doc_string
    for domain, version in from_.graph.opset_imports.items():
        opset = model_proto.opset_import.add()
        opset.domain = domain
        opset.version = version
    _serialize_metadata_props_into(model_proto.metadata_props, from_.metadata_props)
    serialize_graph_into(model_proto.graph, from_.graph, external_data_writer=external_data_writer)
    for func in from_.functions.values():
        serialize_function_into(model_proto.functions.add(), func, external_data_writer=external_data_writer)


def serialize_model(from_: Model, *, external_data_writer: Optional[ExternalDataWriter] = None):
    """Serializes a model into a new ``ModelProto``."""
    model_proto = onnx.ModelProto()
    serialize_model_into(model_proto, from_, external_data_writer=external_data_writer)
    return model_proto


def to_proto(from_: Any):
    """Serializes an IR object (model, graph, function, node, tensor or value)."""
    if isinstance(from_, Model):
        return serialize_model(from_)
    if isinstance(from_, Graph):
        return serialize_graph(from_)
    if isinstance(from_, Function):
        return serialize_function(from_)
    if isinstance(from_, TensorBase):
        return serialize_tensor(from_)
    if isinstance(from_, Node):
        node_proto = onnx.NodeProto()
        serialize_node_into(node_proto, from_)
        return node_proto
    if isinstance(from_, Value):
        value_info = onnx.ValueInfoProto()
        serialize_value_into(value_info, from_)
        return value_info
    raise TypeError(f"Unable to serialize an object of type {type(from_)}.")


# --------------------------------------------------------------------------
# Deserialization
# --------------------------------------------------------------------------


def _external_data_info(tensor_proto) -> Dict[str, str]:
    return {entry.key: entry.value for entry in tensor_proto.external_data}


def deserialize_tensor(tensor_proto, base_dir: str = "") -> TensorBase:
    """Creates a tensor from a ``TensorProto``."""
    dtype = DataType(int(tensor_proto.data_type))
    dims = tuple(int(d) for d in tensor_proto.dims)
    name = tensor_proto.name or None

    if int(getattr(tensor_proto, "data_location", 0)) == int(onnx.TensorProto.EXTERNAL):
        info = _external_data_info(tensor_proto)
        return ExternalTensor(
            location=info.get("location", ""),
            offset=int(info.get("offset", 0)),
            length=int(info.get("length", 0)),
            dtype=dtype,
            shape=dims,
            name=name,
            base_dir=base_dir,
        )

    if dtype == DataType.STRING:
        return StringTensor(list(tensor_proto.string_data), shape=dims, name=name)

    raw = bytes(tensor_proto.raw_data)
    if raw:
        if dtype in _SUB_BYTE_TYPES:
            return PackedTensor(np.frombuffer(raw, dtype=np.uint8).copy(), dtype, shape=dims, name=name)
        array = np.frombuffer(raw, dtype=dtype.numpy()).copy()
        return Tensor(array.reshape(dims), dtype=dtype, name=name)

    array = _array_from_typed_fields(tensor_proto, dtype, dims)
    return Tensor(array, dtype=dtype, name=name)


def _array_from_typed_fields(tensor_proto, dtype: DataType, dims: Tuple[int, ...]) -> np.ndarray:
    """Reads the tensor values stored in the typed fields of a ``TensorProto``."""
    if dtype == DataType.FLOAT:
        array = np.array(list(tensor_proto.float_data), dtype=np.float32)
    elif dtype == DataType.DOUBLE:
        array = np.array(list(tensor_proto.double_data), dtype=np.float64)
    elif dtype == DataType.INT64:
        array = np.array(list(tensor_proto.int64_data), dtype=np.int64)
    elif dtype in {DataType.UINT32, DataType.UINT64}:
        array = np.array(list(tensor_proto.uint64_data), dtype=np.uint64).astype(dtype.numpy())
    elif dtype in _INT32_DATA_TYPES:
        raw = np.array(list(tensor_proto.int32_data), dtype=np.int32)
        if dtype in {DataType.FLOAT16, DataType.BFLOAT16}:
            array = raw.astype(np.uint16).view(dtype.numpy())
        elif dtype in _SUB_BYTE_TYPES:
            array = raw.astype(np.uint8).view(dtype.numpy())
        else:
            array = raw.astype(dtype.numpy())
    else:
        raise NotImplementedError(f"Unable to read the values of a tensor with type {dtype!r}.")
    size = int(np.prod(dims)) if dims else 1
    if array.size < size:
        array = np.zeros(size, dtype=array.dtype)
    return array.reshape(dims)


def deserialize_type(type_proto) -> Tuple[Optional[TensorType], Optional[Shape]]:
    """Reads the element type and the shape stored in a ``TypeProto``."""
    tensor_type = type_proto.tensor_type
    elem_type = int(tensor_type.elem_type)
    ir_type = TensorType(elem_type) if elem_type else None
    dims: List[Any] = []
    if hasattr(tensor_type, "has_shape"):
        has_shape = bool(tensor_type.has_shape())
    else:
        has_shape = tensor_type.HasField("shape")
    for dim in tensor_type.shape.dim:
        has_shape = True
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return ir_type, (Shape(dims) if has_shape else None)


def deserialize_value_info_proto(value_info_proto, value: Optional[Value] = None) -> Value:
    """Creates or updates a value from a ``ValueInfoProto``."""
    if value is None:
        value = Value(name=value_info_proto.name)
    ir_type, shape = deserialize_type(value_info_proto.type)
    if ir_type is not None:
        value.type = ir_type
    if shape is not None:
        value.shape = shape
    if value_info_proto.doc_string:
        value.doc_string = value_info_proto.doc_string
    return value


def deserialize_attribute(attribute_proto, base_dir: str = "") -> Attr:
    """Creates an attribute from an ``AttributeProto``."""
    name = attribute_proto.name
    kind = AttributeType(int(attribute_proto.type))
    if attribute_proto.ref_attr_name:
        return Attr(name, kind, None, ref_attr_name=attribute_proto.ref_attr_name)
    if kind == AttributeType.INT:
        return Attr(name, kind, int(attribute_proto.i))
    if kind == AttributeType.FLOAT:
        return Attr(name, kind, float(attribute_proto.f))
    if kind == AttributeType.STRING:
        return Attr(name, kind, bytes(attribute_proto.s).decode("utf-8"))
    if kind == AttributeType.TENSOR:
        return Attr(name, kind, deserialize_tensor(attribute_proto.t, base_dir))
    if kind == AttributeType.GRAPH:
        return Attr(name, kind, deserialize_graph(attribute_proto.g, base_dir=base_dir))
    if kind == AttributeType.INTS:
        return Attr(name, kind, [int(i) for i in attribute_proto.ints])
    if kind == AttributeType.FLOATS:
        return Attr(name, kind, [float(f) for f in attribute_proto.floats])
    if kind == AttributeType.STRINGS:
        return Attr(name, kind, [bytes(s).decode("utf-8") for s in attribute_proto.strings])
    if kind == AttributeType.TENSORS:
        return Attr(name, kind, [deserialize_tensor(t, base_dir) for t in attribute_proto.tensors])
    if kind == AttributeType.GRAPHS:
        return Attr(name, kind, [deserialize_graph(g, base_dir=base_dir) for g in attribute_proto.graphs])
    if kind == AttributeType.TYPE_PROTO:
        ir_type, _ = deserialize_type(attribute_proto.tp)
        return Attr(name, kind, ir_type)
    raise NotImplementedError(f"Unable to deserialize an attribute of type {kind!r}.")


def deserialize_graph(graph_proto, base_dir: str = "", outer_values: Optional[Dict[str, Value]] = None) -> Graph:
    """Creates a graph from a ``GraphProto``."""
    values: Dict[str, Value] = dict(outer_values or {})
    value_infos = {info.name: info for info in graph_proto.value_info}

    inputs = []
    for input_proto in graph_proto.input:
        value = Value(name=input_proto.name)
        deserialize_value_info_proto(input_proto, value)
        values[value.name] = value
        inputs.append(value)

    initializers = []
    for tensor_proto in graph_proto.initializer:
        name = tensor_proto.name
        tensor = deserialize_tensor(tensor_proto, base_dir)
        value = values.get(name)
        if value is None:
            value = Value(name=name)
            values[name] = value
        if name in value_infos:
            deserialize_value_info_proto(value_infos[name], value)
        else:
            value.type = TensorType(tensor.dtype)
            value.shape = Shape(tensor.shape)
        value.const_value = tensor
        initializers.append(value)

    graph = Graph(inputs, (), nodes=(), initializers=initializers, name=graph_proto.name or None, doc_string=graph_proto.doc_string or None)

    for node_proto in graph_proto.node:
        graph.append(_deserialize_node(node_proto, values, value_infos, base_dir))

    for output_proto in graph_proto.output:
        value = values.get(output_proto.name)
        if value is None:
            value = Value(name=output_proto.name)
            values[output_proto.name] = value
        deserialize_value_info_proto(output_proto, value)
        graph.outputs.append(value)

    return graph


def _deserialize_node(node_proto, values: Dict[str, Value], value_infos: Dict[str, Any], base_dir: str) -> Node:
    inputs: List[Optional[Value]] = []
    for name in node_proto.input:
        if not name:
            inputs.append(None)
            continue
        value = values.get(name)
        if value is None:
            value = Value(name=name)
            values[name] = value
        inputs.append(value)

    outputs: List[Value] = []
    for name in node_proto.output:
        value = Value(name=name) if name else Value(name="")
        if name:
            values[name] = value
            if name in value_infos:
                deserialize_value_info_proto(value_infos[name], value)
        outputs.append(value)

    attributes = [deserialize_attribute(attr, base_dir) for attr in node_proto.attribute]
    return Node(
        node_proto.domain,
        node_proto.op_type,
        inputs,
        attributes,
        outputs=outputs,
        name=node_proto.name or None,
        overload=getattr(node_proto, "overload", "") or "",
        doc_string=node_proto.doc_string or None,
    )


def deserialize_function(function_proto, base_dir: str = "") -> Function:
    """Creates a local function from a ``FunctionProto``."""
    values: Dict[str, Value] = {}
    inputs = []
    for name in function_proto.input:
        value = Value(name=name)
        values[name] = value
        inputs.append(value)

    opset_imports = {opset.domain: opset.version for opset in function_proto.opset_import}
    graph = Graph(inputs, (), nodes=(), opset_imports=opset_imports, name=function_proto.name)
    for node_proto in function_proto.node:
        graph.append(_deserialize_node(node_proto, values, {}, base_dir))
    for name in function_proto.output:
        value = values.get(name)
        if value is None:
            value = Value(name=name)
            values[name] = value
        graph.outputs.append(value)

    return Function(
        function_proto.domain,
        function_proto.name,
        getattr(function_proto, "overload", "") or "",
        graph=graph,
        attributes=list(function_proto.attribute),
    )


def deserialize_model(model_proto, base_dir: str = "") -> Model:
    """Creates a model from a ``ModelProto``."""
    graph = deserialize_graph(model_proto.graph, base_dir=base_dir)
    graph.opset_imports.update({opset.domain: opset.version for opset in model_proto.opset_import})
    functions = [deserialize_function(func, base_dir) for func in model_proto.functions]
    return Model(
        graph,
        ir_version=int(model_proto.ir_version),
        producer_name=model_proto.producer_name or None,
        producer_version=model_proto.producer_version or None,
        domain=model_proto.domain or None,
        model_version=int(model_proto.model_version) if model_proto.model_version else None,
        doc_string=model_proto.doc_string or None,
        functions=functions,
    )


def from_proto(proto: Any, base_dir: str = "") -> Any:
    """Creates an IR object from a proto (model, graph, function or tensor)."""
    type_name = type(proto).__name__
    if type_name == "ModelProto":
        return deserialize_model(proto, base_dir=base_dir)
    if type_name == "GraphProto":
        return deserialize_graph(proto, base_dir=base_dir)
    if type_name == "FunctionProto":
        return deserialize_function(proto, base_dir=base_dir)
    if type_name == "TensorProto":
        return deserialize_tensor(proto, base_dir=base_dir)
    if type_name == "ValueInfoProto":
        return deserialize_value_info_proto(proto)
    if type_name == "NodeProto":
        return _deserialize_node(proto, {}, {}, base_dir)
    raise TypeError(f"Unable to deserialize an object of type {type(proto)}.")


def set_base_dir(graph: Graph, base_dir: str) -> None:
    """Sets the base directory of every external tensor of a graph."""
    for current in graph.graphs():
        for value in current.initializers.values():
            if isinstance(value.const_value, ExternalTensor):
                value.const_value.base_dir = os.fspath(base_dir)
