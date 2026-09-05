# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""In-memory graph representation of an ONNX model.

This module implements the subset of the ``onnx_ir`` (ir-py) API used by
mbext.  The objects are plain Python containers; they are converted to (and
from) ``onnx_light`` protos by :mod:`modelbuilder.ir._serde`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

from ._enums import AttributeType, DataType

_SUB_BYTE_TYPES = frozenset({DataType.INT4, DataType.UINT4, DataType.FLOAT4E2M1, DataType.INT2, DataType.UINT2})


def _is_torch_tensor(value: Any) -> bool:
    """Returns ``True`` when ``value`` is a ``torch.Tensor`` without importing torch."""
    return type(value).__module__.startswith("torch") and type(value).__name__ in ("Tensor", "Parameter")


class Shape:
    """Shape of a tensor or a value.

    Dimensions are stored as ``int`` (static), ``str`` (symbolic) or ``None``
    (unknown).
    """

    __slots__ = ("_dims",)

    def __init__(self, dims: Union["Shape", Sequence[Any], None] = ()):
        if isinstance(dims, Shape):
            dims = dims._dims
        self._dims: List[Union[int, str, None]] = [self._normalize(d) for d in (dims or ())]

    @staticmethod
    def _normalize(dim: Any) -> Union[int, str, None]:
        if dim is None:
            return None
        if isinstance(dim, (int, np.integer)):
            return int(dim)
        if isinstance(dim, str):
            return dim
        # torch.SymInt and other integer-like objects
        try:
            return int(dim)
        except (TypeError, ValueError):
            return str(dim)

    @property
    def dims(self) -> Tuple[Union[int, str, None], ...]:
        "Returns the dimensions as a tuple."
        return tuple(self._dims)

    def numpy(self) -> Tuple[int, ...]:
        """Returns the shape as a tuple of integers, static dimensions only."""
        if any(not isinstance(d, int) for d in self._dims):
            raise ValueError(f"Shape {self!r} contains dynamic dimensions.")
        return tuple(self._dims)  # type: ignore[arg-type]

    def freeze(self) -> None:
        """Kept for API compatibility, shapes are not frozen in this implementation."""

    def __len__(self) -> int:
        return len(self._dims)

    def __getitem__(self, index):
        return self._dims[index]

    def __setitem__(self, index, value) -> None:
        self._dims[index] = self._normalize(value)

    def __iter__(self) -> Iterator[Union[int, str, None]]:
        return iter(self._dims)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Shape):
            return self._dims == other._dims
        if isinstance(other, Sequence):
            return self._dims == [self._normalize(d) for d in other]
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self._dims))

    def __repr__(self) -> str:
        return f"Shape({self._dims!r})"

    def __str__(self) -> str:
        return "[" + ",".join("?" if d is None else str(d) for d in self._dims) + "]"


class TensorType:
    """Type of a tensor value: an element type and (optionally) a shape."""

    __slots__ = ("dtype", "denotation")

    def __init__(self, dtype: Union[DataType, int], denotation: Optional[str] = None):
        self.dtype = DataType(dtype)
        self.denotation = denotation

    @property
    def elem_type(self) -> DataType:
        "Element type of the tensor."
        return self.dtype

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorType):
            return NotImplemented
        return self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash(self.dtype)

    def __repr__(self) -> str:
        return f"TensorType({self.dtype!r})"


@runtime_checkable
class TensorProtocol(Protocol):
    """Interface implemented by all tensor classes of this module."""

    name: Optional[str]

    @property
    def dtype(self) -> DataType: ...

    @property
    def shape(self) -> Shape: ...

    def numpy(self) -> np.ndarray: ...

    def tobytes(self) -> bytes: ...


def _pack_4bit(array: np.ndarray) -> np.ndarray:
    """Packs an unpacked 4-bit array into ``uint8`` pairs (little-endian nibbles)."""
    flat = array.reshape(-1).view(np.uint8)
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    return ((flat[1::2] & 0x0F) << 4) | (flat[0::2] & 0x0F)


def _unpack_4bit(packed: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Unpacks ``uint8`` pairs into one byte per 4-bit element."""
    packed = packed.reshape(-1).view(np.uint8)
    low = packed & 0x0F
    high = packed >> 4
    unpacked = np.empty(packed.size * 2, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high
    size = int(np.prod(shape)) if len(shape) else 1
    return unpacked[:size].reshape(shape)


class TensorBase:
    """Common implementation shared by all tensor classes."""

    def __init__(self, name: Optional[str] = None, doc_string: Optional[str] = None, metadata_props: Optional[Dict[str, str]] = None):
        self.name = name
        self.doc_string = doc_string
        self.metadata_props = metadata_props or {}

    @property
    def dtype(self) -> DataType:  # pragma: no cover - implemented by subclasses
        "Element type."
        raise NotImplementedError

    @property
    def shape(self) -> Shape:  # pragma: no cover - implemented by subclasses
        "Shape of the tensor."
        raise NotImplementedError

    @property
    def size(self) -> int:
        "Number of elements."
        dims = self.shape.numpy()
        size = 1
        for d in dims:
            size *= d
        return size

    @property
    def nbytes(self) -> int:
        "Number of bytes used by the serialized tensor."
        bitwidth = self.dtype.bitwidth
        return (self.size * bitwidth + 7) // 8

    def numpy(self) -> np.ndarray:  # pragma: no cover - implemented by subclasses
        "Returns the tensor as a numpy array."
        raise NotImplementedError

    def tobytes(self) -> bytes:  # pragma: no cover - implemented by subclasses
        "Returns the raw ONNX representation of the tensor."
        raise NotImplementedError

    def _repr_base(self) -> str:
        return f"{self.__class__.__name__}<{self.dtype!s},{self.shape!s}>"

    def __repr__(self) -> str:
        return f"{self._repr_base()}(name={self.name!r})"


class Tensor(TensorBase):
    """Tensor backed by a numpy (or numpy compatible) array."""

    def __init__(
        self,
        value: Any,
        dtype: Optional[DataType] = None,
        *,
        shape: Union[Shape, Sequence[int], None] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name=name, doc_string=doc_string, metadata_props=metadata_props)
        self._raw = value
        if dtype is None:
            self._dtype = DataType.from_numpy(np.asarray(value).dtype)
        else:
            self._dtype = DataType(dtype)
        self._shape = Shape(shape) if shape is not None else Shape(tuple(value.shape))

    @property
    def raw(self) -> Any:
        "Backing data of the tensor."
        return self._raw

    @property
    def dtype(self) -> DataType:
        "Element type."
        return self._dtype

    @property
    def shape(self) -> Shape:
        "Shape of the tensor."
        return self._shape

    def numpy(self) -> np.ndarray:
        "Returns the tensor as a numpy array."
        return np.asarray(self._raw)

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        array = self.numpy()
        return array if dtype is None else array.__array__(dtype)

    def tobytes(self) -> bytes:
        "Returns the raw ONNX representation of the tensor."
        array = np.ascontiguousarray(self.numpy())
        if self._dtype in _SUB_BYTE_TYPES and array.dtype.itemsize == 1:
            array = _pack_4bit(array)
        return array.tobytes()


class LazyTensor(TensorBase):
    """Tensor whose data is computed the first time it is needed."""

    def __init__(
        self,
        func: Callable[[], Any],
        dtype: DataType,
        shape: Union[Shape, Sequence[int]],
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name=name, doc_string=doc_string, metadata_props=metadata_props)
        self._func = func
        self._dtype = DataType(dtype)
        self._shape = Shape(shape)
        self._tensor: Optional[TensorBase] = None

    @property
    def dtype(self) -> DataType:
        "Element type."
        return self._dtype

    @property
    def shape(self) -> Shape:
        "Shape of the tensor."
        return self._shape

    @property
    def raw(self) -> Any:
        "Materializes the tensor and returns its backing data."
        return self._materialize().raw  # type: ignore[attr-defined]

    def _materialize(self) -> TensorBase:
        if self._tensor is None:
            tensor = self._func()
            if not isinstance(tensor, TensorBase):
                tensor = Tensor(tensor, name=self.name)
            self._tensor = tensor
        return self._tensor

    def numpy(self) -> np.ndarray:
        "Returns the tensor as a numpy array."
        return self._materialize().numpy()

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        array = self.numpy()
        return array if dtype is None else array.__array__(dtype)

    def tobytes(self) -> bytes:
        "Returns the raw ONNX representation of the tensor."
        return self._materialize().tobytes()


class PackedTensor(TensorBase):
    """Tensor holding 2-bit or 4-bit data already packed into bytes."""

    def __init__(
        self,
        value: Any,
        dtype: DataType,
        *,
        shape: Union[Shape, Sequence[int]],
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name=name, doc_string=doc_string, metadata_props=metadata_props)
        self._dtype = DataType(dtype)
        if self._dtype.bitwidth not in (2, 4):
            raise TypeError(f"PackedTensor only supports 2-bit and 4-bit data types, not {self._dtype}.")
        self._shape = Shape(shape)
        self._raw = value

    @property
    def raw(self) -> Any:
        "Backing (packed) data of the tensor."
        return self._raw

    @property
    def dtype(self) -> DataType:
        "Element type."
        return self._dtype

    @property
    def shape(self) -> Shape:
        "Shape of the tensor (in number of elements, not bytes)."
        return self._shape

    def numpy_packed(self) -> np.ndarray:
        "Returns the packed representation as an array of bytes."
        array = np.ascontiguousarray(np.asarray(self._raw))
        return array.view(np.uint8)

    def numpy(self) -> np.ndarray:
        "Returns the unpacked tensor as a numpy array."
        unpacked = _unpack_4bit(self.numpy_packed(), self._shape.numpy())
        return unpacked.view(self._dtype.numpy())

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        array = self.numpy()
        return array if dtype is None else array.__array__(dtype)

    def tobytes(self) -> bytes:
        "Returns the raw ONNX representation of the tensor."
        return self.numpy_packed().tobytes()


class ExternalTensor(TensorBase):
    """Tensor stored in an external file, loaded lazily with a memory map."""

    def __init__(
        self,
        location: str,
        offset: int,
        length: int,
        dtype: DataType,
        shape: Union[Shape, Sequence[int]],
        name: Optional[str] = None,
        base_dir: str = "",
        doc_string: Optional[str] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name=name, doc_string=doc_string, metadata_props=metadata_props)
        self.location = location
        self.offset = offset
        self.length = length
        self.base_dir = base_dir
        self._dtype = DataType(dtype)
        self._shape = Shape(shape)

    @property
    def path(self) -> str:
        "Full path of the external data file."
        import os

        return os.path.join(self.base_dir, self.location)

    @property
    def dtype(self) -> DataType:
        "Element type."
        return self._dtype

    @property
    def shape(self) -> Shape:
        "Shape of the tensor."
        return self._shape

    def tobytes(self) -> bytes:
        "Returns the raw ONNX representation of the tensor."
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            return f.read(self.length)

    def numpy(self) -> np.ndarray:
        "Returns the tensor as a numpy array."
        data = self.tobytes()
        if self._dtype in _SUB_BYTE_TYPES:
            return _unpack_4bit(np.frombuffer(data, dtype=np.uint8), self._shape.numpy()).view(self._dtype.numpy())
        return np.frombuffer(data, dtype=self._dtype.numpy()).reshape(self._shape.numpy())

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        array = self.numpy()
        return array if dtype is None else array.__array__(dtype)


class StringTensor(TensorBase):
    """Tensor of UTF-8 strings."""

    def __init__(
        self,
        value: Sequence[bytes],
        *,
        shape: Union[Shape, Sequence[int], None] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name=name, doc_string=doc_string, metadata_props=metadata_props)
        self._raw = [v if isinstance(v, bytes) else str(v).encode("utf-8") for v in value]
        self._shape = Shape(shape) if shape is not None else Shape((len(self._raw),))

    @property
    def raw(self) -> List[bytes]:
        "Backing data of the tensor."
        return self._raw

    @property
    def dtype(self) -> DataType:
        "Element type."
        return DataType.STRING

    @property
    def shape(self) -> Shape:
        "Shape of the tensor."
        return self._shape

    def numpy(self) -> np.ndarray:
        "Returns the tensor as a numpy array of objects."
        return np.array(self._raw, dtype=object).reshape(self._shape.numpy())

    def tobytes(self) -> bytes:  # pragma: no cover - strings are not stored as raw data
        "String tensors have no raw representation."
        raise TypeError("String tensors cannot be serialized as raw data.")


def tensor(value: Any, dtype: Optional[DataType] = None, name: Optional[str] = None, doc_string: Optional[str] = None) -> TensorBase:
    """Creates a tensor from an array, a torch tensor or a plain Python value.

    Args:
        value: numpy array, torch tensor, or Python scalar/sequence.
        dtype: data type, mandatory when ``value`` is a plain Python object
            without an unambiguous type.
        name: name of the tensor.
        doc_string: documentation string.

    Returns:
        A tensor implementing :class:`TensorProtocol`.
    """
    if isinstance(value, TensorBase):
        if dtype is not None and DataType(dtype) != value.dtype:
            raise ValueError(f"dtype {dtype} does not match the tensor dtype {value.dtype}.")
        if name is not None:
            value.name = name
        return value
    if _is_torch_tensor(value):
        from .tensor_adapters import TorchTensor

        return TorchTensor(value, name=name, doc_string=doc_string)
    if isinstance(value, np.ndarray):
        if dtype is not None and DataType(dtype) != DataType.from_numpy(value.dtype):
            value = value.astype(DataType(dtype).numpy())
        return Tensor(value, dtype=dtype, name=name, doc_string=doc_string)

    if dtype is not None:
        numpy_dtype = DataType(dtype).numpy()
    elif isinstance(value, bool):
        numpy_dtype = np.dtype(np.bool_)
    elif isinstance(value, int):
        numpy_dtype = np.dtype(np.int64)
    elif isinstance(value, float):
        numpy_dtype = np.dtype(np.float32)
    elif isinstance(value, Sequence) and value:
        if all(isinstance(v, bool) for v in value):
            numpy_dtype = np.dtype(np.bool_)
        elif all(isinstance(v, int) for v in value):
            numpy_dtype = np.dtype(np.int64)
        else:
            numpy_dtype = np.dtype(np.float32)
    else:
        raise ValueError(f"dtype must be specified for value {value!r}.")

    array = np.array(value, dtype=numpy_dtype)
    return Tensor(array, dtype=dtype, name=name, doc_string=doc_string)


class Attr:
    """An ONNX attribute (name, type and value)."""

    __slots__ = ("name", "type", "value", "doc_string", "ref_attr_name")

    def __init__(
        self,
        name: str,
        type: AttributeType,  # noqa: A002
        value: Any,
        doc_string: Optional[str] = None,
        ref_attr_name: Optional[str] = None,
    ):
        self.name = name
        self.type = AttributeType(type)
        self.value = value
        self.doc_string = doc_string
        self.ref_attr_name = ref_attr_name

    def is_ref(self) -> bool:
        "Returns ``True`` for a reference attribute (used inside a function body)."
        return self.ref_attr_name is not None

    def as_int(self) -> int:
        "Returns the attribute value as an integer."
        return int(self.value)

    def as_float(self) -> float:
        "Returns the attribute value as a float."
        return float(self.value)

    def as_string(self) -> str:
        "Returns the attribute value as a string."
        return self.value if isinstance(self.value, str) else self.value.decode("utf-8")

    def as_tensor(self) -> TensorBase:
        "Returns the attribute value as a tensor."
        return self.value

    def as_graph(self) -> "Graph":
        "Returns the attribute value as a graph."
        return self.value

    def as_ints(self) -> List[int]:
        "Returns the attribute value as a list of integers."
        return list(self.value)

    def as_floats(self) -> List[float]:
        "Returns the attribute value as a list of floats."
        return list(self.value)

    def as_strings(self) -> List[str]:
        "Returns the attribute value as a list of strings."
        return [v if isinstance(v, str) else v.decode("utf-8") for v in self.value]

    def as_tensors(self) -> List[TensorBase]:
        "Returns the attribute value as a list of tensors."
        return list(self.value)

    def as_graphs(self) -> List["Graph"]:
        "Returns the attribute value as a list of graphs."
        return list(self.value)

    def __repr__(self) -> str:
        return f"Attr({self.name!r}, {self.type!r}, {self.value!r})"


def _convert_attribute(name: str, value: Any) -> Attr:
    """Converts a Python value into an :class:`Attr`."""
    if isinstance(value, Attr):
        if value.name != name:
            return Attr(name, value.type, value.value, value.doc_string, value.ref_attr_name)
        return value
    if isinstance(value, (bool, np.bool_)):
        return Attr(name, AttributeType.INT, int(value))
    if isinstance(value, (int, np.integer)):
        return Attr(name, AttributeType.INT, int(value))
    if isinstance(value, (float, np.floating)):
        return Attr(name, AttributeType.FLOAT, float(value))
    if isinstance(value, str):
        return Attr(name, AttributeType.STRING, value)
    if isinstance(value, bytes):
        return Attr(name, AttributeType.STRING, value.decode("utf-8"))
    if isinstance(value, TensorBase):
        return Attr(name, AttributeType.TENSOR, value)
    if isinstance(value, Graph):
        return Attr(name, AttributeType.GRAPH, value)
    if isinstance(value, TensorType):
        return Attr(name, AttributeType.TYPE_PROTO, value)
    if isinstance(value, np.ndarray):
        return Attr(name, AttributeType.TENSOR, tensor(value, name=name))
    if isinstance(value, Sequence):
        values = list(value)
        if not values:
            return Attr(name, AttributeType.INTS, [])
        if all(isinstance(v, Graph) for v in values):
            return Attr(name, AttributeType.GRAPHS, values)
        if all(isinstance(v, TensorBase) for v in values):
            return Attr(name, AttributeType.TENSORS, values)
        if all(isinstance(v, str) for v in values):
            return Attr(name, AttributeType.STRINGS, values)
        if all(isinstance(v, (bool, np.bool_)) for v in values):
            return Attr(name, AttributeType.INTS, [int(v) for v in values])
        if all(isinstance(v, (int, np.integer)) for v in values):
            return Attr(name, AttributeType.INTS, [int(v) for v in values])
        if all(isinstance(v, (float, int, np.floating, np.integer)) for v in values):
            return Attr(name, AttributeType.FLOATS, [float(v) for v in values])
    raise TypeError(f"Unsupported attribute type {type(value)} for attribute {name!r}.")


def _convert_attributes(attributes: Union[Mapping[str, Any], Sequence[Attr], None]) -> Dict[str, Attr]:
    """Converts a mapping or a sequence of attributes into a dictionary of :class:`Attr`."""
    converted: Dict[str, Attr] = {}
    if attributes is None:
        return converted
    if isinstance(attributes, Mapping):
        for name, value in attributes.items():
            # An attribute set to None is not defined and is skipped.
            if value is None:
                continue
            converted[name] = _convert_attribute(name, value)
        return converted
    for attr in attributes:
        if not isinstance(attr, Attr):
            raise TypeError(f"Expecting an Attr instance not {type(attr)}.")
        converted[attr.name] = attr
    return converted


class Value:
    """A value in the graph: a graph input/output, an initializer or a node output."""

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        type: Optional[TensorType] = None,  # noqa: A002
        shape: Union[Shape, Sequence[Any], None] = None,
        const_value: Optional[TensorBase] = None,
        doc_string: Optional[str] = None,
        metadata_props: Optional[Dict[str, str]] = None,
        producer: Optional["Node"] = None,
        index: Optional[int] = None,
    ):
        self.name = name
        self.type = type
        self._shape = Shape(shape) if shape is not None else None
        self.const_value = const_value
        self.doc_string = doc_string
        self.metadata_props = metadata_props or {}
        self.meta: Dict[str, Any] = {}
        self._producer = producer
        self._index = index
        self._uses: List[Tuple["Node", int]] = []
        self._is_graph_input = False
        self._is_graph_output = False
        self._is_initializer = False

    @property
    def shape(self) -> Optional[Shape]:
        "Shape of the value, ``None`` when unknown."
        return self._shape

    @shape.setter
    def shape(self, value: Union[Shape, Sequence[Any], None]) -> None:
        self._shape = None if value is None else Shape(value)

    @property
    def dtype(self) -> Optional[DataType]:
        "Element type of the value, ``None`` when unknown."
        return None if self.type is None else self.type.dtype

    @dtype.setter
    def dtype(self, value: Union[DataType, int, None]) -> None:
        if value is None:
            self.type = None
        elif self.type is None:
            self.type = TensorType(value)
        else:
            self.type.dtype = DataType(value)

    def producer(self) -> Optional["Node"]:
        "Returns the node producing this value, if any."
        return self._producer

    def index(self) -> Optional[int]:
        "Returns the output index in the producing node."
        return self._index

    def uses(self) -> List[Tuple["Node", int]]:
        "Returns the (node, input index) pairs consuming this value."
        return list(self._uses)

    def consumers(self) -> List["Node"]:
        "Returns the nodes consuming this value."
        return [node for node, _ in self._uses]

    def is_graph_input(self) -> bool:
        "Returns ``True`` when the value is a graph input."
        return self._is_graph_input

    def is_graph_output(self) -> bool:
        "Returns ``True`` when the value is a graph output."
        return self._is_graph_output

    def is_initializer(self) -> bool:
        "Returns ``True`` when the value is an initializer."
        return self._is_initializer

    def __repr__(self) -> str:
        return f"Value(name={self.name!r}, type={self.type!r}, shape={self._shape!r})"


class Node:
    """A node in an ONNX graph."""

    def __init__(
        self,
        domain: str = "",
        op_type: str = "",
        inputs: Sequence[Optional[Value]] = (),
        attributes: Union[Mapping[str, Any], Sequence[Attr], None] = None,
        *,
        overload: str = "",
        num_outputs: Optional[int] = None,
        outputs: Optional[Sequence[Value]] = None,
        version: Optional[int] = None,
        graph: Optional["Graph"] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        self.domain = domain
        self.op_type = op_type
        self.overload = overload
        self.name = name
        self.version = version
        self.doc_string = doc_string
        self.metadata_props = metadata_props or {}
        self.meta: Dict[str, Any] = {}
        self.attributes: Dict[str, Attr] = _convert_attributes(attributes)
        self.graph = graph

        self.inputs: List[Optional[Value]] = list(inputs)
        for index, value in enumerate(self.inputs):
            if value is not None:
                value._uses.append((self, index))

        if outputs is None:
            n_outputs = 1 if num_outputs is None else num_outputs
            self.outputs: List[Value] = [Value(name=None) for _ in range(n_outputs)]
        else:
            self.outputs = list(outputs)
        for index, value in enumerate(self.outputs):
            value._producer = self
            value._index = index

    def replace_input_with(self, index: int, value: Optional[Value]) -> None:
        "Replaces one input of the node."
        old = self.inputs[index]
        if old is not None:
            old._uses = [use for use in old._uses if use != (self, index)]
        self.inputs[index] = value
        if value is not None:
            value._uses.append((self, index))

    def subgraphs(self) -> Iterator["Graph"]:
        "Iterates over the subgraphs held by the node attributes."
        for attr in self.attributes.values():
            if attr.type == AttributeType.GRAPH:
                yield attr.value
            elif attr.type == AttributeType.GRAPHS:
                yield from attr.value

    def __repr__(self) -> str:
        return f"Node(name={self.name!r}, op_type={self.op_type!r}, domain={self.domain!r})"


def node(
    op_type: str,
    inputs: Sequence[Optional[Value]] = (),
    attributes: Union[Mapping[str, Any], Sequence[Attr], None] = None,
    *,
    domain: str = "",
    overload: str = "",
    num_outputs: Optional[int] = None,
    outputs: Optional[Sequence[Value]] = None,
    version: Optional[int] = None,
    graph: Optional["Graph"] = None,
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    metadata_props: Optional[Dict[str, str]] = None,
) -> Node:
    """Creates a :class:`Node`, converting the attributes from Python values."""
    return Node(
        domain,
        op_type,
        inputs,
        attributes,
        overload=overload,
        num_outputs=num_outputs,
        outputs=outputs,
        version=version,
        graph=graph,
        name=name,
        doc_string=doc_string,
        metadata_props=metadata_props,
    )


class _GraphInitializers(Dict[str, Value]):
    """Dictionary of initializers keeping the ``is_initializer`` flag up to date."""

    def __setitem__(self, key: str, value: Value) -> None:
        value._is_initializer = True
        super().__setitem__(key, value)


class _GraphIOList(List[Value]):
    """List of graph inputs or outputs tracking the corresponding value flag."""

    def __init__(self, values: Sequence[Value] = (), *, attribute: str = "_is_graph_input"):
        super().__init__()
        self._attribute = attribute
        self.extend(values)

    def append(self, value: Value) -> None:
        "Appends a value and flags it."
        setattr(value, self._attribute, True)
        super().append(value)

    def extend(self, values) -> None:
        "Appends several values and flags them."
        for value in values:
            self.append(value)

    def insert(self, index: int, value: Value) -> None:
        "Inserts a value and flags it."
        setattr(value, self._attribute, True)
        super().insert(index, value)


class Graph:
    """An ONNX graph: inputs, outputs, initializers and a list of nodes."""

    def __init__(
        self,
        inputs: Sequence[Value] = (),
        outputs: Sequence[Value] = (),
        *,
        nodes: Sequence[Node] = (),
        initializers: Sequence[Value] = (),
        doc_string: Optional[str] = None,
        opset_imports: Optional[Mapping[str, int]] = None,
        name: Optional[str] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.doc_string = doc_string
        self.metadata_props = metadata_props or {}
        self.meta: Dict[str, Any] = {}
        self.opset_imports: Dict[str, int] = dict(opset_imports or {})
        self.inputs = _GraphIOList(inputs, attribute="_is_graph_input")
        self.outputs = _GraphIOList(outputs, attribute="_is_graph_output")
        self.initializers = _GraphInitializers()
        for value in initializers:
            self.register_initializer(value)
        self._nodes: List[Node] = []
        self.extend(nodes)

    @property
    def nodes(self) -> List[Node]:
        "Returns the list of nodes."
        return self._nodes

    def append(self, node_: Node) -> None:
        "Appends a node at the end of the graph."
        node_.graph = self
        self._nodes.append(node_)

    def extend(self, nodes: Sequence[Node]) -> None:
        "Appends several nodes at the end of the graph."
        for node_ in nodes:
            self.append(node_)

    def insert_before(self, reference: Node, new_node: Node) -> None:
        "Inserts a node before another one."
        new_node.graph = self
        self._nodes.insert(self._nodes.index(reference), new_node)

    def insert_after(self, reference: Node, new_node: Node) -> None:
        "Inserts a node after another one."
        new_node.graph = self
        self._nodes.insert(self._nodes.index(reference) + 1, new_node)

    def remove(self, nodes: Union[Node, Sequence[Node]], safe: bool = False) -> None:
        "Removes one or several nodes from the graph."
        del safe  # kept for API compatibility
        if isinstance(nodes, Node):
            nodes = [nodes]
        for node_ in nodes:
            self._nodes.remove(node_)
            node_.graph = None

    def register_initializer(self, value: Value) -> None:
        "Registers a value as an initializer. The value must have a name."
        if not value.name:
            raise ValueError("An initializer must have a name.")
        if value.name in self.initializers and self.initializers[value.name] is not value:
            raise ValueError(f"An initializer named {value.name!r} is already registered.")
        self.initializers[value.name] = value

    def all_nodes(self) -> Iterator[Node]:
        "Iterates over every node of the graph including the ones in subgraphs."
        for node_ in self._nodes:
            yield node_
            for subgraph in node_.subgraphs():
                yield from subgraph.all_nodes()

    def subgraphs(self) -> Iterator["Graph"]:
        "Iterates over every subgraph, recursively."
        for node_ in self._nodes:
            for subgraph in node_.subgraphs():
                yield subgraph
                yield from subgraph.subgraphs()

    def graphs(self) -> Iterator["Graph"]:
        "Iterates over this graph and every subgraph, recursively."
        yield self
        yield from self.subgraphs()

    def sort(self) -> None:
        """Sorts the nodes in topological order.

        Nodes producing a value are moved before the nodes consuming it.  Values
        consumed by a subgraph (outer scope values) are taken into account.  The
        subgraphs are sorted recursively.
        """
        for subgraph in self.subgraphs():
            subgraph.sort()

        node_index = {id(node_): i for i, node_ in enumerate(self._nodes)}

        def dependencies(node_: Node) -> List[Node]:
            deps = []
            for value in _node_used_values(node_):
                producer = value.producer()
                if producer is not None and id(producer) in node_index:
                    deps.append(producer)
            return deps

        sorted_nodes: List[Node] = []
        # 0: not visited, 1: being visited, 2: done
        state: Dict[int, int] = {}
        for start in self._nodes:
            if state.get(id(start), 0) == 2:
                continue
            stack: List[Tuple[Node, bool]] = [(start, False)]
            while stack:
                current, processed = stack.pop()
                key = id(current)
                if processed:
                    if state.get(key) != 2:
                        state[key] = 2
                        sorted_nodes.append(current)
                    continue
                current_state = state.get(key, 0)
                if current_state == 2:
                    continue
                if current_state == 1:
                    # Cycle: keep the original order for this node.
                    continue
                state[key] = 1
                stack.append((current, True))
                for dep in reversed(dependencies(current)):
                    if state.get(id(dep), 0) == 0:
                        stack.append((dep, False))
        self._nodes = sorted_nodes

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"Graph(name={self.name!r}, nodes={len(self._nodes)})"


def _node_used_values(node_: Node) -> Iterator[Value]:
    """Yields every value used by a node, including the ones used by its subgraphs."""
    for value in node_.inputs:
        if value is not None:
            yield value
    for subgraph in node_.subgraphs():
        local = {id(v) for v in subgraph.inputs}
        local.update(id(v) for v in subgraph.initializers.values())
        for sub_node in subgraph.all_nodes():
            for value in sub_node.outputs:
                local.add(id(value))
        for sub_node in subgraph.all_nodes():
            for value in sub_node.inputs:
                if value is not None and id(value) not in local:
                    yield value


class Function:
    """An ONNX local function."""

    def __init__(
        self,
        domain: str,
        name: str,
        overload: str = "",
        *,
        graph: Graph,
        attributes: Union[Sequence[str], Mapping[str, Any], None] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        self.domain = domain
        self.name = name
        self.overload = overload
        self._graph = graph
        self.attributes = dict(attributes) if isinstance(attributes, Mapping) else {name_: None for name_ in (attributes or ())}
        self.metadata_props = metadata_props or {}
        self.meta: Dict[str, Any] = {}

    def identifier(self) -> Tuple[str, str, str]:
        "Returns the ``(domain, name, overload)`` key identifying the function."
        return (self.domain, self.name, self.overload)

    @property
    def graph(self) -> Graph:
        "Body of the function."
        return self._graph

    @property
    def inputs(self) -> List[Value]:
        "Formal inputs."
        return self._graph.inputs

    @property
    def outputs(self) -> List[Value]:
        "Formal outputs."
        return self._graph.outputs

    @property
    def opset_imports(self) -> Dict[str, int]:
        "Opsets used by the function body."
        return self._graph.opset_imports

    @property
    def doc_string(self) -> Optional[str]:
        "Documentation string."
        return self._graph.doc_string

    def __iter__(self) -> Iterator[Node]:
        return iter(self._graph)

    def __len__(self) -> int:
        return len(self._graph)

    def __repr__(self) -> str:
        return f"Function(domain={self.domain!r}, name={self.name!r}, overload={self.overload!r})"


class Model:
    """An ONNX model: a main graph, its opsets and the local functions."""

    def __init__(
        self,
        graph: Graph,
        *,
        ir_version: int = 10,
        producer_name: Optional[str] = None,
        producer_version: Optional[str] = None,
        domain: Optional[str] = None,
        model_version: Optional[int] = None,
        doc_string: Optional[str] = None,
        functions: Union[Sequence[Function], Mapping[Tuple[str, str, str], Function], None] = None,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        self.graph = graph
        self.ir_version = ir_version
        self.producer_name = producer_name
        self.producer_version = producer_version
        self.domain = domain
        self.model_version = model_version
        self.doc_string = doc_string
        self.metadata_props = metadata_props or {}
        self.meta: Dict[str, Any] = {}
        self.functions: Dict[Tuple[str, str, str], Function] = {}
        if functions:
            if isinstance(functions, Mapping):
                self.functions.update(functions)
            else:
                for func in functions:
                    self.functions[func.identifier()] = func

    @property
    def opset_imports(self) -> Dict[str, int]:
        "Opsets used by the main graph."
        return self.graph.opset_imports

    def graphs(self) -> Iterator[Graph]:
        "Iterates over the main graph, its subgraphs and the function bodies."
        yield from self.graph.graphs()
        for func in self.functions.values():
            yield from func.graph.graphs()

    def __repr__(self) -> str:
        return f"Model(ir_version={self.ir_version!r}, graph={self.graph!r})"
