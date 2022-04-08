from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    List,
    Mapping,
    NewType,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from gpflow.experimental.check_shapes import ShapeChecker
from gpflow.experimental.check_shapes.error_contexts import (
    AttributeContext,
    ObjectTypeContext,
    StackContext,
)

T = TypeVar("T")
_Sentinel = NewType("_Sentinel", object)
_SENTINEL = _Sentinel(object())
_EMPTY_SET: Set["V[Any]"] = set()

VM = TypeVar("VM", bound="VModule")


class VModule:
    def __init__(self: VM, *parents: VM, **assignments: Any) -> None:
        self._values: Dict["V[Any]", Any] = {}
        self._dependencies: Dict["V[Any]", Set["V[Any]"]] = {}
        self._shape_checker = ShapeChecker()
        self._current: Optional["V[Any]"] = None

        if parents:
            assert len(parents) == 1, "Only a single parent supported."
            (parent,) = parents
            self._values.update(parent._values)
            for key, dependencies in parent._dependencies.items():
                self._dependencies[key] = set(dependencies)

        cls = self.__class__
        new_values = {getattr(cls, name): value for name, value in assignments.items()}
        assert all(isinstance(key, Input) for key in new_values)

        clear_queue = {k for k in new_values if k in self._values}
        while clear_queue:
            i = clear_queue.pop()
            for dependency in self._dependencies.get(i, _EMPTY_SET):
                if (dependency in self._values) and (dependency not in clear_queue):
                    clear_queue.add(dependency)
            del self._values[i]
            del self._dependencies[i]

        for key, value in new_values.items():
            self._values[key] = value
            self._dependencies[key] = set()

        shape_checks = []
        for key, value in self._values.items():
            shape = key._shape
            if shape:
                assert key._name
                shape_checks.append(
                    (
                        value,
                        shape,
                        StackContext(ObjectTypeContext(self), AttributeContext(key._name)),
                    )
                )
        self._shape_checker.check_shapes(shape_checks)

        assert set(self._values) == set(self._dependencies)

    def __call__(self: VM, **assignments: Any) -> VM:
        cls = self.__class__
        return cls(self, **assignments)


VT = TypeVar("VT", bound="V[Any]")


class V(Generic[T], ABC):
    def __init__(self, shape: Optional[str]) -> None:
        self._name: Optional[str] = None
        self._shape = shape

    @overload
    def __get__(self: VT, instance: None, owner: Type[VModule]) -> VT:
        ...

    @overload
    def __get__(self, instance: VModule, owner: Type[VModule]) -> T:
        ...

    def __get__(self: VT, instance: Optional[VModule], owner: Type[VModule]) -> Union[T, VT]:
        if instance is None:
            return self

        # pylint: disable=protected-access

        value = instance._values.get(self, _SENTINEL)
        if value is _SENTINEL:
            value = self._compute(instance)
            instance._values[self] = value
            instance._dependencies[self] = set()

            if self._shape:
                assert self._name
                instance._shape_checker.check_shape(
                    value,
                    self._shape,
                    StackContext(ObjectTypeContext(instance), AttributeContext(self._name)),
                )

            assert set(instance._values) == set(instance._dependencies)

        if instance._current is not None:
            instance._dependencies[self].add(instance._current)

        return cast(T, value)

    def __set_name__(self, owner: Type[VModule], name: str) -> None:
        self._name = name

    @abstractmethod
    def _compute(self, instance: VModule) -> T:
        pass

    def __repr__(self) -> str:
        token_list = []
        if self._name is not None:
            token_list.append(f"name={self._name}")
        if self._shape is not None:
            token_list.append(f"shape={self._shape}")
        tokens = ",".join(token_list)
        return f"{self.__class__.__name__}[{tokens}]"


class Input(V[T], Generic[T]):
    def __init__(
        self,
        value_type: Optional[Type[T]] = None,
        *,
        default: Union[T, _Sentinel] = _SENTINEL,
        shape: Optional[str] = None,
    ) -> None:
        super().__init__(shape)
        self._default = default

    def _compute(self, instance: VModule) -> T:
        assert self._default is not _SENTINEL
        return cast(T, self._default)


class Derived(V[T], Generic[T]):
    def __init__(self, factory: Callable[[Any], T], *, shape: Optional[str] = None) -> None:
        super().__init__(shape)
        self._factory = factory

    def _compute(self, instance: VModule) -> T:
        # pylint: disable=protected-access
        prev = instance._current
        instance._current = self
        try:
            return self._factory(instance)
        finally:
            instance._current = prev


def derived(*, shape: Optional[str] = None) -> Callable[[Callable[[Any], T]], Derived[T]]:
    def wrap(func: Callable[[Any], T]) -> Derived[T]:
        return Derived(func, shape=shape)

    return wrap


def multi_get(instance: VModule, names: Collection[str], prefix: str = "") -> Mapping[str, Any]:
    result = {}
    by_head: Dict[str, List[str]] = {}
    for name in names:
        head, *tails = name.split(".", 1)
        if tails:
            (tail,) = tails
            by_head.setdefault(head, []).append(tail)
        else:
            result[prefix + head] = getattr(instance, head)

    for head, tails in by_head.items():
        next_prefix = prefix + head + "."
        next_instance = getattr(instance, head)
        result.update(multi_get(next_instance, tails, next_prefix))

    return result


def multi_set(instance: VM, values: Mapping[str, Any]) -> VM:
    kwargs = {}
    by_head: Dict[str, Dict[str, Any]] = {}
    for name, value in values.items():
        head, *tails = name.split(".", 1)
        if tails:
            (tail,) = tails
            by_head.setdefault(head, {})[tail] = value
        else:
            kwargs[head] = value

    for head, tail_dict in by_head.items():
        next_instance = getattr(instance, head)
        kwargs[head] = multi_set(next_instance, tail_dict)

    return cast(VM, instance(**kwargs))


def to_loss_function(
    instance: VM, trainable_names: Collection[str], loss_name: str
) -> Tuple[Callable[[Mapping[str, Any]], Any], Mapping[str, Any]]:
    def loss(values: Mapping[str, Any]) -> Any:
        next_instance = multi_set(instance, values)
        return multi_get(next_instance, [loss_name])[loss_name]

    return loss, multi_get(instance, trainable_names)
