import pytest
from vortex.development.core.registry import Registry

class DummyBase:
    def __init__(self):
        pass

# create registry instance with DummyBase as base class req
glob_registry = Registry("testing",base_class=DummyBase)

# name derived from class name
@glob_registry.register()
class DummyOne(DummyBase):
    def __init__(self):
        pass

# custom name
@glob_registry.register(name="Dummy2")
class DummyTwo(DummyBase):
    def __init__(self):
        pass

# overwrite
@glob_registry.register(name="DummyOne",force=True)
class DummyThree(DummyBase):
    def __init__(self):
        pass

class AnotherType:
    def __init__(self):
        pass

# register method is decorator factory
f = glob_registry.register()
g = glob_registry.register(name="AnotherDummy")

@f
class DummyFour(DummyBase):
    def __init__(self):
        pass

@g
class DummyFive(DummyBase):
    def __init__(self, str_value: str, int_value: int):
        self.value = str_value, int_value
        pass

def test_glob_registry():
    assert len(glob_registry) == 4
    assert "DummyOne" in glob_registry
    assert "Dummy2" in glob_registry
    assert "DummyFour" in glob_registry
    assert "DummyThree" not in glob_registry
    assert "AnotherDummy" in glob_registry
    repr(glob_registry)
    # actually glob already has DummyThree type, this provide alias
    glob_registry.register_module(DummyThree)
    assert len(glob_registry) == 5
    assert "DummyThree" in glob_registry

    # glob registry is strict
    with pytest.raises(TypeError):
        glob_registry.register_module(AnotherType)
    
    # should use force
    with pytest.raises(KeyError):
        glob_registry.register_module(DummyThree)
    
    # instance creation test

    dummy2 = glob_registry.create_from_args("Dummy2")
    assert isinstance(dummy2, DummyTwo)
    another_dummy = glob_registry.create_from_args("AnotherDummy", str_value="some string", int_value=0)
    assert isinstance(another_dummy, DummyFive)

    d = dict(str_value="some string", int_value=0)
    with pytest.raises(TypeError):
        dummy = glob_registry.create_from_args("AnotherDummy", d)
    
    # to construct from dict, use create_from_dict
    dummy = glob_registry.create_from_dict("AnotherDummy", d)
    assert isinstance(dummy, DummyFive)
    assert dummy.value == ("some string", 0)

    args = dict(
        module="AnotherDummy",
        str_value="some string",
        int_value=0
    )
    another_dummy = glob_registry.create_from_args(**args)
    assert isinstance(another_dummy, DummyFive)

def test_registry():
    reg = Registry("testing")
    # can mix type if desired
    reg.register_module(AnotherType)
    reg.register_module(DummyBase)
    assert len(reg) == 2
