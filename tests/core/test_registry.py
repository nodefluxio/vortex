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
    def __init__(self):
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

def test_registry():
    reg = Registry("testing")
    # can mix type if desired
    reg.register_module(AnotherType)
    reg.register_module(DummyBase)
    assert len(reg) == 2
