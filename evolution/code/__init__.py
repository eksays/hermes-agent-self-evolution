"""Code evolution — evolve tool source code via DSPy GEPA."""
# NOTE: Do not re-export the ``evolve_code`` *function* here. Its name collides
# with the ``evolve_code`` submodule, and binding the function on the package
# shadows the submodule so that ``import evolution.code.evolve_code`` resolves
# to the function instead of the module. Import it via its full path:
#     from evolution.code.evolve_code import evolve_code
from evolution.code.code_module import CodeModule

__all__ = ["CodeModule"]
