from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .distill_solver import DistillSolver



from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'classification': ClasSolver,
    'detection': DetSolver,
    'distillation': DistillSolver,
}