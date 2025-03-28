import numpy as np
import qpsolvers

def _compute_qp_objective(
        env, tasks, damping: float
) -> Objective:
    H = np.eye(env.model.nv) * damping
    c = np.zeros(env.model.nv)
    for task in tasks:
        H_task, c_task = task.compute_qp_objective