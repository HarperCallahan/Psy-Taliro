import logging
import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

from staliro.core import best_eval, best_run
from staliro.models import State, ode
from staliro.optimizers import CoordinateDescent,CoordinateAscent,UniformRandom
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro


@ode()
def nonlinear_model(time: float, state: State, _: Any) -> State:
    x1_dot = state[0] - state[1] + 0.1 * time
    x2_dot = state[1] * math.cos(2 * math.pi * state[0]) + 0.1 * time

    return np.array([x1_dot, x2_dot])





if __name__ == "__main__":
    """for i in range(10):
        initial_conditions = [(-1, 1), (-1, 1)]
        phi = r"always !(a >= -1.6 and a <= -1.4  and b >= -1.1 and b <= -0.9)"
        specification = RTAMTDense(phi, {"a": 0, "b": 1})
        options = Options(runs=1, iterations=100, interval=(0, 2), static_parameters=initial_conditions)
        optimizer = [CoordinateDescent(), UniformRandom()]
        for k in range(2):
            logging.basicConfig(level=logging.DEBUG)
            
            
            result = staliro(nonlinear_model, specification, optimizer[k], options)

            best_run_ = best_run(result)
            best_sample = best_eval(best_run_).sample
            best_result = simulate_model(nonlinear_model, options, best_sample)

            sample_xs = [evaluation.sample[0] for evaluation in best_run_.history]
            sample_ys = [evaluation.sample[1] for evaluation in best_run_.history]

            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    name="Falsification area",
                    x=[-1.6, -1.4, -1.4, -1.6],
                    y=[-1.1, -1.1, -0.9, -0.9],
                    fill="toself",
                    fillcolor="red",
                    line_color="red",
                    mode="lines+markers",
                )
            )
            figure.add_trace(
                go.Scatter(
                    name="Initial condition region",
                    x=[-1, 1, 1, -1],
                    y=[-1, -1, 1, 1],
                    fill="toself",
                    fillcolor="green",
                    line_color="green",
                    mode="lines+markers",
                )
            )
            figure.add_trace(
                go.Scatter(
                    name="Samples",
                    x=sample_xs,
                    y=sample_ys,
                    mode="markers",
                    marker=go.scatter.Marker(color="lightblue", symbol="circle"),
                )
            )
            figure.add_trace(
                go.Scatter(
                    name="Best evaluation trajectory",
                    x=best_result.trace.states[0],
                    y=best_result.trace.states[1],
                    mode="lines+markers",
                    line=go.scatter.Line(color="blue", shape="spline"),
                )
            )
            opti = ["coordinatedescent","uniformrandom"]
            file = ("data/{type}/{type}{i}.jpeg").format(type = opti[k], i = i +1)
            figure.write_image(file)"""
    initial_conditions = [(-1, 1), (-1, 1)]
    phi = r"always !(a >= -1.6 and a <= -1.4  and b >= -1.1 and b <= -0.9)"
    specification = RTAMTDense(phi, {"a": 0, "b": 1})
    options = Options(runs=1, iterations=100, interval=(0, 2), static_parameters=initial_conditions)
    optimizer = CoordinateAscent()
    
    logging.basicConfig(level=logging.DEBUG)
    
    
    result = staliro(nonlinear_model, specification, optimizer, options)

    best_run_ = best_run(result)
    best_sample = best_eval(best_run_).sample
    best_result = simulate_model(nonlinear_model, options, best_sample)

    sample_xs = [evaluation.sample[0] for evaluation in best_run_.history]
    sample_ys = [evaluation.sample[1] for evaluation in best_run_.history]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            name="Falsification area",
            x=[-1.6, -1.4, -1.4, -1.6],
            y=[-1.1, -1.1, -0.9, -0.9],
            fill="toself",
            fillcolor="red",
            line_color="red",
            mode="lines+markers",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Initial condition region",
            x=[-1, 1, 1, -1],
            y=[-1, -1, 1, 1],
            fill="toself",
            fillcolor="green",
            line_color="green",
            mode="lines+markers",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Samples",
            x=sample_xs,
            y=sample_ys,
            mode="markers",
            marker=go.scatter.Marker(color="lightblue", symbol="circle"),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Best evaluation trajectory",
            x=best_result.trace.states[0],
            y=best_result.trace.states[1],
            mode="lines+markers",
            line=go.scatter.Line(color="blue", shape="spline"),
        )
    )

    file = ("data/coordinateascent/coordinateascent.jpeg")
    figure.write_image(file)

