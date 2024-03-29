from __future__ import annotations

from math import inf
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, NonCallableMock

from staliro.core.cost import CostFn, Evaluation, SpecificationFactory, Thunk, TimingData
from staliro.core.interval import Interval
from staliro.core.layout import SampleLayout
from staliro.core.model import BasicResult, FailureResult, Model, ModelResult, Trace
from staliro.core.sample import Sample
from staliro.core.signal import Signal
from staliro.core.specification import Specification, SpecificationError


class TimingDataTestCase(TestCase):
    def test_total_time(self) -> None:
        self.assertEqual(TimingData(10.0, 10.0).total, 20)


class ThunkTestCase(TestCase):
    def setUp(self) -> None:
        signal = NonCallableMock(spec=Signal)
        factory = Mock(return_value=signal)

        self.sample = Sample([1, 2, 3, 4])
        self.model = NonCallableMock(spec=Model)
        self.interval = Interval(0, 1)
        self.layout = SampleLayout(
            static_parameters=(0, 2),
            signals={(2, 4): lambda vs: factory([1.0, 2.0], vs)},  # type: ignore
        )

    def test_specification_noncallable(self) -> None:
        specification: Specification[Any, Any] = NonCallableMock(spec=Specification)
        thunk: Thunk[Any, Any, Any] = Thunk(
            self.sample,
            self.model,
            specification,
            self.interval,
            self.layout,
        )

        self.assertEqual(thunk.specification, specification)

    def test_specification_callable(self) -> None:
        specification: Specification[Any, Any] = NonCallableMock(spec=Specification)
        specification_factory: SpecificationFactory[Any, Any] = Mock(return_value=specification)
        thunk: Thunk[Any, Any, Any] = Thunk(
            self.sample,
            self.model,
            specification_factory,
            self.interval,
            self.layout,
        )

        factory_result = thunk.specification

        specification_factory.assert_called_once()  # type: ignore
        specification_factory.assert_called_with(self.sample)  # type: ignore
        self.assertEqual(factory_result, specification)

    def test_specification_callable_return_type(self) -> None:
        bad_factory = Mock(return_value=None)
        thunk: Thunk[Any, Any, Any] = Thunk(
            self.sample,
            self.model,
            bad_factory,
            self.interval,
            self.layout,
        )

        with self.assertRaises(SpecificationError):
            thunk.specification

    def test_result_evaluation(self) -> None:
        trace = Trace([0.0], [0.0])
        model_result = BasicResult(trace)

        model = NonCallableMock(spec=Model)
        model.simulate = Mock(return_value=model_result)
        specification = NonCallableMock(spec=Specification)
        specification.evaluate = Mock(return_value=0)

        thunk: Thunk[Any, Any, Any] = Thunk(
            self.sample,
            model,
            specification,
            self.interval,
            self.layout,
        )
        evaluation = thunk.evaluate()
        inputs = self.layout.decompose_sample(self.sample)

        model.simulate.assert_called_once()
        model.simulate.assert_called_with(inputs, self.interval)
        specification.evaluate.assert_called_once()
        specification.evaluate.assert_called_with(
            model_result.trace.states,
            model_result.trace.times,
        )

        self.assertIsInstance(evaluation, Evaluation)
        self.assertEqual(evaluation.cost, 0)
        self.assertEqual(evaluation.sample, self.sample)

    def test_failure_evaluation(self) -> None:
        model_result: ModelResult[Any, None] = FailureResult(None)

        model = NonCallableMock(spec=Model)
        model.simulate = Mock(return_value=model_result)
        specification = NonCallableMock(spec=Specification)
        specification.evaluate = Mock(return_value=0)
        specification.failure_cost = -inf

        thunk: Thunk[Any, Any, Any] = Thunk(
            self.sample,
            model,
            specification,
            self.interval,
            self.layout,
        )
        evaluation = thunk.evaluate()
        inputs = self.layout.decompose_sample(self.sample)

        model.simulate.assert_called_once()
        model.simulate.assert_called_with(inputs, self.interval)
        specification.evaluate.assert_not_called()

        self.assertIsInstance(evaluation, Evaluation)
        self.assertEqual(evaluation.cost, -inf)
        self.assertEqual(evaluation.sample, self.sample)


class CostFnTestCase(TestCase):
    def setUp(self) -> None:
        signal = NonCallableMock(spec=Signal)
        factory = Mock(return_value=signal)
        model_result = BasicResult(Trace([0.0], [0.0]))

        self.model = NonCallableMock(spec=Model)
        self.model.simulate = Mock(return_value=model_result)
        self.specification = NonCallableMock(spec=Specification)
        self.interval = Interval(0, 1)
        self.layout = SampleLayout(
            static_parameters=(0, 2),
            signals={(2, 4): lambda vs: factory([1.0, 2.0], vs)},  # type: ignore
        )
        self.cost_fn: CostFn[Any, Any, Any] = CostFn(
            self.model,
            self.specification,
            self.interval,
            self.layout,
        )

    def test_eval_sample(self) -> None:
        sample = Sample([1, 2, 3, 4])
        cost = self.cost_fn.eval_sample(sample)

        self.model.simulate.assert_called_once()
        self.specification.evaluate.assert_called_once()

        self.assertEqual(cost, self.specification.evaluate.return_value)

        self.assertEqual(len(self.cost_fn.history), 1)
        self.assertEqual(self.cost_fn.history[0].sample, sample)

    def test_eval_samples(self) -> None:
        samples = [Sample([1, 2, 3, 4]), Sample([5, 6, 7, 8])]
        costs = self.cost_fn.eval_samples(samples)

        self.model.simulate.assert_called()
        self.assertEqual(self.model.simulate.call_count, 2)

        self.specification.evaluate.assert_called()
        self.assertEqual(self.specification.evaluate.call_count, 2)

        self.assertListEqual(costs, [self.specification.evaluate.return_value] * 2)

        self.assertEqual(len(self.cost_fn.history), 2)
        self.assertEqual(self.cost_fn.history[0].sample, samples[0])
        self.assertEqual(self.cost_fn.history[1].sample, samples[1])

    def test_single_vs_many_samples(self) -> None:
        samples = [Sample([1, 2, 3, 4]), Sample([5, 6, 7, 8])]
        single_cost_fn: CostFn[Any, Any, Any] = CostFn(
            self.model,
            self.specification,
            self.interval,
            self.layout,
        )
        many_cost_fn: CostFn[Any, Any, Any] = CostFn(
            self.model,
            self.specification,
            self.interval,
            self.layout,
        )

        single_costs = [single_cost_fn.eval_sample(sample) for sample in samples]
        many_costs = many_cost_fn.eval_samples(samples)

        self.assertListEqual(single_costs, many_costs)

        for single_eval, many_eval in zip(single_cost_fn.history, many_cost_fn.history):
            self.assertEqual(single_eval.sample, many_eval.sample)
            self.assertEqual(single_eval.cost, many_eval.cost)
            self.assertEqual(single_eval.extra, many_eval.extra)
