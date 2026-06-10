from __future__ import annotations

import sys
import types


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, data):
        self.data = data
        if data and isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        else:
            self.shape = (len(data),)

    def __iter__(self):
        if self.shape and len(self.shape) == 2:
            for row in self.data:
                yield FakeTensor(row)
        else:
            for value in self.data:
                yield FakeScalar(value)

    def __getitem__(self, index):
        value = self.data[index]
        if isinstance(value, list):
            return FakeTensor(value)
        return FakeScalar(value)

    def to(self, device, **kwargs):
        self.to_kwargs = kwargs
        return self

    def detach(self):
        return self

    def cpu(self):
        return self


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _softmax(tensor, dim=-1):
    return FakeTensor([[0.05, 0.1, 0.85] for _ in tensor.data])


def _argmax(tensor, dim=-1):
    return FakeTensor([max(range(len(row)), key=row.__getitem__) for row in tensor.data])


def install_fake_ml_modules() -> types.ModuleType:
    fake_torch = types.ModuleType("torch")
    fake_torch.long = "long"
    fake_torch.float = "float"
    fake_torch.qint8 = "qint8"
    fake_torch.Tensor = FakeTensor
    fake_torch.ones = lambda shape, dtype=None: FakeTensor(
        [[1] * shape[1] for _ in range(shape[0])]
    )
    fake_torch.tensor = lambda rows: FakeTensor(rows)
    fake_torch.softmax = _softmax
    fake_torch.argmax = _argmax
    fake_torch.no_grad = FakeInferenceMode
    fake_torch.inference_mode = FakeInferenceMode
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.nn = types.SimpleNamespace(Linear=object)
    fake_torch.ao = types.SimpleNamespace(
        quantization=types.SimpleNamespace(
            quantize_dynamic=lambda model, layers, dtype=None: model
        )
    )
    sys.modules["torch"] = fake_torch

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForSequenceClassification = object
    fake_transformers.AutoTokenizer = object
    sys.modules["transformers"] = fake_transformers
    return fake_torch


def install_fake_fastapi_modules() -> None:
    fake_fastapi = types.ModuleType("fastapi")

    class FakeFastAPI:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get(self, *args, **kwargs):
            return lambda func: func

        def post(self, *args, **kwargs):
            return lambda func: func

    class FakeHTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeRequest:
        def __init__(self, payload=None) -> None:
            self._payload = {} if payload is None else payload

        async def json(self):
            return self._payload

    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.HTTPException = FakeHTTPException
    fake_fastapi.Request = FakeRequest
    sys.modules["fastapi"] = fake_fastapi

    fake_concurrency = types.ModuleType("fastapi.concurrency")

    async def run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    fake_concurrency.run_in_threadpool = run_in_threadpool
    sys.modules["fastapi.concurrency"] = fake_concurrency

    fake_responses = types.ModuleType("fastapi.responses")

    class FakeJSONResponse:
        def __init__(self, content=None, status_code: int = 200, **kwargs) -> None:
            self.content = content
            self.status_code = status_code

    fake_responses.JSONResponse = FakeJSONResponse
    sys.modules["fastapi.responses"] = fake_responses
