# SageMaker 배포

이 저장소의 두 모델 서비스는 SageMaker custom container로 등록할 수 있다.

- `sentiment-service`
- `embedding-service`

각 서비스는 SageMaker가 기대하는 기본 엔드포인트를 제공한다.

- `GET /ping`: 헬스 체크
- `POST /invocations`: 추론 요청

## 이미지 종류

각 서비스에는 SageMaker용 Dockerfile이 두 개씩 있다.

- `Dockerfile.sagemaker-cpu`: CPU / Serverless Inference 용도
- `Dockerfile.sagemaker-cu128`: CUDA 12.8 / real-time endpoint 용도

SageMaker에서 쓸 이미지는 서비스 디렉터리에서 `linux/amd64` 플랫폼으로 빌드한다.
ECR에 바로 push할 때는 SageMaker 호환을 위해 `--provenance=false --sbom=false`를 같이 준다.

```bash
cd sentiment-service
docker buildx build --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -f Dockerfile.sagemaker-cpu \
  -t sentiment-service-sagemaker:cpu \
  --load .
```

```bash
cd embedding-service
docker buildx build --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -f Dockerfile.sagemaker-cpu \
  -t embedding-service-sagemaker:cpu \
  --load .
```

CUDA 12.8 기반 real-time endpoint에 올릴 때는 `cu128` Dockerfile을 쓴다.

```bash
docker buildx build --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -f Dockerfile.sagemaker-cu128 \
  -t sentiment-service-sagemaker:cu128 \
  --load .
```

Dockerfile은 기본적으로 빌드 시점에 Hugging Face 모델을 미리 다운로드한다. 이미지 크기는 커지지만 endpoint cold start 때 모델 다운로드 시간을 줄일 수 있다.

모델을 이미지에 미리 넣지 않고 endpoint 시작 시점에 받으려면 `PRELOAD_MODEL=0`을 준다.

```bash
docker buildx build --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --build-arg PRELOAD_MODEL=0 \
  -f Dockerfile.sagemaker-cpu \
  -t sentiment-service-sagemaker:cpu \
  --load .
```

## ECR Push

예시는 `sentiment-service` 기준이다.

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-northeast-2
REPO=sentiment-service-sagemaker

aws ecr create-repository --repository-name "$REPO" --region "$REGION"
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker tag sentiment-service-sagemaker:cpu \
  "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:cpu"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:cpu"
```

`embedding-service`는 `REPO=embedding-service-sagemaker`로 바꿔서 같은 방식으로 push한다.

또는 ECR에 바로 push한다.

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-northeast-2
REPO=sentiment-service-sagemaker

docker buildx build --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -f Dockerfile.sagemaker-cpu \
  -t "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:cpu" \
  --push .
```

## Serverless 모델 등록

SageMaker Serverless Inference CPU endpoint 예시다.

```python
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

role = "arn:aws:iam::<ACCOUNT_ID>:role/<SAGEMAKER_EXECUTION_ROLE>"
image_uri = "<ACCOUNT_ID>.dkr.ecr.ap-northeast-2.amazonaws.com/sentiment-service-sagemaker:cpu"

model = Model(
    image_uri=image_uri,
    role=role,
    name="sentiment-service",
)

predictor = model.deploy(
    endpoint_name="sentiment-service-serverless",
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=6144,
        max_concurrency=5,
    ),
)
```

## 호출

`sentiment-service` 단건 호출:

```bash
aws sagemaker-runtime invoke-endpoint \
  --region ap-northeast-2 \
  --endpoint-name sentiment-service-serverless \
  --content-type application/json \
  --cli-binary-format raw-in-base64-out \
  --body '{"text":"The battery life is terrible","aspect":"battery"}' \
  output.json
cat output.json
```

`sentiment-service` 배치 호출:

```bash
aws sagemaker-runtime invoke-endpoint \
  --region ap-northeast-2 \
  --endpoint-name sentiment-service-serverless \
  --content-type application/json \
  --cli-binary-format raw-in-base64-out \
  --body '{"texts":["The battery life is terrible","The screen is great"],"aspect":"overall"}' \
  output.json
cat output.json
```

`embedding-service`는 해당 ECR 이미지로 endpoint를 만들고 아래 형태로 호출한다.

```json
{"texts":["삼성 갤럭시 카메라 성능"],"prefix":"query: "}
```

## 선택 기준

간헐적으로 호출하는 endpoint는 `Dockerfile.sagemaker-cpu`와 SageMaker Serverless Inference부터 시도한다.

GPU가 필요하거나 Serverless 메모리 한도에 걸리면 `Dockerfile.sagemaker-cu128`로 이미지를 만들고 real-time endpoint에 올린다.
