from income.predictor import ModelResolver
from income.utils import load_object
model_resolver = ModelResolver(model_registry="saved_models")
transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
model = load_object(file_path=model_resolver.get_latest_model_path())
print(transformer)