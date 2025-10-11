from .embedding import EmbeddingJob
from .evaluation import EvaluationJob
from .extract import ExtractJob
from .labelling import LabellingJob
from .preprocessing import PreprocessingJob
from .taxonomy import TaxonomyJob

JobKind = (
    EmbeddingJob
    | EvaluationJob
    | ExtractJob
    | LabellingJob
    | PreprocessingJob
    | TaxonomyJob
)
