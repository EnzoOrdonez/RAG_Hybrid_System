"""Study-only conditions. Historical experiment configs and UI recipes stay intact."""
import os

from src.pipeline.pipeline_config import SURVEY_DEPLOY

STUDY_NO_RAG = SURVEY_DEPLOY.model_copy(update={
    'name': 'Study generator without evidence', 'retrieval_method': 'none',
    'embedding_model': None, 'reranker': None, 'fusion_method': None,
    'query_expansion': False, 'terminology_normalization': False,
    'prompt_routing': False, 'balance_cross_cloud_providers': False,
    'retrieval_top_k': 0, 'reranker_top_k': 0, 'final_top_k': 0,
})


def build_study_pipeline(condition, *, index_factory=None, llm_factory=None, pipeline_factory=None):
    from src.ui.components.index_loader import load_hybrid_index
    from src.generation.llm_manager import LLMManager
    from src.pipeline.rag_pipeline import RAGPipeline
    if condition not in ('hybrid', 'no_rag'):
        raise ValueError('Unknown study condition')
    index_factory = index_factory or load_hybrid_index
    llm_factory = llm_factory or LLMManager
    pipeline_factory = pipeline_factory or RAGPipeline
    config = SURVEY_DEPLOY.model_copy() if condition == 'hybrid' else STUDY_NO_RAG.model_copy()
    llm = llm_factory(provider='ollama', model=SURVEY_DEPLOY.llm_model, cache_enabled=False,
        seed=42, max_retries=1, timeout=60, read_timeout=180, default_keep_alive='30m',
        enforce_timeout=True, num_ctx=4096, expected_model_digest=os.environ.get('CLOUDRAG_MODEL_DIGEST'))
    return pipeline_factory(config=config, hybrid_index=index_factory() if condition == 'hybrid' else None,
                            llm_manager=llm)
