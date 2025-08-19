# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code=import-untyped

import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass
import importlib
import inspect
import os
import traceback
from lagom import Container, Singleton
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    cast,
)
import rich
import toml
from typing_extensions import NoReturn
import click
from pathlib import Path
import sys
import uvicorn

from orionix_ai.adapters.loggers.websocket import WebSocketLogger
from orionix_ai.api.authorization import (
    AuthorizationPolicy,
    DevelopmentAuthorizationPolicy,
    ProductionAuthorizationPolicy,
)
from orionix_ai.core.capabilities import CapabilityStore, CapabilityVectorStore
from orionix_ai.core.common import IdGenerator
from orionix_ai.core.engines.alpha import message_generator
from orionix_ai.core.engines.alpha.guideline_matching.generic import (
    guideline_actionable_batch,
    guideline_previously_applied_actionable_batch,
    guideline_previously_applied_actionable_customer_dependent_batch,
    response_analysis_batch,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.disambiguation_batch import (
    DisambiguationGuidelineMatchesSchema,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.journey_node_selection_batch import (
    JourneyNodeSelectionSchema,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic_guideline_matching_strategy_resolver import (
    GenericGuidelineMatchingStrategyResolver,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableGuidelineMatching,
    GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisSchema,
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisShot,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.guideline_actionable_batch import (
    GenericActionableGuidelineMatchesSchema,
    GenericActionableGuidelineMatching,
    GenericActionableGuidelineGuidelineMatchingShot,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_customer_dependent_batch import (
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic import observational_batch
from orionix_ai.core.engines.alpha.guideline_matching.generic.observational_batch import (
    GenericObservationalGuidelineMatchesSchema,
    ObservationalGuidelineMatching,
    GenericObservationalGuidelineMatchingShot,
)
from orionix_ai.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatcher,
    GuidelineMatchingStrategyResolver,
    ResponseAnalysisBatch,
)
from orionix_ai.core.engines.alpha.hooks import EngineHooks
from orionix_ai.core.engines.alpha.optimization_policy import (
    BasicOptimizationPolicy,
    OptimizationPolicy,
)
from orionix_ai.core.engines.alpha.perceived_performance_policy import (
    BasicPerceivedPerformancePolicy,
    PerceivedPerformancePolicy,
)
from orionix_ai.core.engines.alpha.relational_guideline_resolver import RelationalGuidelineResolver
from orionix_ai.core.engines.alpha.tool_calling.overlapping_tools_batch import (
    OverlappingToolsBatchSchema,
)
from orionix_ai.core.engines.alpha.canned_response_generator import (
    CannedResponseDraftSchema,
    CannedResponseFieldExtractionSchema,
    CannedResponseFieldExtractor,
    CannedResponsePreambleSchema,
    CannedResponseSelectionSchema,
    CannedResponseRevisionSchema,
    CannedResponseGenerator,
    BasicNoMatchResponseProvider,
    NoMatchResponseProvider,
)
from orionix_ai.core.journey_guideline_projection import JourneyGuidelineProjection
from orionix_ai.core.services.indexing.guideline_agent_intention_proposer import (
    AgentIntentionProposerSchema,
)
from orionix_ai.core.journeys import JourneyStore, JourneyVectorStore
from orionix_ai.core.persistence.vector_database import VectorDatabase
from orionix_ai.core.services.indexing.customer_dependent_action_detector import (
    CustomerDependentActionDetector,
    CustomerDependentActionSchema,
)
from orionix_ai.core.services.indexing.guideline_action_proposer import (
    GuidelineActionProposer,
    GuidelineActionPropositionSchema,
)
from orionix_ai.core.services.indexing.guideline_continuous_proposer import (
    GuidelineContinuousProposer,
    GuidelineContinuousPropositionSchema,
)
from orionix_ai.core.services.indexing.relative_action_proposer import RelativeActionSchema
from orionix_ai.core.services.indexing.tool_running_action_detector import (
    ToolRunningActionDetector,
    ToolRunningActionSchema,
)
from orionix_ai.core.canned_responses import CannedResponseStore, CannedResponseVectorStore
from orionix_ai.core.nlp.service import NLPService
from orionix_ai.core.persistence.common import MigrationRequired, ServerOutdated
from orionix_ai.core.shots import ShotCollection
from orionix_ai.core.tags import TagDocumentStore, TagStore
from orionix_ai.api.app import create_api_app, ASGIApplication
from orionix_ai.core.background_tasks import BackgroundTaskService
from orionix_ai.core.contextual_correlator import ContextualCorrelator
from orionix_ai.core.agents import AgentDocumentStore, AgentStore
from orionix_ai.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from orionix_ai.core.emission.event_publisher import EventPublisherFactory
from orionix_ai.core.emissions import EventEmitterFactory
from orionix_ai.core.customers import CustomerDocumentStore, CustomerStore
from orionix_ai.core.evaluations import (
    EvaluationListener,
    PollingEvaluationListener,
    EvaluationDocumentStore,
    EvaluationStatus,
    EvaluationStore,
)
from orionix_ai.core.entity_cq import EntityQueries, EntityCommands
from orionix_ai.core.relationships import (
    RelationshipDocumentStore,
    RelationshipStore,
)
from orionix_ai.core.guidelines import (
    GuidelineDocumentStore,
    GuidelineStore,
)
from orionix_ai.adapters.db.json_file import JSONFileDocumentDatabase
from orionix_ai.core.nlp.embedding import (
    BasicEmbeddingCache,
    Embedder,
    EmbedderFactory,
    EmbeddingCache,
    NullEmbeddingCache,
)
from orionix_ai.core.nlp.generation import SchematicGenerator
from orionix_ai.core.persistence.data_collection import DataCollectingSchematicGenerator
from orionix_ai.core.services.tools.service_registry import (
    ServiceRegistry,
    ServiceDocumentRegistry,
)
from orionix_ai.core.sessions import (
    PollingSessionListener,
    SessionDocumentStore,
    SessionListener,
    SessionStore,
)
from orionix_ai.core.glossary import GlossaryStore, GlossaryVectorStore
from orionix_ai.core.engines.alpha.engine import AlphaEngine
from orionix_ai.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from orionix_ai.core.engines.alpha.tool_calling import single_tool_batch
from orionix_ai.core.engines.alpha.tool_calling.default_tool_call_batcher import DefaultToolCallBatcher
from orionix_ai.core.engines.alpha.tool_calling.single_tool_batch import (
    SingleToolBatchSchema,
    SingleToolBatchShot,
)
from orionix_ai.core.engines.alpha.tool_calling.tool_caller import ToolCallBatcher, ToolCaller


from orionix_ai.core.engines.alpha.message_generator import (
    MessageGenerator,
    MessageGeneratorShot,
    MessageSchema,
)
from orionix_ai.core.engines.alpha.tool_event_generator import ToolEventGenerator
from orionix_ai.core.engines.types import Engine
from orionix_ai.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
    LegacyBehavioralChangeEvaluator,
)
from orionix_ai.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    ConditionsEntailmentTestsSchema,
    ActionsContradictionTestsSchema,
)
from orionix_ai.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
    GuidelineConnectionPropositionsSchema,
)
from orionix_ai.core.loggers import CompositeLogger, FileLogger, LogLevel, Logger
from orionix_ai.core.application import Application
from orionix_ai.core.version import VERSION


DEFAULT_PORT = 8800
SERVER_ADDRESS = "https://localhost"
CONFIG_FILE_PATH = Path("orionix_ai.toml")

DEFAULT_NLP_SERVICE = "openai"

DEFAULT_HOME_DIR = "runtime-data" if Path("runtime-data").exists() else "orionix_ai-data"
PARLANT_HOME_DIR = Path(os.environ.get("PARLANT_HOME", DEFAULT_HOME_DIR))
PARLANT_HOME_DIR.mkdir(parents=True, exist_ok=True)

EXIT_STACK: AsyncExitStack

DEFAULT_AGENT_NAME = "Default Agent"

sys.path.append(PARLANT_HOME_DIR.as_posix())
sys.path.append(".")

CORRELATOR = ContextualCorrelator()

LOGGER = FileLogger(PARLANT_HOME_DIR / "orionix_ai.log", CORRELATOR, LogLevel.INFO)

BACKGROUND_TASK_SERVICE = BackgroundTaskService(LOGGER)


class StartupError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


NLPServiceName = Literal[
    "anthropic",
    "aws",
    "azure",
    "cerebras",
    "deepseek",
    "gemini",
    "openai",
    "together",
    "litellm",
]


@dataclass
class StartupParameters:
    port: int
    nlp_service: NLPServiceName | Callable[[Container], Awaitable[NLPService]]
    log_level: str | LogLevel
    modules: list[str]
    migrate: bool
    configure: Callable[[Container], Awaitable[Container]] | None = None
    initialize: Callable[[Container], Awaitable[None]] | None = None


def load_nlp_service(name: str, extra_name: str, class_name: str, module_path: str) -> NLPService:
    try:
        module = importlib.import_module(module_path)
        service = getattr(module, class_name)
        return cast(NLPService, service(LOGGER))
    except ModuleNotFoundError as exc:
        LOGGER.error(f"Failed to import module: {exc.name}")
        LOGGER.critical(
            f"{name} support is not installed. Please install it with: pip install orionix_ai[{extra_name}]."
        )
        sys.exit(1)


def load_anthropic() -> NLPService:
    return load_nlp_service(
        "Anthropic", "anthropic", "AnthropicService", "orionix_ai.adapters.nlp.anthropic_service"
    )


def load_aws() -> NLPService:
    return load_nlp_service("AWS", "aws", "BedrockService", "orionix_ai.adapters.nlp.aws_service")


def load_azure() -> NLPService:
    from orionix_ai.adapters.nlp.azure_service import AzureService

    return AzureService(LOGGER)


def load_cerebras() -> NLPService:
    return load_nlp_service(
        "Cerebras", "cerebras", "CerebrasService", "orionix_ai.adapters.nlp.cerebras_service"
    )


def load_deepseek() -> NLPService:
    return load_nlp_service(
        "DeepSeek", "deepseek", "DeepSeekService", "orionix_ai.adapters.nlp.deepseek_service"
    )


def load_gemini() -> NLPService:
    return load_nlp_service(
        "Gemini", "gemini", "GeminiService", "orionix_ai.adapters.nlp.gemini_service"
    )


def load_openai() -> NLPService:
    from orionix_ai.adapters.nlp.openai_service import OpenAIService

    return OpenAIService(LOGGER)


def load_together() -> NLPService:
    return load_nlp_service(
        "Together.ai", "together", "TogetherService", "orionix_ai.adapters.nlp.together_service"
    )


def load_litellm() -> NLPService:
    return load_nlp_service(
        "LiteLLM", "litellm", "LiteLLMService", "orionix_ai.adapters.nlp.litellm_service"
    )


NLP_SERVICE_INITIALIZERS: dict[NLPServiceName, Callable[[], NLPService]] = {
    "anthropic": load_anthropic,
    "aws": load_aws,
    "azure": load_azure,
    "cerebras": load_cerebras,
    "deepseek": load_deepseek,
    "gemini": load_gemini,
    "openai": load_openai,
    "together": load_together,
    "litellm": load_litellm,
}


async def create_agent_if_absent(agent_store: AgentStore) -> None:
    agents = await agent_store.list_agents()
    if not agents:
        await agent_store.create_agent(name=DEFAULT_AGENT_NAME)


async def get_module_list_from_config() -> list[str]:
    if CONFIG_FILE_PATH.exists():
        config = toml.load(CONFIG_FILE_PATH)
        # Expecting the following toml structure:
        #
        # [orionix_ai]
        # modules = ["module_1", "module_2"]
        return list(config.get("orionix_ai", {}).get("modules", []))

    return []


@asynccontextmanager
async def load_modules(
    container: Container,
    modules: Iterable[str],
) -> AsyncIterator[tuple[Container, Sequence[tuple[str, Callable[[Container], Awaitable[None]]]]]]:
    imported_modules = []
    initializers: list[tuple[str, Callable[[Container], Awaitable[None]]]] = []

    for module_path in modules:
        module = importlib.import_module(module_path)
        imported_modules.append(module)

        if configure_module := getattr(module, "configure_module", None):
            LOGGER.info(f"Configuring module '{module.__name__}'")
            if new_container := await configure_module(container.clone()):
                container = new_container

        if initialize_module := getattr(module, "initialize_module", None):
            initializers.append((module.__name__, initialize_module))

    try:
        yield container, initializers
    finally:
        for m in reversed(imported_modules):
            if shutdown_module := getattr(module, "shutdown_module", None):
                LOGGER.info(f"Shutting down module '{m.__name__}'")
                await shutdown_module()


def _define_singleton(container: Container, interface: type, implementation: type) -> None:
    try:
        container[implementation] = Singleton(implementation)

        if interface != implementation:
            container[interface] = lambda c: c[implementation]
    except BaseException:
        rich.print(
            rich.text.Text(
                f"Error adding {implementation} as implementation for {interface}",
                style="bold red",
            )
        )
        raise


def _define_singleton_value(container: Container, interface: type, implementation: Any) -> None:
    implementation_type = getattr(implementation, "__orig_class__", type(implementation))

    try:
        container[implementation_type] = implementation

        if interface != implementation_type:
            container[interface] = lambda c: c[implementation_type]
    except BaseException:
        rich.print(
            rich.text.Text(
                f"Error adding {implementation_type} instance as implementation for {interface}",
                style="bold red",
            )
        )
        raise


@asynccontextmanager
async def setup_container() -> AsyncIterator[Container]:
    c = Container()

    c[BackgroundTaskService] = BACKGROUND_TASK_SERVICE
    c[ContextualCorrelator] = CORRELATOR
    web_socket_logger = WebSocketLogger(CORRELATOR, LogLevel.INFO)
    c[WebSocketLogger] = web_socket_logger
    c[Logger] = CompositeLogger([LOGGER, web_socket_logger])

    _define_singleton(c, IdGenerator, IdGenerator)

    _define_singleton_value(
        c, ShotCollection[GenericResponseAnalysisShot], response_analysis_batch.shot_collection
    )
    _define_singleton_value(
        c,
        ShotCollection[GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot],
        guideline_previously_applied_actionable_batch.shot_collection,
    )
    _define_singleton_value(
        c,
        ShotCollection[GenericActionableGuidelineGuidelineMatchingShot],
        guideline_actionable_batch.shot_collection,
    )
    _define_singleton_value(
        c,
        ShotCollection[GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot],
        guideline_previously_applied_actionable_customer_dependent_batch.shot_collection,
    )
    _define_singleton_value(
        c,
        ShotCollection[GenericObservationalGuidelineMatchingShot],
        observational_batch.shot_collection,
    )
    _define_singleton_value(
        c, ShotCollection[SingleToolBatchShot], single_tool_batch.shot_collection
    )
    _define_singleton_value(
        c, ShotCollection[MessageGeneratorShot], message_generator.shot_collection
    )

    _define_singleton_value(c, EngineHooks, EngineHooks())

    _define_singleton(c, EventEmitterFactory, EventPublisherFactory)

    _define_singleton(c, EntityQueries, EntityQueries)
    _define_singleton(c, EntityCommands, EntityCommands)

    _define_singleton(c, ToolEventGenerator, ToolEventGenerator)
    _define_singleton(c, CannedResponseFieldExtractor, CannedResponseFieldExtractor)
    _define_singleton(c, CannedResponseGenerator, CannedResponseGenerator)
    _define_singleton(c, NoMatchResponseProvider, BasicNoMatchResponseProvider)
    _define_singleton(c, MessageGenerator, MessageGenerator)
    _define_singleton(c, PerceivedPerformancePolicy, BasicPerceivedPerformancePolicy)
    _define_singleton(c, OptimizationPolicy, BasicOptimizationPolicy)

    _define_singleton(c, GuidelineConnectionProposer, GuidelineConnectionProposer)
    _define_singleton(c, CoherenceChecker, CoherenceChecker)
    _define_singleton(c, GuidelineActionProposer, GuidelineActionProposer)
    _define_singleton(c, GuidelineContinuousProposer, GuidelineContinuousProposer)
    _define_singleton(c, CustomerDependentActionDetector, CustomerDependentActionDetector)
    _define_singleton(c, ToolRunningActionDetector, ToolRunningActionDetector)

    _define_singleton(c, JourneyGuidelineProjection, JourneyGuidelineProjection)

    _define_singleton(c, LegacyBehavioralChangeEvaluator, LegacyBehavioralChangeEvaluator)
    _define_singleton(c, BehavioralChangeEvaluator, BehavioralChangeEvaluator)
    _define_singleton(c, EvaluationListener, PollingEvaluationListener)

    _define_singleton(c, ResponseAnalysisBatch, GenericResponseAnalysisBatch)
    _define_singleton(c, ObservationalGuidelineMatching, ObservationalGuidelineMatching)
    _define_singleton(
        c,
        GenericPreviouslyAppliedActionableGuidelineMatching,
        GenericPreviouslyAppliedActionableGuidelineMatching,
    )
    _define_singleton(c, GenericActionableGuidelineMatching, GenericActionableGuidelineMatching)
    _define_singleton(
        c,
        GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching,
        GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching,
    )

    _define_singleton(
        c, GuidelineMatchingStrategyResolver, GenericGuidelineMatchingStrategyResolver
    )

    _define_singleton(c, GuidelineMatcher, GuidelineMatcher)

    _define_singleton(c, ToolCallBatcher, DefaultToolCallBatcher)
    _define_singleton(c, ToolCaller, ToolCaller)

    _define_singleton(c, RelationalGuidelineResolver, RelationalGuidelineResolver)

    _define_singleton(
        c,
        AuthorizationPolicy,
        (
            ProductionAuthorizationPolicy
            if os.environ.get("PARLANT_ENV") == "production"
            else DevelopmentAuthorizationPolicy
        ),
    )

    _define_singleton(c, Engine, AlphaEngine)

    # The application object requires the actual container
    c[Application] = lambda rc: Application(rc)

    yield c


async def initialize_container(
    c: Container,
    nlp_service_descriptor: NLPServiceName | Callable[[Container], Awaitable[NLPService]],
    log_level: str | LogLevel,
    migrate: bool,
) -> None:
    def try_define(t: type, value: object) -> None:
        if t not in c.defined_types:
            if isinstance(value, type):
                _define_singleton(c, t, value)
            else:
                _define_singleton_value(c, t, value)

    async def try_define_func(
        t: type,
        value_func: Callable[[], Awaitable[object]],
    ) -> None:
        if t not in c.defined_types:
            c[t] = await value_func()

    async def try_define_document_store(
        store_interface: type,
        store_implementation: type,
        filename: str,
    ) -> None:
        if store_interface not in c.defined_types:
            db = await EXIT_STACK.enter_async_context(
                JSONFileDocumentDatabase(
                    c[Logger],
                    PARLANT_HOME_DIR / filename,
                )
            )

            sig = inspect.signature(store_implementation)
            params = list(sig.parameters.keys())

            # Remove 'self' from parameters list
            if "self" in params:
                params.remove("self")

            # Build arguments based on what the constructor accepts
            args: list[Any] = []

            if "id_generator" in params:
                args.append(c[IdGenerator])

            args.extend([db, migrate])

            c[store_implementation] = await EXIT_STACK.enter_async_context(
                store_implementation(*args)
            )
            c[store_interface] = lambda _c: c[store_implementation]

    async def try_define_vector_store(
        store_interface: type,
        store_implementation: type,
        vector_db_factory: Callable[[], Awaitable[VectorDatabase]],
        document_db_filename: str,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
    ) -> None:
        if store_interface not in c.defined_types:
            vector_db = await vector_db_factory()
            document_db = await EXIT_STACK.enter_async_context(
                JSONFileDocumentDatabase(
                    c[Logger],
                    PARLANT_HOME_DIR / document_db_filename,
                )
            )
            c[store_implementation] = await EXIT_STACK.enter_async_context(
                store_implementation(
                    id_generator=c[IdGenerator],
                    vector_db=vector_db,
                    document_db=document_db,
                    embedder_type_provider=embedder_type_provider,
                    embedder_factory=embedder_factory,
                )
            )
            c[store_interface] = lambda _c: c[store_implementation]

    await EXIT_STACK.enter_async_context(c[BackgroundTaskService])

    c[Logger].set_level(
        log_level
        if isinstance(log_level, LogLevel)
        else {
            "info": LogLevel.INFO,
            "debug": LogLevel.DEBUG,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL,
        }[log_level],
    )

    await c[BackgroundTaskService].start(c[WebSocketLogger].start(), tag="websocket-logger")

    try_define(SessionListener, PollingSessionListener)

    nlp_service_name: str
    nlp_service_instance: NLPService

    if isinstance(nlp_service_descriptor, str):
        nlp_service_name = nlp_service_descriptor
        nlp_service_instance = NLP_SERVICE_INITIALIZERS[nlp_service_name]()
    else:
        nlp_service_instance = await nlp_service_descriptor(c)
        nlp_service_name = nlp_service_instance.__class__.__name__

    try:
        for interface, implementation, filename in [
            (AgentStore, AgentDocumentStore, "agents.json"),
            (ContextVariableStore, ContextVariableDocumentStore, "context_variables.json"),
            (CustomerStore, CustomerDocumentStore, "customers.json"),
            (EvaluationStore, EvaluationDocumentStore, "evaluations.json"),
            (TagStore, TagDocumentStore, "tags.json"),
            (GuidelineStore, GuidelineDocumentStore, "guidelines.json"),
            (
                GuidelineToolAssociationStore,
                GuidelineToolAssociationDocumentStore,
                "guideline_tool_associations.json",
            ),
            (RelationshipStore, RelationshipDocumentStore, "relationships.json"),
            (SessionStore, SessionDocumentStore, "sessions.json"),
        ]:
            await try_define_document_store(interface, implementation, filename)

        async def make_service_document_registry() -> ServiceRegistry:
            db = await EXIT_STACK.enter_async_context(
                JSONFileDocumentDatabase(
                    c[Logger],
                    PARLANT_HOME_DIR / "services.json",
                )
            )

            return await EXIT_STACK.enter_async_context(
                ServiceDocumentRegistry(
                    database=db,
                    event_emitter_factory=c[EventEmitterFactory],
                    logger=c[Logger],
                    correlator=c[ContextualCorrelator],
                    nlp_services_provider=lambda: {nlp_service_name: nlp_service_instance},
                    allow_migration=migrate,
                )
            )

        await try_define_func(ServiceRegistry, make_service_document_registry)

        try_define(NLPService, nlp_service_instance)

        embedder_factory = EmbedderFactory(c)

        shared_chroma_db: VectorDatabase | None = None

        if c[OptimizationPolicy].use_embedding_cache():
            c[EmbeddingCache] = BasicEmbeddingCache(
                await EXIT_STACK.enter_async_context(
                    JSONFileDocumentDatabase(
                        c[Logger],
                        PARLANT_HOME_DIR / "cache_embeddings.json",
                    )
                )
            )
        else:
            c[EmbeddingCache] = NullEmbeddingCache()

        async def get_shared_chroma_db() -> VectorDatabase:
            nonlocal shared_chroma_db
            if shared_chroma_db is None:
                from orionix_ai.adapters.vector_db.chroma import ChromaDatabase

                shared_chroma_db = await EXIT_STACK.enter_async_context(
                    ChromaDatabase(
                        c[Logger],
                        PARLANT_HOME_DIR,
                        embedder_factory,
                        lambda: c[EmbeddingCache],
                    ),
                )
            return cast(VectorDatabase, shared_chroma_db)

        async def get_embedder_type() -> type[Embedder]:
            return type(await nlp_service_instance.get_embedder())

        for store_interface, store_implementation, document_db_filename in [
            (GlossaryStore, GlossaryVectorStore, "glossary_tags.json"),
            (CannedResponseStore, CannedResponseVectorStore, "canned_responses.json"),
            (JourneyStore, JourneyVectorStore, "journey_associations.json"),
            (CapabilityStore, CapabilityVectorStore, "capabilities.json"),
        ]:
            await try_define_vector_store(
                store_interface,
                store_implementation,
                lambda: get_shared_chroma_db(),
                document_db_filename,
                get_embedder_type,
                embedder_factory,
            )

    except MigrationRequired as e:
        c[Logger].critical(str(e))
        die("Please re-run with `--migrate` to migrate your data to the new version.")
    except ServerOutdated as e:
        c[Logger].critical(str(e))
        die(
            "Your runtime data came from a higher server version and is not supported.\nPlease upgrade to the latest version of Parlant."
        )

    for schema in (
        GenericResponseAnalysisSchema,
        GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
        GenericActionableGuidelineMatchesSchema,
        GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
        GenericObservationalGuidelineMatchesSchema,
        MessageSchema,
        CannedResponseDraftSchema,
        CannedResponseSelectionSchema,
        CannedResponsePreambleSchema,
        CannedResponseRevisionSchema,
        CannedResponseFieldExtractionSchema,
        SingleToolBatchSchema,
        ConditionsEntailmentTestsSchema,
        ActionsContradictionTestsSchema,
        GuidelineConnectionPropositionsSchema,
        OverlappingToolsBatchSchema,
        GuidelineActionPropositionSchema,
        GuidelineContinuousPropositionSchema,
        CustomerDependentActionSchema,
        ToolRunningActionSchema,
        AgentIntentionProposerSchema,
        DisambiguationGuidelineMatchesSchema,
        JourneyNodeSelectionSchema,
        RelativeActionSchema,
    ):
        generator = await nlp_service_instance.get_schematic_generator(schema)

        if os.environ.get("PARLANT_DATA_COLLECTION", "false").lower() not in ["false", "no", "0"]:
            generator = DataCollectingSchematicGenerator[schema](  # type: ignore
                generator,
                c[ContextualCorrelator],
            )

        try_define(
            SchematicGenerator[schema],  # type: ignore
            generator,
        )


async def recover_server_tasks(
    evaluation_store: EvaluationStore,
    evaluator: LegacyBehavioralChangeEvaluator,
) -> None:
    for evaluation in await evaluation_store.list_evaluations():
        if evaluation.status in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
            LOGGER.info(f"Recovering evaluation task: '{evaluation.id}'")
            await evaluator.run_evaluation(evaluation)


async def check_required_schema_migrations() -> None:
    from orionix_ai.bin.prepare_migration import detect_required_migrations
    from orionix_ai.adapters.vector_db.chroma import ChromaDatabase

    if await detect_required_migrations(JSONFileDocumentDatabase, ChromaDatabase):
        die(
            "You're running a particularly old version of Parlant.\n"
            "To upgrade your existing data to the new schema version, please run\n"
            "`orionix_ai-prepare-migration` and then re-run the server with `--migrate`."
        )


@asynccontextmanager
async def load_app(params: StartupParameters) -> AsyncIterator[tuple[ASGIApplication, Container]]:
    if not params.configure:
        # Running in non-pico mode
        await check_required_schema_migrations()

    global EXIT_STACK

    EXIT_STACK = AsyncExitStack()

    async with (
        setup_container() as base_container,
        EXIT_STACK,
    ):
        modules = set(await get_module_list_from_config() + params.modules)

        if modules:
            # Allow modules to return a different container
            actual_container, module_initializers = await EXIT_STACK.enter_async_context(
                load_modules(base_container, modules),
            )
        else:
            actual_container, module_initializers = base_container, []
            LOGGER.info("No external modules selected")

        if params.configure:
            actual_container = await params.configure(actual_container.clone())

        await initialize_container(
            actual_container,
            params.nlp_service,
            params.log_level,
            params.migrate,
        )

        for module_name, initializer in module_initializers:
            LOGGER.info(f"Initializing module '{module_name}'")
            await initializer(actual_container)

        if params.initialize:
            await params.initialize(actual_container)

        await recover_server_tasks(
            evaluation_store=actual_container[EvaluationStore],
            evaluator=actual_container[LegacyBehavioralChangeEvaluator],
        )

        if not params.configure:
            # Running in non-SDK mode
            await create_agent_if_absent(actual_container[AgentStore])

        _print_startup_banner()

        yield await create_api_app(actual_container), actual_container


def _print_startup_banner() -> None:
    ascii_logo = rf"""
                           ..
                        :=++++=-
                      :+***+++**+.
                    .=*****++++*+=:.
                   .=+++*******-
           ..:::::...  .::::=++
       .-+***#####**+=-..=+=:.
     :+######***********. =***=.
    =####**###**********+ .*****-
   =#******###** v{VERSION[:3]} **+ .******-
  :#*******#######****=. =********:
  .*#******#*:---=-::..-*********+
   -##*##***. -----=++*******++**:
    :*###**: =****###**********+:
      -+*#- -****************+-
        .: .*******++++++==-.
          .****+=:.
          =+=:.
         ..
    """.strip("\n")

    ascii_logo = "\n".join([f"  {line}" for line in ascii_logo.splitlines()])
    ascii_logo = f"\n{ascii_logo}\n"

    rich.print(rich.text.Text(ascii_logo, style="bold #0e8766"))


async def serve_app(
    container: Container,
    app: ASGIApplication,
    port: int,
) -> None:
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="critical",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)

    try:
        LOGGER.info(".-----------------------------------------.")
        LOGGER.info("| Server is ready for some serious action |")
        LOGGER.info("'-----------------------------------------'")
        LOGGER.info(f"Server authorization policy: {container[AuthorizationPolicy].name}")

        if isinstance(container[AuthorizationPolicy], DevelopmentAuthorizationPolicy):
            LOGGER.info(f"Try the Sandbox UI at http://localhost:{port}")
        else:
            LOGGER.info(f"Server address: http://localhost:{port}")

        await server.serve()
        await asyncio.sleep(0)  # Required to trigger the possible cancellation error
    except (KeyboardInterrupt, asyncio.CancelledError):
        await BACKGROUND_TASK_SERVICE.cancel_all(reason="Server shutting down")
    except BaseException as e:
        LOGGER.critical(traceback.format_exc())
        LOGGER.critical(e.__class__.__name__ + ": " + str(e))
        sys.exit(1)


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(1)


def require_env_keys(keys: list[str]) -> None:
    if missing_keys := [k for k in keys if not os.environ.get(k)]:
        die(f"The following environment variables are missing:\n{', '.join(missing_keys)}")


@asynccontextmanager
async def start_orionix_ai(params: StartupParameters) -> AsyncIterator[Container]:
    LOGGER.set_level(
        params.log_level
        if isinstance(params.log_level, LogLevel)
        else {
            "info": LogLevel.INFO,
            "debug": LogLevel.DEBUG,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL,
        }[params.log_level],
    )

    LOGGER.info(f"Parlant server version {VERSION}")
    LOGGER.info(f"Using home directory '{PARLANT_HOME_DIR.absolute()}'")

    if "PARLANT_HOME" not in os.environ and DEFAULT_HOME_DIR == "runtime-data":
        LOGGER.warning(
            "'runtime-data' as the default PARLANT_HOME directory is deprecated "
            "and will be removed in a future release. "
        )
        LOGGER.warning(
            "Please rename 'runtime-data' to 'orionix_ai-data' to avoid this warning in the future."
        )

    async with load_app(params) as (app, container):
        yield container

        await serve_app(
            container,
            app,
            params.port,
        )


def main() -> None:
    @click.group(invoke_without_command=True)
    @click.pass_context
    def cli(context: click.Context) -> None:
        if not context.invoked_subcommand:
            die(context.get_help())

    @cli.command(
        "help",
        context_settings={"ignore_unknown_options": True},
        help="Show help for a command",
    )
    @click.argument("command", nargs=-1, required=False)
    @click.pass_context
    def help_command(ctx: click.Context, command: Optional[tuple[str]] = None) -> None:
        def transform_and_exec_help(command: str) -> None:
            new_args = [sys.argv[0]] + command.split() + ["--help"]
            os.execvp(sys.executable, [sys.executable] + new_args)

        if not command:
            click.echo(cli.get_help(ctx))
        else:
            transform_and_exec_help(" ".join(command))

    @cli.command("run", help="Run the server")
    @click.option(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port",
    )
    @click.option(
        "--openai",
        is_flag=True,
        help="Run with OpenAI. The environment variable OPENAI_API_KEY must be set",
        default=True,
    )
    @click.option(
        "--anthropic",
        is_flag=True,
        help="Run with Anthropic. The environment variable ANTHROPIC_API_KEY must be set and install the extra package orionix_ai[anthropic].",
        default=False,
    )
    @click.option(
        "--aws",
        is_flag=True,
        help=(
            """
    Run with AWS Bedrock. The following environment variables must be set: 
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION 
    (optionally AWS_SESSION_TOKEN if you are using temporary credentials). 
    Also, install the extra package orionix_ai[aws]."""
        ),
        default=False,
    )
    @click.option(
        "--azure",
        is_flag=True,
        help="Run with Azure OpenAI. The following environment variables must be set: AZURE_API_KEY, AZURE_ENDPOINT",
        default=False,
    )
    @click.option(
        "--cerebras",
        is_flag=True,
        help="Run with Cerebras. The environment variable CEREBRAS_API_KEY must be set and install the extra package orionix_ai[cerebras].",
        default=False,
    )
    @click.option(
        "--deepseek",
        is_flag=True,
        help="Run with DeepSeek. You must set the DEEPSEEK_API_KEY environment variable and install the extra package orionix_ai[deepseek].",
        default=False,
    )
    @click.option(
        "--gemini",
        is_flag=True,
        help="Run with Gemini. The environment variable GEMINI_API_KEY must be set and install the extra package orionix_ai[gemini].",
        default=False,
    )
    @click.option(
        "--together",
        is_flag=True,
        help="Run with Together AI. The environment variable TOGETHER_API_KEY must be set and install the extra package orionix_ai[together].",
        default=False,
    )
    @click.option(
        "--litellm",
        is_flag=True,
        help="""Run with LiteLLM. The following environment variables must be set: LITELLM_PROVIDER_MODEL_NAME, LITELLM_PROVIDER_API_KEY.
                Check this link https://docs.litellm.ai/docs/providers for additional environment variables required for your provider,
                set them and install the extra package orionix_ai[litellm].""",
        default=False,
    )
    @click.option(
        "--log-level",
        type=click.Choice(["debug", "info", "warning", "error", "critical"]),
        default="info",
        help="Log level",
    )
    @click.option(
        "--module",
        multiple=True,
        default=[],
        metavar="MODULE",
        help=(
            "Specify a module to load. To load multiple modules, pass this argument multiple times. "
            "If orionix_ai.toml exists in the working directory, any additional modules specified "
            "in it will also be loaded."
        ),
    )
    @click.option(
        "--version",
        is_flag=True,
        help="Print server version and exit",
    )
    @click.option(
        "--migrate",
        is_flag=True,
        help=(
            "Enable to migrate the database schema to the latest version. "
            "Disable to exit if the database schema is not up-to-date."
        ),
    )
    @click.pass_context
    def run(
        ctx: click.Context,
        port: int,
        openai: bool,
        aws: bool,
        azure: bool,
        gemini: bool,
        deepseek: bool,
        anthropic: bool,
        cerebras: bool,
        together: bool,
        litellm: bool,
        log_level: str,
        module: tuple[str],
        version: bool,
        migrate: bool,
    ) -> None:
        if version:
            print(f"Parlant v{VERSION}")
            sys.exit(0)

        if sum([openai, aws, azure, deepseek, gemini, anthropic, cerebras, together, litellm]) > 2:
            print("error: only one NLP service profile can be selected")
            sys.exit(1)

        non_default_service_selected = any(
            (aws, azure, deepseek, gemini, anthropic, cerebras, together, litellm)
        )

        if not non_default_service_selected:
            nlp_service = "openai"
            require_env_keys(["OPENAI_API_KEY"])
        elif aws:
            nlp_service = "aws"
            require_env_keys(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"])
        elif azure:
            nlp_service = "azure"
            require_env_keys(["AZURE_API_KEY", "AZURE_ENDPOINT"])
        elif gemini:
            nlp_service = "gemini"
            require_env_keys(["GEMINI_API_KEY"])
        elif deepseek:
            nlp_service = "deepseek"
            require_env_keys(["DEEPSEEK_API_KEY"])
        elif anthropic:
            nlp_service = "anthropic"
            require_env_keys(["ANTHROPIC_API_KEY"])
        elif cerebras:
            nlp_service = "cerebras"
            require_env_keys(["CEREBRAS_API_KEY"])
        elif together:
            nlp_service = "together"
            require_env_keys(["TOGETHER_API_KEY"])
        elif litellm:
            nlp_service = "litellm"
            require_env_keys(["LITELLM_PROVIDER_MODEL_NAME", "LITELLM_PROVIDER_API_KEY"])
        else:
            assert False, "Should never get here"

        ctx.obj = StartupParameters(
            port=port,
            nlp_service=cast(NLPServiceName, nlp_service),
            log_level=log_level,
            modules=list(module),
            migrate=migrate,
        )

        async def start() -> None:
            async with start_orionix_ai(ctx.obj):
                pass

        asyncio.run(start())

    @cli.group("module", help="Create and manage enabled modules")
    def module() -> None:
        pass

    def enable_module(name: str) -> None:
        if not Path(f"{name}.py").exists():
            rich.print(rich.text.Text(f"> Module file {name}.py not found", style="bold red"))
            return

        if not CONFIG_FILE_PATH.exists():
            CONFIG_FILE_PATH.write_text(toml.dumps({"orionix_ai": {"modules": [name]}}))
        else:
            content = toml.loads(CONFIG_FILE_PATH.read_text())
            enabled_modules = cast(list[str], content["orionix_ai"]["modules"])

            if name not in enabled_modules:
                enabled_modules.append(name)

            CONFIG_FILE_PATH.write_text(toml.dumps(content))

        rich.print(rich.text.Text(f"> Enabled module {name}.py", style="bold green"))

    @module.command("create", help="Create a new module")
    @click.option(
        "-n",
        "--no-enable",
        default=False,
        is_flag=True,
        help="Do not automatically enable this module",
    )
    @click.option(
        "-t",
        "--template",
        type=click.Choice(["blank", "tool-service"]),
        default="blank",
        help="Start with a module template",
    )
    @click.argument("MODULE_NAME")
    def create_module(module_name: str, no_enable: bool, template: str) -> None:
        filename = Path(f"{module_name}.py")

        if filename.exists():
            die("Module already exists. Please remove it to create a new one under the same name.")

        if template == "blank":
            content = """\
from lagom import Container

async def configure_module(container: Container) -> Container:
    pass

async def initialize_module(container: Container) -> None:
    pass

async def shutdown_module() -> None:
    pass
"""
        elif template == "tool-service":
            content = f"""\
from contextlib import AsyncExitStack
from lagom import Container
from typing import Annotated

from orionix_ai.sdk import (
    PluginServer,
    ServiceRegistry,
    ToolContext,
    ToolParameterOptions,
    ToolResult,
    tool,
)


EXIT_STACK = AsyncExitStack()


@tool
async def greet_person(
    context: ToolContext,
    person_name: Annotated[
        str,
        ToolParameterOptions(
            description="The name of the person to greet",
            source="any",
        ),
    ],
) -> ToolResult:
    return ToolResult({{"message": f"Howdy, {{person_name}}!"}})

PORT = 8199
TOOLS = [greet_person]

async def initialize_module(container: Container) -> None:
    host = "127.0.0.1"

    server = PluginServer(
        tools=TOOLS,
        port=PORT,
        host=host,
        hosted=True,
    )

    await container[ServiceRegistry].update_tool_service(
        name="{module_name}",
        kind="sdk",
        url=f"http://{{host}}:{{PORT}}",
        transient=True,
    )

    await EXIT_STACK.enter_async_context(server)
    EXIT_STACK.push_async_callback(server.shutdown)


async def shutdown_module() -> None:
    await EXIT_STACK.aclose()

"""

        filename.write_text(content)

        rich.print(rich.text.Text(f"> Created module file {module_name}.py", style="bold green"))

        if not no_enable:
            enable_module(module_name)

    @module.command("enable", help="Enable a module")
    @click.argument("MODULE_NAME")
    def module_enable(module_name: str) -> None:
        enable_module(module_name)

    @module.command("disable", help="Disable a module")
    @click.argument("MODULE_NAME")
    def module_disable(module_name: str) -> None:
        if not CONFIG_FILE_PATH.exists():
            rich.print(rich.text.Text(f"> Module {module_name} was not enabled", style="bold red"))
            return
        else:
            content = toml.loads(CONFIG_FILE_PATH.read_text())
            enabled_modules = cast(list[str], content["orionix_ai"]["modules"])

            if module_name in enabled_modules:
                enabled_modules.remove(module_name)

            CONFIG_FILE_PATH.write_text(toml.dumps(content))

        rich.print(rich.text.Text(f"> Disabled module {module_name}", style="bold green"))

    @module.command("list", help="List enabled modules")
    def module_list() -> None:
        if not CONFIG_FILE_PATH.exists():
            print("No modules enabled")
            return
        else:
            content = toml.loads(CONFIG_FILE_PATH.read_text())
            enabled_modules = cast(list[str], content["orionix_ai"]["modules"])
            print(", ".join(enabled_modules))

    try:
        cli()
    except StartupError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
