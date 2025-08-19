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

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import os
from typing import Any, AsyncIterator, Iterator, cast
from fastapi import FastAPI
import httpx
from lagom import Container, Singleton
from pytest import fixture, Config
import pytest

from orionix_ai.adapters.db.json_file import JSONFileDocumentDatabase
from orionix_ai.adapters.loggers.websocket import WebSocketLogger
from orionix_ai.adapters.nlp.openai_service import OpenAIService
from orionix_ai.adapters.vector_db.transient import TransientVectorDatabase
from orionix_ai.api.app import create_api_app, ASGIApplication
from orionix_ai.api.authorization import AuthorizationPolicy, DevelopmentAuthorizationPolicy
from orionix_ai.core.background_tasks import BackgroundTaskService
from orionix_ai.core.capabilities import CapabilityStore, CapabilityVectorStore
from orionix_ai.core.common import IdGenerator
from orionix_ai.core.contextual_correlator import ContextualCorrelator
from orionix_ai.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from orionix_ai.core.emission.event_publisher import EventPublisherFactory
from orionix_ai.core.emissions import EventEmitterFactory
from orionix_ai.core.customers import CustomerDocumentStore, CustomerStore
from orionix_ai.core.engines.alpha.guideline_matching.generic import (
    observational_batch,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic import (
    guideline_previously_applied_actionable_batch,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic import (
    guideline_actionable_batch,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic import (
    guideline_previously_applied_actionable_customer_dependent_batch,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic import (
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
from orionix_ai.core.engines.alpha.optimization_policy import (
    BasicOptimizationPolicy,
    OptimizationPolicy,
)
from orionix_ai.core.engines.alpha.perceived_performance_policy import (
    NullPerceivedPerformancePolicy,
    PerceivedPerformancePolicy,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_customer_dependent_batch import (
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching,
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.guideline_actionable_batch import (
    GenericActionableGuidelineMatchesSchema,
    GenericActionableGuidelineMatching,
    GenericActionableGuidelineGuidelineMatchingShot,
)
from orionix_ai.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
    GenericPreviouslyAppliedActionableGuidelineMatching,
    GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot,
)
from orionix_ai.core.engines.alpha.tool_calling import overlapping_tools_batch, single_tool_batch
from orionix_ai.core.engines.alpha.guideline_matching.generic.response_analysis_batch import (
    GenericResponseAnalysisBatch,
    GenericResponseAnalysisSchema,
    GenericResponseAnalysisShot,
)
from orionix_ai.core.engines.alpha import message_generator
from orionix_ai.core.engines.alpha.hooks import EngineHooks
from orionix_ai.core.engines.alpha.relational_guideline_resolver import RelationalGuidelineResolver
from orionix_ai.core.engines.alpha.tool_calling.default_tool_call_batcher import DefaultToolCallBatcher
from orionix_ai.core.engines.alpha.canned_response_generator import (
    CannedResponseDraftSchema,
    CannedResponseFieldExtractionSchema,
    CannedResponseFieldExtractor,
    CannedResponsePreambleSchema,
    CannedResponseGenerator,
    CannedResponseSelectionSchema,
    CannedResponseRevisionSchema,
    BasicNoMatchResponseProvider,
    NoMatchResponseProvider,
)
from orionix_ai.core.evaluations import (
    EvaluationListener,
    PollingEvaluationListener,
    EvaluationDocumentStore,
    EvaluationStore,
)
from orionix_ai.core.journey_guideline_projection import JourneyGuidelineProjection
from orionix_ai.core.journeys import JourneyStore, JourneyVectorStore
from orionix_ai.core.services.indexing.customer_dependent_action_detector import (
    CustomerDependentActionDetector,
    CustomerDependentActionSchema,
)
from orionix_ai.core.services.indexing.guideline_action_proposer import (
    GuidelineActionProposer,
    GuidelineActionPropositionSchema,
)
from orionix_ai.core.services.indexing.guideline_agent_intention_proposer import (
    AgentIntentionProposer,
    AgentIntentionProposerSchema,
)
from orionix_ai.core.services.indexing.guideline_continuous_proposer import (
    GuidelineContinuousProposer,
    GuidelineContinuousPropositionSchema,
)
from orionix_ai.core.services.indexing.relative_action_proposer import (
    RelativeActionProposer,
    RelativeActionSchema,
)
from orionix_ai.core.services.indexing.tool_running_action_detector import (
    ToolRunningActionDetector,
    ToolRunningActionSchema,
)
from orionix_ai.core.canned_responses import CannedResponseStore, CannedResponseVectorStore
from orionix_ai.core.nlp.embedding import (
    BasicEmbeddingCache,
    Embedder,
    EmbedderFactory,
    EmbeddingCache,
    NullEmbeddingCache,
)
from orionix_ai.core.nlp.generation import T, SchematicGenerator
from orionix_ai.core.relationships import (
    RelationshipDocumentStore,
    RelationshipStore,
)
from orionix_ai.core.guidelines import GuidelineDocumentStore, GuidelineStore
from orionix_ai.adapters.db.transient import TransientDocumentDatabase
from orionix_ai.core.nlp.service import NLPService
from orionix_ai.core.persistence.data_collection import DataCollectingSchematicGenerator
from orionix_ai.core.persistence.document_database import DocumentCollection
from orionix_ai.core.services.tools.service_registry import (
    ServiceDocumentRegistry,
    ServiceRegistry,
)
from orionix_ai.core.sessions import (
    PollingSessionListener,
    SessionDocumentStore,
    SessionListener,
    SessionStore,
)
from orionix_ai.core.engines.alpha.engine import AlphaEngine
from orionix_ai.core.glossary import GlossaryStore, GlossaryVectorStore
from orionix_ai.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatcher,
    GuidelineMatchingStrategyResolver,
    ResponseAnalysisBatch,
)

from orionix_ai.core.engines.alpha.guideline_matching.generic.observational_batch import (
    GenericObservationalGuidelineMatchesSchema,
    GenericObservationalGuidelineMatchingShot,
    ObservationalGuidelineMatching,
)
from orionix_ai.core.engines.alpha.message_generator import (
    MessageGenerator,
    MessageGeneratorShot,
    MessageSchema,
)
from orionix_ai.core.engines.alpha.tool_calling.tool_caller import (
    ToolCallBatcher,
    ToolCaller,
)
from orionix_ai.core.engines.alpha.tool_event_generator import ToolEventGenerator
from orionix_ai.core.engines.types import Engine
from orionix_ai.core.services.indexing.behavioral_change_evaluation import (
    GuidelineEvaluator,
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
from orionix_ai.core.loggers import LogLevel, Logger, StdoutLogger
from orionix_ai.core.application import Application
from orionix_ai.core.agents import AgentDocumentStore, AgentStore
from orionix_ai.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from orionix_ai.core.shots import ShotCollection
from orionix_ai.core.entity_cq import EntityQueries, EntityCommands
from orionix_ai.core.tags import TagDocumentStore, TagStore
from orionix_ai.core.tools import LocalToolService

from .test_utilities import (
    GLOBAL_EMBEDDER_CACHE_FILE,
    CachedSchematicGenerator,
    JournalingEngineHooks,
    SchematicGenerationResultDocument,
    SyncAwaiter,
    create_schematic_generation_result_collection,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("caching")

    group.addoption(
        "--no-cache",
        action="store_true",
        dest="no_cache",
        default=False,
        help="Whether to avoid using the cache during the current test suite",
    )


@fixture
def correlator(request: pytest.FixtureRequest) -> Iterator[ContextualCorrelator]:
    correlator = ContextualCorrelator()

    with correlator.properties({"scope": request.node.name}):
        yield correlator


@fixture
def logger(correlator: ContextualCorrelator) -> Logger:
    return StdoutLogger(correlator=correlator, log_level=LogLevel.INFO)


@dataclass(frozen=True)
class CacheOptions:
    cache_enabled: bool
    cache_schematic_generation_collection: (
        DocumentCollection[SchematicGenerationResultDocument] | None
    )


@fixture
async def cache_options(
    request: pytest.FixtureRequest,
    logger: Logger,
) -> AsyncIterator[CacheOptions]:
    if not request.config.getoption("no_cache", True):
        logger.warning("*** Cache is enabled")

        async with (
            create_schematic_generation_result_collection(logger=logger) as schematic_collection,
        ):
            yield CacheOptions(
                cache_enabled=True,
                cache_schematic_generation_collection=schematic_collection,
            )

    else:
        yield CacheOptions(
            cache_enabled=False,
            cache_schematic_generation_collection=None,
        )


@fixture
async def sync_await() -> SyncAwaiter:
    return SyncAwaiter(asyncio.get_event_loop())


@fixture
def test_config(pytestconfig: Config) -> dict[str, Any]:
    return {"patience": 10}


async def make_schematic_generator(
    container: Container,
    cache_options: CacheOptions,
    schema: type[T],
) -> SchematicGenerator[T]:
    generator = await container[NLPService].get_schematic_generator(schema)

    if cache_options.cache_enabled:
        assert cache_options.cache_schematic_generation_collection

        generator = CachedSchematicGenerator[schema](  # type: ignore
            base_generator=generator,
            collection=cache_options.cache_schematic_generation_collection,
            use_cache=True,
        )

    if os.environ.get("PARLANT_DATA_COLLECTION", "false").lower() not in ["false", "no", "0"]:
        generator = DataCollectingSchematicGenerator[schema](  # type: ignore
            generator,
            container[ContextualCorrelator],
        )

    return generator


@fixture
async def container(
    correlator: ContextualCorrelator,
    logger: Logger,
    cache_options: CacheOptions,
) -> AsyncIterator[Container]:
    container = Container()

    container[ContextualCorrelator] = correlator
    container[Logger] = logger
    container[WebSocketLogger] = WebSocketLogger(container[ContextualCorrelator])

    container[IdGenerator] = Singleton(IdGenerator)

    async with AsyncExitStack() as stack:
        container[BackgroundTaskService] = await stack.enter_async_context(
            BackgroundTaskService(container[Logger])
        )

        await container[BackgroundTaskService].start(
            container[WebSocketLogger].start(), tag="websocket-logger"
        )

        container[AgentStore] = await stack.enter_async_context(
            AgentDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[GuidelineStore] = await stack.enter_async_context(
            GuidelineDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[RelationshipStore] = await stack.enter_async_context(
            RelationshipDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[SessionStore] = await stack.enter_async_context(
            SessionDocumentStore(TransientDocumentDatabase())
        )
        container[ContextVariableStore] = await stack.enter_async_context(
            ContextVariableDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[TagStore] = await stack.enter_async_context(
            TagDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[CustomerStore] = await stack.enter_async_context(
            CustomerDocumentStore(container[IdGenerator], TransientDocumentDatabase())
        )
        container[GuidelineToolAssociationStore] = await stack.enter_async_context(
            GuidelineToolAssociationDocumentStore(
                container[IdGenerator], TransientDocumentDatabase()
            )
        )
        container[SessionListener] = PollingSessionListener
        container[EvaluationStore] = await stack.enter_async_context(
            EvaluationDocumentStore(TransientDocumentDatabase())
        )
        container[EvaluationListener] = PollingEvaluationListener
        container[LegacyBehavioralChangeEvaluator] = LegacyBehavioralChangeEvaluator
        container[EventEmitterFactory] = Singleton(EventPublisherFactory)

        container[ServiceRegistry] = await stack.enter_async_context(
            ServiceDocumentRegistry(
                database=TransientDocumentDatabase(),
                event_emitter_factory=container[EventEmitterFactory],
                logger=container[Logger],
                correlator=container[ContextualCorrelator],
                nlp_services_provider=lambda: {"default": OpenAIService(container[Logger])},
            )
        )

        container[NLPService] = await container[ServiceRegistry].read_nlp_service("default")

        async def get_embedder_type() -> type[Embedder]:
            return type(await container[NLPService].get_embedder())

        embedder_factory = EmbedderFactory(container)

        if cache_options.cache_enabled:
            embedding_cache: EmbeddingCache = BasicEmbeddingCache(
                document_database=await stack.enter_async_context(
                    JSONFileDocumentDatabase(logger, GLOBAL_EMBEDDER_CACHE_FILE),
                )
            )
        else:
            embedding_cache = NullEmbeddingCache()

        container[JourneyStore] = await stack.enter_async_context(
            JourneyVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    embedder_factory,
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[GlossaryStore] = await stack.enter_async_context(
            GlossaryVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    embedder_factory,
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[CannedResponseStore] = await stack.enter_async_context(
            CannedResponseVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger], embedder_factory, lambda: embedding_cache
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[CapabilityStore] = await stack.enter_async_context(
            CapabilityVectorStore(
                container[IdGenerator],
                vector_db=TransientVectorDatabase(
                    container[Logger],
                    embedder_factory,
                    lambda: embedding_cache,
                ),
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

        container[EntityQueries] = Singleton(EntityQueries)
        container[EntityCommands] = Singleton(EntityCommands)

        container[JourneyGuidelineProjection] = Singleton(JourneyGuidelineProjection)

        for generation_schema in (
            GenericObservationalGuidelineMatchesSchema,
            GenericActionableGuidelineMatchesSchema,
            GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
            MessageSchema,
            CannedResponseDraftSchema,
            CannedResponseSelectionSchema,
            CannedResponsePreambleSchema,
            CannedResponseRevisionSchema,
            CannedResponseFieldExtractionSchema,
            single_tool_batch.SingleToolBatchSchema,
            overlapping_tools_batch.OverlappingToolsBatchSchema,
            ConditionsEntailmentTestsSchema,
            ActionsContradictionTestsSchema,
            GuidelineConnectionPropositionsSchema,
            GuidelineActionPropositionSchema,
            GuidelineContinuousPropositionSchema,
            CustomerDependentActionSchema,
            ToolRunningActionSchema,
            GenericResponseAnalysisSchema,
            AgentIntentionProposerSchema,
            DisambiguationGuidelineMatchesSchema,
            JourneyNodeSelectionSchema,
            RelativeActionSchema,
        ):
            container[SchematicGenerator[generation_schema]] = await make_schematic_generator(  # type: ignore
                container,
                cache_options,
                generation_schema,
            )

        container[
            ShotCollection[GenericPreviouslyAppliedActionableGuidelineGuidelineMatchingShot]
        ] = guideline_previously_applied_actionable_batch.shot_collection
        container[ShotCollection[GenericActionableGuidelineGuidelineMatchingShot]] = (
            guideline_actionable_batch.shot_collection
        )
        container[
            ShotCollection[GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchingShot]
        ] = guideline_previously_applied_actionable_customer_dependent_batch.shot_collection
        container[ShotCollection[GenericObservationalGuidelineMatchingShot]] = (
            observational_batch.shot_collection
        )
        container[ShotCollection[GenericResponseAnalysisShot]] = (
            response_analysis_batch.shot_collection
        )
        container[ShotCollection[single_tool_batch.SingleToolBatchShot]] = (
            single_tool_batch.shot_collection
        )
        container[ShotCollection[overlapping_tools_batch.OverlappingToolsBatchShot]] = (
            overlapping_tools_batch.shot_collection
        )
        container[ShotCollection[MessageGeneratorShot]] = message_generator.shot_collection

        container[GuidelineConnectionProposer] = Singleton(GuidelineConnectionProposer)
        container[CoherenceChecker] = Singleton(CoherenceChecker)
        container[GuidelineActionProposer] = Singleton(GuidelineActionProposer)
        container[GuidelineContinuousProposer] = Singleton(GuidelineContinuousProposer)
        container[CustomerDependentActionDetector] = Singleton(CustomerDependentActionDetector)
        container[AgentIntentionProposer] = Singleton(AgentIntentionProposer)
        container[ToolRunningActionDetector] = Singleton(ToolRunningActionDetector)
        container[RelativeActionProposer] = Singleton(RelativeActionProposer)
        container[LocalToolService] = cast(
            LocalToolService,
            await container[ServiceRegistry].update_tool_service(
                name="local", kind="local", url=""
            ),
        )
        container[GenericGuidelineMatchingStrategyResolver] = Singleton(
            GenericGuidelineMatchingStrategyResolver
        )
        container[GuidelineMatchingStrategyResolver] = lambda container: container[
            GenericGuidelineMatchingStrategyResolver
        ]
        container[ObservationalGuidelineMatching] = Singleton(ObservationalGuidelineMatching)
        container[GenericActionableGuidelineMatching] = Singleton(
            GenericActionableGuidelineMatching
        )
        container[GenericPreviouslyAppliedActionableGuidelineMatching] = Singleton(
            GenericPreviouslyAppliedActionableGuidelineMatching
        )
        container[GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching] = Singleton(
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatching
        )
        container[ResponseAnalysisBatch] = Singleton(GenericResponseAnalysisBatch)
        container[GuidelineMatcher] = Singleton(GuidelineMatcher)
        container[GuidelineEvaluator] = Singleton(GuidelineEvaluator)

        container[DefaultToolCallBatcher] = Singleton(DefaultToolCallBatcher)
        container[ToolCallBatcher] = lambda container: container[DefaultToolCallBatcher]
        container[ToolCaller] = Singleton(ToolCaller)
        container[RelationalGuidelineResolver] = Singleton(RelationalGuidelineResolver)
        container[CannedResponseGenerator] = Singleton(CannedResponseGenerator)
        container[NoMatchResponseProvider] = Singleton(BasicNoMatchResponseProvider)
        container[CannedResponseFieldExtractor] = Singleton(CannedResponseFieldExtractor)
        container[MessageGenerator] = Singleton(MessageGenerator)
        container[ToolEventGenerator] = Singleton(ToolEventGenerator)
        container[PerceivedPerformancePolicy] = Singleton(NullPerceivedPerformancePolicy)
        container[OptimizationPolicy] = Singleton(BasicOptimizationPolicy)

        hooks = JournalingEngineHooks()
        container[JournalingEngineHooks] = hooks
        container[EngineHooks] = hooks

        container[AuthorizationPolicy] = Singleton(DevelopmentAuthorizationPolicy)

        container[Engine] = Singleton(AlphaEngine)

        container[Application] = Application(container)

        yield container

        await container[BackgroundTaskService].cancel_all()


@fixture
async def api_app(container: Container) -> ASGIApplication:
    return await create_api_app(container)


@fixture
async def async_client(api_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app),
        base_url="http://testserver",
    ) as client:
        yield client


class NoCachedGenerations:
    pass


@fixture
def no_cache(container: Container) -> None:
    if isinstance(
        container[SchematicGenerator[GenericPreviouslyAppliedActionableGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericPreviouslyAppliedActionableGuidelineMatchesSchema],
            container[SchematicGenerator[GenericPreviouslyAppliedActionableGuidelineMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[GenericActionableGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericActionableGuidelineMatchesSchema],
            container[SchematicGenerator[GenericActionableGuidelineMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[
            SchematicGenerator[
                GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
            ]
        ],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[
                GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
            ],
            container[
                SchematicGenerator[
                    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema
                ]
            ],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[GenericObservationalGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GenericObservationalGuidelineMatchesSchema],
            container[SchematicGenerator[GenericObservationalGuidelineMatchesSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[MessageSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[MessageSchema],
            container[SchematicGenerator[MessageSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponseDraftSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponseDraftSchema],
            container[SchematicGenerator[CannedResponseDraftSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponseSelectionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponseSelectionSchema],
            container[SchematicGenerator[CannedResponseSelectionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponsePreambleSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponsePreambleSchema],
            container[SchematicGenerator[CannedResponsePreambleSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponseRevisionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponseRevisionSchema],
            container[SchematicGenerator[CannedResponseRevisionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[CannedResponseFieldExtractionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[CannedResponseFieldExtractionSchema],
            container[SchematicGenerator[CannedResponseFieldExtractionSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[single_tool_batch.SingleToolBatchSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[single_tool_batch.SingleToolBatchSchema],
            container[SchematicGenerator[single_tool_batch.SingleToolBatchSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[ConditionsEntailmentTestsSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[ConditionsEntailmentTestsSchema],
            container[SchematicGenerator[ConditionsEntailmentTestsSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[ActionsContradictionTestsSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[ActionsContradictionTestsSchema],
            container[SchematicGenerator[ActionsContradictionTestsSchema]],
        ).use_cache = False

    if isinstance(
        container[SchematicGenerator[GuidelineConnectionPropositionsSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[GuidelineConnectionPropositionsSchema],
            container[SchematicGenerator[GuidelineConnectionPropositionsSchema]],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[DisambiguationGuidelineMatchesSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[DisambiguationGuidelineMatchesSchema],
            container[SchematicGenerator[DisambiguationGuidelineMatchesSchema]],
        ).use_cache = False
    if isinstance(
        container[SchematicGenerator[JourneyNodeSelectionSchema]],
        CachedSchematicGenerator,
    ):
        cast(
            CachedSchematicGenerator[JourneyNodeSelectionSchema],
            container[SchematicGenerator[JourneyNodeSelectionSchema]],
        ).use_cache = False
