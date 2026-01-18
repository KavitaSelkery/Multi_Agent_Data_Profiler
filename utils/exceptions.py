"""
Custom exceptions for the application
"""

class SnowflakeAgentError(Exception):
    """Base exception for Snowflake Agent errors"""
    pass

class ConnectionError(SnowflakeAgentError):
    """Raised when Snowflake connection fails"""
    pass

class AuthenticationError(SnowflakeAgentError):
    """Raised when authentication fails"""
    pass

class QueryError(SnowflakeAgentError):
    """Raised when SQL query execution fails"""
    pass

class ValidationError(SnowflakeAgentError):
    """Raised when input validation fails"""
    pass

class ConfigurationError(SnowflakeAgentError):
    """Raised when configuration is invalid or missing"""
    pass

class AgentError(SnowflakeAgentError):
    """Raised when agent execution fails"""
    pass

class WorkflowError(SnowflakeAgentError):
    """Raised when workflow execution fails"""
    pass

class DataQualityError(SnowflakeAgentError):
    """Raised when data quality checks fail"""
    pass

class ReportGenerationError(SnowflakeAgentError):
    """Raised when report generation fails"""
    pass

class CacheError(SnowflakeAgentError):
    """Raised when cache operations fail"""
    pass

class MemoryError(SnowflakeAgentError):
    """Raised when memory operations fail"""
    pass

class VectorStoreError(SnowflakeAgentError):
    """Raised when vector store operations fail"""
    pass

class LLMError(SnowflakeAgentError):
    """Raised when LLM operations fail"""
    pass

class ChainError(SnowflakeAgentError):
    """Raised when chain operations fail"""
    pass

class ToolError(SnowflakeAgentError):
    """Raised when tool execution fails"""
    pass

class RetryExhaustedError(SnowflakeAgentError):
    """Raised when all retry attempts are exhausted"""
    pass

class TimeoutError(SnowflakeAgentError):
    """Raised when operation times out"""
    pass

class ResourceExhaustedError(SnowflakeAgentError):
    """Raised when resources are exhausted"""
    pass

class PermissionError(SnowflakeAgentError):
    """Raised when permission is denied"""
    pass

class SchemaError(SnowflakeAgentError):
    """Raised when schema operations fail"""
    pass

class DataTypeError(SnowflakeAgentError):
    """Raised when data type operations fail"""
    pass

class AnomalyDetectionError(SnowflakeAgentError):
    """Raised when anomaly detection fails"""
    pass

class ProfilingError(SnowflakeAgentError):
    """Raised when data profiling fails"""
    pass

class SQLGenerationError(SnowflakeAgentError):
    """Raised when SQL generation fails"""
    pass

class VisualizationError(SnowflakeAgentError):
    """Raised when visualization generation fails"""
    pass

class ExportError(SnowflakeAgentError):
    """Raised when data export fails"""
    pass

class ImportError(SnowflakeAgentError):
    """Raised when data import fails"""
    pass

class SerializationError(SnowflakeAgentError):
    """Raised when serialization/deserialization fails"""
    pass

class DeserializationError(SnowflakeAgentError):
    """Raised when deserialization fails"""
    pass

class FileSystemError(SnowflakeAgentError):
    """Raised when file system operations fail"""
    pass

class NetworkError(SnowflakeAgentError):
    """Raised when network operations fail"""
    pass

class RateLimitError(SnowflakeAgentError):
    """Raised when rate limit is exceeded"""
    pass

class BudgetExceededError(SnowflakeAgentError):
    """Raised when budget is exceeded"""
    pass

class SecurityError(SnowflakeAgentError):
    """Raised when security violations occur"""
    pass

class AuditError(SnowflakeAgentError):
    """Raised when audit operations fail"""
    pass

class MonitoringError(SnowflakeAgentError):
    """Raised when monitoring operations fail"""
    pass

class AlertError(SnowflakeAgentError):
    """Raised when alert operations fail"""
    pass

class NotificationError(SnowflakeAgentError):
    """Raised when notification operations fail"""
    pass

class IntegrationError(SnowflakeAgentError):
    """Raised when integration operations fail"""
    pass

class DependencyError(SnowflakeAgentError):
    """Raised when dependencies are missing or fail"""
    pass

class VersionError(SnowflakeAgentError):
    """Raised when version incompatibilities occur"""
    pass

class MigrationError(SnowflakeAgentError):
    """Raised when migration operations fail"""
    pass

class BackupError(SnowflakeAgentError):
    """Raised when backup operations fail"""
    pass

class RecoveryError(SnowflakeAgentError):
    """Raised when recovery operations fail"""
    pass

class DeploymentError(SnowflakeAgentError):
    """Raised when deployment operations fail"""
    pass

class ScalingError(SnowflakeAgentError):
    """Raised when scaling operations fail"""
    pass

class PerformanceError(SnowflakeAgentError):
    """Raised when performance issues occur"""
    pass

class OptimizationError(SnowflakeAgentError):
    """Raised when optimization operations fail"""
    pass

class MaintenanceError(SnowflakeAgentError):
    """Raised when maintenance operations fail"""
    pass

class HealthCheckError(SnowflakeAgentError):
    """Raised when health checks fail"""
    pass

class DiagnosticError(SnowflakeAgentError):
    """Raised when diagnostics fail"""
    pass

class TroubleshootingError(SnowflakeAgentError):
    """Raised when troubleshooting fails"""
    pass

class DocumentationError(SnowflakeAgentError):
    """Raised when documentation operations fail"""
    pass

class TrainingError(SnowflakeAgentError):
    """Raised when training operations fail"""
    pass

class TestingError(SnowflakeAgentError):
    """Raised when testing operations fail"""
    pass

class DebuggingError(SnowflakeAgentError):
    """Raised when debugging operations fail"""
    pass

class LoggingError(SnowflakeAgentError):
    """Raised when logging operations fail"""
    pass

class MetricsError(SnowflakeAgentError):
    """Raised when metrics operations fail"""
    pass

class AnalyticsError(SnowflakeAgentError):
    """Raised when analytics operations fail"""
    pass

class BusinessIntelligenceError(SnowflakeAgentError):
    """Raised when business intelligence operations fail"""
    pass

class DataScienceError(SnowflakeAgentError):
    """Raised when data science operations fail"""
    pass

class MachineLearningError(SnowflakeAgentError):
    """Raised when machine learning operations fail"""
    pass

class AIError(SnowflakeAgentError):
    """Raised when AI operations fail"""
    pass

class ModelError(SnowflakeAgentError):
    """Raised when model operations fail"""
    pass

class PredictionError(SnowflakeAgentError):
    """Raised when prediction operations fail"""
    pass

class RecommendationError(SnowflakeAgentError):
    """Raised when recommendation operations fail"""
    pass

class InsightError(SnowflakeAgentError):
    """Raised when insight generation fails"""
    pass

class DecisionError(SnowflakeAgentError):
    """Raised when decision operations fail"""
    pass

class ActionError(SnowflakeAgentError):
    """Raised when action operations fail"""
    pass

class AutomationError(SnowflakeAgentError):
    """Raised when automation operations fail"""
    pass

class OrchestrationError(SnowflakeAgentError):
    """Raised when orchestration operations fail"""
    pass

class WorkflowOrchestrationError(SnowflakeAgentError):
    """Raised when workflow orchestration fails"""
    pass

class PipelineError(SnowflakeAgentError):
    """Raised when pipeline operations fail"""
    pass

class ETLerror(SnowflakeAgentError):
    """Raised when ETL operations fail"""
    pass

class DataPipelineError(SnowflakeAgentError):
    """Raised when data pipeline operations fail"""
    pass

class StreamingError(SnowflakeAgentError):
    """Raised when streaming operations fail"""
    pass

class BatchError(SnowflakeAgentError):
    """Raised when batch operations fail"""
    pass

class RealTimeError(SnowflakeAgentError):
    """Raised when real-time operations fail"""
    pass

class HistoricalError(SnowflakeAgentError):
    """Raised when historical operations fail"""
    pass

class ForecastError(SnowflakeAgentError):
    """Raised when forecasting operations fail"""
    pass

class TrendError(SnowflakeAgentError):
    """Raised when trend analysis fails"""
    pass

class PatternError(SnowflakeAgentError):
    """Raised when pattern operations fail"""
    pass

class CorrelationError(SnowflakeAgentError):
    """Raised when correlation operations fail"""
    pass

class CausationError(SnowflakeAgentError):
    """Raised when causation operations fail"""
    pass

class InferenceError(SnowflakeAgentError):
    """Raised when inference operations fail"""
    pass

class StatisticalError(SnowflakeAgentError):
    """Raised when statistical operations fail"""
    pass

class MathematicalError(SnowflakeAgentError):
    """Raised when mathematical operations fail"""
    pass

class AlgorithmError(SnowflakeAgentError):
    """Raised when algorithm operations fail"""
    pass

class HeuristicError(SnowflakeAgentError):
    """Raised when heuristic operations fail"""
    pass

class RuleError(SnowflakeAgentError):
    """Raised when rule operations fail"""
    pass

class LogicError(SnowflakeAgentError):
    """Raised when logic operations fail"""
    pass

class SemanticError(SnowflakeAgentError):
    """Raised when semantic operations fail"""
    pass

class SyntaxError(SnowflakeAgentError):
    """Raised when syntax operations fail"""
    pass

class GrammarError(SnowflakeAgentError):
    """Raised when grammar operations fail"""
    pass

class ParsingError(SnowflakeAgentError):
    """Raised when parsing operations fail"""
    pass

class TokenizationError(SnowflakeAgentError):
    """Raised when tokenization operations fail"""
    pass

class EmbeddingError(SnowflakeAgentError):
    """Raised when embedding operations fail"""
    pass

class VectorError(SnowflakeAgentError):
    """Raised when vector operations fail"""
    pass

class SimilarityError(SnowflakeAgentError):
    """Raised when similarity operations fail"""
    pass

class DistanceError(SnowflakeAgentError):
    """Raised when distance operations fail"""
    pass

class ClusterError(SnowflakeAgentError):
    """Raised when clustering operations fail"""
    pass

class ClassificationError(SnowflakeAgentError):
    """Raised when classification operations fail"""
    pass

class RegressionError(SnowflakeAgentError):
    """Raised when regression operations fail"""
    pass

class FeatureError(SnowflakeAgentError):
    """Raised when feature operations fail"""
    pass

class LabelError(SnowflakeAgentError):
    """Raised when labeling operations fail"""
    pass

class TrainingDataError(SnowflakeAgentError):
    """Raised when training data operations fail"""
    pass

class TestDataError(SnowflakeAgentError):
    """Raised when test data operations fail"""
    pass

class ValidationDataError(SnowflakeAgentError):
    """Raised when validation data operations fail"""
    pass

class CrossValidationError(SnowflakeAgentError):
    """Raised when cross-validation operations fail"""
    pass

class HyperparameterError(SnowflakeAgentError):
    """Raised when hyperparameter operations fail"""
    pass

class ModelSelectionError(SnowflakeAgentError):
    """Raised when model selection operations fail"""
    pass

class EnsembleError(SnowflakeAgentError):
    """Raised when ensemble operations fail"""
    pass

class BoostingError(SnowflakeAgentError):
    """Raised when boosting operations fail"""
    pass

class BaggingError(SnowflakeAgentError):
    """Raised when bagging operations fail"""
    pass

class StackingError(SnowflakeAgentError):
    """Raised when stacking operations fail"""
    pass

class VotingError(SnowflakeAgentError):
    """Raised when voting operations fail"""
    pass

class AggregationError(SnowflakeAgentError):
    """Raised when aggregation operations fail"""
    pass

class ConsensusError(SnowflakeAgentError):
    """Raised when consensus operations fail"""
    pass

class CollaborationError(SnowflakeAgentError):
    """Raised when collaboration operations fail"""
    pass

class CoordinationError(SnowflakeAgentError):
    """Raised when coordination operations fail"""
    pass

class SynchronizationError(SnowflakeAgentError):
    """Raised when synchronization operations fail"""
    pass

class CommunicationError(SnowflakeAgentError):
    """Raised when communication operations fail"""
    pass

class MessagingError(SnowflakeAgentError):
    """Raised when messaging operations fail"""
    pass

class NotificationDeliveryError(SnowflakeAgentError):
    """Raised when notification delivery fails"""
    pass

class AlertDeliveryError(SnowflakeAgentError):
    """Raised when alert delivery fails"""
    pass

class ReportDeliveryError(SnowflakeAgentError):
    """Raised when report delivery fails"""
    pass

class ExportDeliveryError(SnowflakeAgentError):
    """Raised when export delivery fails"""
    pass

class ImportDeliveryError(SnowflakeAgentError):
    """Raised when import delivery fails"""
    pass

class DataDeliveryError(SnowflakeAgentError):
    """Raised when data delivery fails"""
    pass

class InformationDeliveryError(SnowflakeAgentError):
    """Raised when information delivery fails"""
    pass

class KnowledgeDeliveryError(SnowflakeAgentError):
    """Raised when knowledge delivery fails"""
    pass

class WisdomDeliveryError(SnowflakeAgentError):
    """Raised when wisdom delivery fails"""
    pass

class InsightDeliveryError(SnowflakeAgentError):
    """Raised when insight delivery fails"""
    pass

class DecisionDeliveryError(SnowflakeAgentError):
    """Raised when decision delivery fails"""
    pass

class ActionDeliveryError(SnowflakeAgentError):
    """Raised when action delivery fails"""
    pass

class ResultDeliveryError(SnowflakeAgentError):
    """Raised when result delivery fails"""
    pass

class OutcomeDeliveryError(SnowflakeAgentError):
    """Raised when outcome delivery fails"""
    pass

class ImpactDeliveryError(SnowflakeAgentError):
    """Raised when impact delivery fails"""
    pass

class ValueDeliveryError(SnowflakeAgentError):
    """Raised when value delivery fails"""
    pass

class BenefitDeliveryError(SnowflakeAgentError):
    """Raised when benefit delivery fails"""
    pass

class ROIError(SnowflakeAgentError):
    """Raised when ROI calculations fail"""
    pass

class CostError(SnowflakeAgentError):
    """Raised when cost operations fail"""
    pass

class RevenueError(SnowflakeAgentError):
    """Raised when revenue operations fail"""
    pass

class ProfitError(SnowflakeAgentError):
    """Raised when profit operations fail"""
    pass

class LossError(SnowflakeAgentError):
    """Raised when loss operations fail"""
    pass

class RiskError(SnowflakeAgentError):
    """Raised when risk operations fail"""
    pass

class OpportunityError(SnowflakeAgentError):
    """Raised when opportunity operations fail"""
    pass

class ThreatError(SnowflakeAgentError):
    """Raised when threat operations fail"""
    pass

class StrengthError(SnowflakeAgentError):
    """Raised when strength operations fail"""
    pass

class WeaknessError(SnowflakeAgentError):
    """Raised when weakness operations fail"""
    pass

class SWOTError(SnowflakeAgentError):
    """Raised when SWOT analysis fails"""
    pass

class PESTError(SnowflakeAgentError):
    """Raised when PEST analysis fails"""
    pass

class PorterError(SnowflakeAgentError):
    """Raised when Porter's Five Forces analysis fails"""
    pass

class BCGError(SnowflakeAgentError):
    """Raised when BCG matrix analysis fails"""
    pass

class GEError(SnowflakeAgentError):
    """Raised when GE matrix analysis fails"""
    pass

class AnsoffError(SnowflakeAgentError):
    """Raised when Ansoff matrix analysis fails"""
    pass

class StrategyError(SnowflakeAgentError):
    """Raised when strategy operations fail"""
    pass

class TacticalError(SnowflakeAgentError):
    """Raised when tactical operations fail"""
    pass

class OperationalError(SnowflakeAgentError):
    """Raised when operational operations fail"""
    pass

class StrategicError(SnowflakeAgentError):
    """Raised when strategic operations fail"""
    pass

class PlanningError(SnowflakeAgentError):
    """Raised when planning operations fail"""
    pass

class ExecutionError(SnowflakeAgentError):
    """Raised when execution operations fail"""
    pass

class MonitoringExecutionError(SnowflakeAgentError):
    """Raised when execution monitoring fails"""
    pass

class ControlError(SnowflakeAgentError):
    """Raised when control operations fail"""
    pass

class AdjustmentError(SnowflakeAgentError):
    """Raised when adjustment operations fail"""
    pass

class CorrectionError(SnowflakeAgentError):
    """Raised when correction operations fail"""
    pass

class ImprovementError(SnowflakeAgentError):
    """Raised when improvement operations fail"""
    pass

class OptimizationExecutionError(SnowflakeAgentError):
    """Raised when optimization execution fails"""
    pass

class EnhancementError(SnowflakeAgentError):
    """Raised when enhancement operations fail"""
    pass

class InnovationError(SnowflakeAgentError):
    """Raised when innovation operations fail"""
    pass

class TransformationError(SnowflakeAgentError):
    """Raised when transformation operations fail"""
    pass

class ChangeError(SnowflakeAgentError):
    """Raised when change operations fail"""
    pass

class TransitionError(SnowflakeAgentError):
    """Raised when transition operations fail"""
    pass

class MigrationExecutionError(SnowflakeAgentError):
    """Raised when migration execution fails"""
    pass

class AdoptionError(SnowflakeAgentError):
    """Raised when adoption operations fail"""
    pass

class AdaptationError(SnowflakeAgentError):
    """Raised when adaptation operations fail"""
    pass

class EvolutionError(SnowflakeAgentError):
    """Raised when evolution operations fail"""
    pass

class RevolutionError(SnowflakeAgentError):
    """Raised when revolution operations fail"""
    pass

class DisruptionError(SnowflakeAgentError):
    """Raised when disruption operations fail"""
    pass

class BreakthroughError(SnowflakeAgentError):
    """Raised when breakthrough operations fail"""
    pass

class GameChangerError(SnowflakeAgentError):
    """Raised when game-changer operations fail"""
    pass

class ParadigmShiftError(SnowflakeAgentError):
    """Raised when paradigm shift operations fail"""
    pass

class MindsetError(SnowflakeAgentError):
    """Raised when mindset operations fail"""
    pass

class CultureError(SnowflakeAgentError):
    """Raised when culture operations fail"""
    pass

class BehaviorError(SnowflakeAgentError):
    """Raised when behavior operations fail"""
    pass

class AttitudeError(SnowflakeAgentError):
    """Raised when attitude operations fail"""
    pass

class BeliefError(SnowflakeAgentError):
    """Raised when belief operations fail"""
    pass

class ValueError(SnowflakeAgentError):
    """Raised when value operations fail"""
    pass

class PrincipleError(SnowflakeAgentError):
    """Raised when principle operations fail"""
    pass

class EthicError(SnowflakeAgentError):
    """Raised when ethic operations fail"""
    pass

class MoralError(SnowflakeAgentError):
    """Raised when moral operations fail"""
    pass

class VirtueError(SnowflakeAgentError):
    """Raised when virtue operations fail"""
    pass

class ViceError(SnowflakeAgentError):
    """Raised when vice operations fail"""
    pass

class SinError(SnowflakeAgentError):
    """Raised when sin operations fail"""
    pass

class CrimeError(SnowflakeAgentError):
    """Raised when crime operations fail"""
    pass

class PunishmentError(SnowflakeAgentError):
    """Raised when punishment operations fail"""
    pass

class RewardError(SnowflakeAgentError):
    """Raised when reward operations fail"""
    pass

class IncentiveError(SnowflakeAgentError):
    """Raised when incentive operations fail"""
    pass

class DisincentiveError(SnowflakeAgentError):
    """Raised when disincentive operations fail"""
    pass

class MotivationError(SnowflakeAgentError):
    """Raised when motivation operations fail"""
    pass

class DemotivationError(SnowflakeAgentError):
    """Raised when demotivation operations fail"""
    pass

class InspirationError(SnowflakeAgentError):
    """Raised when inspiration operations fail"""
    pass

class ExpirationError(SnowflakeAgentError):
    """Raised when expiration operations fail"""
    pass

class CreationError(SnowflakeAgentError):
    """Raised when creation operations fail"""
    pass

class DestructionError(SnowflakeAgentError):
    """Raised when destruction operations fail"""
    pass

class ConstructionError(SnowflakeAgentError):
    """Raised when construction operations fail"""
    pass

class DeconstructionError(SnowflakeAgentError):
    """Raised when deconstruction operations fail"""
    pass

class ReconstructionError(SnowflakeAgentError):
    """Raised when reconstruction operations fail"""
    pass

class RestorationError(SnowflakeAgentError):
    """Raised when restoration operations fail"""
    pass

class PreservationError(SnowflakeAgentError):
    """Raised when preservation operations fail"""
    pass

class ConservationError(SnowflakeAgentError):
    """Raised when conservation operations fail"""
    pass

class SustainabilityError(SnowflakeAgentError):
    """Raised when sustainability operations fail"""
    pass

class EnvironmentalError(SnowflakeAgentError):
    """Raised when environmental operations fail"""
    pass

class SocialError(SnowflakeAgentError):
    """Raised when social operations fail"""
    pass

class GovernanceError(SnowflakeAgentError):
    """Raised when governance operations fail"""
    pass

class ESGError(SnowflakeAgentError):
    """Raised when ESG operations fail"""
    pass

class CSRerror(SnowflakeAgentError):
    """Raised when CSR operations fail"""
    pass

class PhilanthropyError(SnowflakeAgentError):
    """Raised when philanthropy operations fail"""
    pass

class CharityError(SnowflakeAgentError):
    """Raised when charity operations fail"""
    pass

class DonationError(SnowflakeAgentError):
    """Raised when donation operations fail"""
    pass

class ContributionError(SnowflakeAgentError):
    """Raised when contribution operations fail"""
    pass

class ParticipationError(SnowflakeAgentError):
    """Raised when participation operations fail"""
    pass

class EngagementError(SnowflakeAgentError):
    """Raised when engagement operations fail"""
    pass

class InvolvementError(SnowflakeAgentError):
    """Raised when involvement operations fail"""
    pass

class CommitmentError(SnowflakeAgentError):
    """Raised when commitment operations fail"""
    pass

class DedicationError(SnowflakeAgentError):
    """Raised when dedication operations fail"""
    pass

class DevotionError(SnowflakeAgentError):
    """Raised when devotion operations fail"""
    pass

class LoyaltyError(SnowflakeAgentError):
    """Raised when loyalty operations fail"""
    pass

class FaithfulnessError(SnowflakeAgentError):
    """Raised when faithfulness operations fail"""
    pass

class TrustError(SnowflakeAgentError):
    """Raised when trust operations fail"""
    pass

class DistrustError(SnowflakeAgentError):
    """Raised when distrust operations fail"""
    pass

class MistrustError(SnowflakeAgentError):
    """Raised when mistrust operations fail"""
    pass

class SuspicionError(SnowflakeAgentError):
    """Raised when suspicion operations fail"""
    pass

class ParanoiaError(SnowflakeAgentError):
    """Raised when paranoia operations fail"""
    pass

class AnxietyError(SnowflakeAgentError):
    """Raised when anxiety operations fail"""
    pass

class FearError(SnowflakeAgentError):
    """Raised when fear operations fail"""
    pass

class CourageError(SnowflakeAgentError):
    """Raised when courage operations fail"""
    pass

class BraveryError(SnowflakeAgentError):
    """Raised when bravery operations fail"""
    pass

class HeroismError(SnowflakeAgentError):
    """Raised when heroism operations fail"""
    pass

class VillainyError(SnowflakeAgentError):
    """Raised when villainy operations fail"""
    pass

class GoodError(SnowflakeAgentError):
    """Raised when good operations fail"""
    pass

class EvilError(SnowflakeAgentError):
    """Raised when evil operations fail"""
    pass

class RightError(SnowflakeAgentError):
    """Raised when right operations fail"""
    pass

class WrongError(SnowflakeAgentError):
    """Raised when wrong operations fail"""
    pass

class CorrectError(SnowflakeAgentError):
    """Raised when correct operations fail"""
    pass

class IncorrectError(SnowflakeAgentError):
    """Raised when incorrect operations fail"""
    pass

class AccurateError(SnowflakeAgentError):
    """Raised when accurate operations fail"""
    pass

class InaccurateError(SnowflakeAgentError):
    """Raised when inaccurate operations fail"""
    pass

class PreciseError(SnowflakeAgentError):
    """Raised when precise operations fail"""
    pass

class ImpreciseError(SnowflakeAgentError):
    """Raised when imprecise operations fail"""
    pass

class ExactError(SnowflakeAgentError):
    """Raised when exact operations fail"""
    pass

class ApproximateError(SnowflakeAgentError):
    """Raised when approximate operations fail"""
    pass

class RoughError(SnowflakeAgentError):
    """Raised when rough operations fail"""
    pass

class SmoothError(SnowflakeAgentError):
    """Raised when smooth operations fail"""
    pass

class ContinuousError(SnowflakeAgentError):
    """Raised when continuous operations fail"""
    pass

class DiscreteError(SnowflakeAgentError):
    """Raised when discrete operations fail"""
    pass

class DigitalError(SnowflakeAgentError):
    """Raised when digital operations fail"""
    pass

class AnalogError(SnowflakeAgentError):
    """Raised when analog operations fail"""
    pass

class BinaryError(SnowflakeAgentError):
    """Raised when binary operations fail"""
    pass

class TernaryError(SnowflakeAgentError):
    """Raised when ternary operations fail"""
    pass

class QuaternaryError(SnowflakeAgentError):
    """Raised when quaternary operations fail"""
    pass

class QuinaryError(SnowflakeAgentError):
    """Raised when quinary operations fail"""
    pass

class SenaryError(SnowflakeAgentError):
    """Raised when senary operations fail"""
    pass

class SeptenaryError(SnowflakeAgentError):
    """Raised when septenary operations fail"""
    pass

class OctonaryError(SnowflakeAgentError):
    """Raised when octonary operations fail"""
    pass

class NonaryError(SnowflakeAgentError):
    """Raised when nonary operations fail"""
    pass

class DenaryError(SnowflakeAgentError):
    """Raised when denary operations fail"""
    pass

class DuodecimalError(SnowflakeAgentError):
    """Raised when duodecimal operations fail"""
    pass

class HexadecimalError(SnowflakeAgentError):
    """Raised when hexadecimal operations fail"""
    pass

class VigesimalError(SnowflakeAgentError):
    """Raised when vigesimal operations fail"""
    pass

class SexagesimalError(SnowflakeAgentError):
    """Raised when sexagesimal operations fail"""
    pass

class BaseError(SnowflakeAgentError):
    """Base class for all custom exceptions"""
    pass