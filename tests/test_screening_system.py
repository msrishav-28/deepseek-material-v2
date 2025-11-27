"""Tests for high-throughput screening system."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from ceramic_discovery.screening import (
    WorkflowOrchestrator,
    JobScheduler,
    WorkflowStatus,
    TaskStatus,
    RetryConfig,
    ScreeningEngine,
    MaterialCandidate,
    ScreeningConfig,
    RankingCriterion,
    ApplicationRanker,
    ApplicationSpec,
)
from ceramic_discovery.dft.stability_analyzer import StabilityAnalyzer


class TestWorkflowOrchestrator:
    """Test workflow orchestrator functionality."""
    
    def test_orchestrator_initialization(self):
        """Test workflow orchestrator initialization."""
        orchestrator = WorkflowOrchestrator(
            workflow_id="test_workflow",
            max_parallel_jobs=4
        )
        
        assert orchestrator.workflow_id == "test_workflow"
        assert orchestrator.max_parallel_jobs == 4
        assert orchestrator.progress is None
    
    def test_task_registration(self):
        """Test task registration."""
        orchestrator = WorkflowOrchestrator(workflow_id="test")
        
        def sample_task(x):
            return x * 2
        
        orchestrator.register_task("double", sample_task)
        assert "double" in orchestrator._task_registry
    
    def test_sequential_execution(self):
        """Test sequential task execution."""
        orchestrator = WorkflowOrchestrator(workflow_id="test_seq")
        
        # Register tasks
        def add_one(x):
            return x + 1
        
        def multiply_two(x):
            return x * 2
        
        orchestrator.register_task("add_one", add_one)
        orchestrator.register_task("multiply_two", multiply_two)
        
        # Execute workflow
        tasks = [
            {'name': 'add_one', 'args': [5]},
            {'name': 'multiply_two', 'args': [3]},
        ]
        
        progress = orchestrator.execute_workflow(tasks, use_parallel=False)
        
        assert progress.status == WorkflowStatus.COMPLETED
        assert progress.total_tasks == 2
        assert progress.completed_tasks == 2
        assert len(progress.task_results) == 2
    
    def test_retry_logic(self):
        """Test task retry with exponential backoff."""
        orchestrator = WorkflowOrchestrator(
            workflow_id="test_retry",
            retry_config=RetryConfig(max_retries=2, initial_delay_seconds=0.1)
        )
        
        # Task that fails first time, succeeds second time
        call_count = {'count': 0}
        
        def flaky_task():
            call_count['count'] += 1
            if call_count['count'] < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        orchestrator.register_task("flaky", flaky_task)
        
        tasks = [{'name': 'flaky', 'args': []}]
        progress = orchestrator.execute_workflow(tasks, use_parallel=False)
        
        assert progress.status == WorkflowStatus.COMPLETED
        assert progress.completed_tasks == 1
        assert progress.task_results[0].retry_count > 0
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            
            orchestrator = WorkflowOrchestrator(
                workflow_id="test_checkpoint",
                checkpoint_dir=checkpoint_dir
            )
            
            def simple_task(x):
                return x + 1
            
            orchestrator.register_task("simple", simple_task)
            
            tasks = [{'name': 'simple', 'args': [1]}]
            progress = orchestrator.execute_workflow(tasks, use_parallel=False)
            
            # Load checkpoint
            loaded_progress = orchestrator.load_checkpoint()
            
            assert loaded_progress is not None
            assert loaded_progress.workflow_id == "test_checkpoint"
            assert loaded_progress.completed_tasks == 1


class TestJobScheduler:
    """Test job scheduler functionality."""
    
    def test_scheduler_initialization(self):
        """Test job scheduler initialization."""
        scheduler = JobScheduler(
            max_concurrent_jobs=4,
            resource_limits={'cpu': 16, 'memory_gb': 64}
        )
        
        assert scheduler.max_concurrent_jobs == 4
        assert scheduler.resource_limits['cpu'] == 16
    
    def test_job_submission(self):
        """Test job submission."""
        scheduler = JobScheduler(max_concurrent_jobs=2)
        
        def sample_job():
            return "result"
        
        job_id = scheduler.submit_job(
            job_id="job1",
            job_func=sample_job,
            priority=1
        )
        
        assert job_id == "job1"
        assert len(scheduler.queued_jobs) == 1
    
    def test_job_scheduling(self):
        """Test job scheduling based on resources."""
        scheduler = JobScheduler(max_concurrent_jobs=2)
        
        def job():
            return "done"
        
        # Submit multiple jobs
        scheduler.submit_job("job1", job, priority=1)
        scheduler.submit_job("job2", job, priority=2)
        scheduler.submit_job("job3", job, priority=0)
        
        # Schedule jobs
        scheduled = scheduler.schedule_jobs()
        
        # Should schedule up to max_concurrent_jobs
        assert len(scheduled) <= 2
        assert len(scheduler.active_jobs) <= 2
        
        # Higher priority job should be scheduled first
        assert "job2" in scheduled
    
    def test_resource_limits(self):
        """Test resource limit enforcement."""
        scheduler = JobScheduler(
            max_concurrent_jobs=10,
            resource_limits={'cpu': 8}
        )
        
        def job():
            return "done"
        
        # Submit jobs with resource requirements
        scheduler.submit_job(
            "job1", job,
            resource_requirements={'cpu': 4}
        )
        scheduler.submit_job(
            "job2", job,
            resource_requirements={'cpu': 4}
        )
        scheduler.submit_job(
            "job3", job,
            resource_requirements={'cpu': 4}
        )
        
        # Schedule jobs
        scheduled = scheduler.schedule_jobs()
        
        # Should only schedule 2 jobs (4 + 4 = 8 CPU)
        assert len(scheduled) == 2
        assert len(scheduler.active_jobs) == 2


class TestScreeningEngine:
    """Test screening engine functionality."""
    
    @pytest.fixture
    def stability_analyzer(self):
        """Create stability analyzer."""
        return StabilityAnalyzer(metastable_threshold=0.1)
    
    @pytest.fixture
    def screening_config(self):
        """Create screening configuration."""
        return ScreeningConfig(
            stability_threshold=0.1,
            require_viable=True,
            stability_weight=0.4,
            performance_weight=0.4,
            cost_weight=0.2,
        )
    
    @pytest.fixture
    def sample_candidates(self):
        """Create sample material candidates."""
        return [
            MaterialCandidate(
                material_id="mat1",
                formula="SiC:Y(1%)",
                base_ceramic="SiC",
                dopant_element="Y",
                dopant_concentration=0.01,
                energy_above_hull=0.05,
                formation_energy=-1.5,
            ),
            MaterialCandidate(
                material_id="mat2",
                formula="SiC:Zr(2%)",
                base_ceramic="SiC",
                dopant_element="Zr",
                dopant_concentration=0.02,
                energy_above_hull=0.15,  # Unstable
                formation_energy=-1.2,
            ),
            MaterialCandidate(
                material_id="mat3",
                formula="B4C:Ti(3%)",
                base_ceramic="B4C",
                dopant_element="Ti",
                dopant_concentration=0.03,
                energy_above_hull=0.02,
                formation_energy=-1.8,
            ),
        ]
    
    def test_engine_initialization(self, stability_analyzer, screening_config):
        """Test screening engine initialization."""
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config
        )
        
        assert engine.stability_analyzer is not None
        assert engine.config.stability_threshold == 0.1
    
    def test_stability_filtering(self, stability_analyzer, screening_config, sample_candidates):
        """Test stability filtering with correct criteria."""
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config
        )
        
        results = engine.screen_candidates(sample_candidates, "test_screening")
        
        # Should filter out mat2 (energy_above_hull = 0.15 > 0.1)
        assert results.viable_candidates == 2
        assert results.total_candidates == 3
        
        # Check that unstable candidate was filtered
        viable_ids = [c.material_id for c in results.ranked_candidates]
        assert "mat1" in viable_ids
        assert "mat3" in viable_ids
        assert "mat2" not in viable_ids
    
    def test_ranking_scores(self, stability_analyzer, screening_config, sample_candidates):
        """Test ranking score calculation."""
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config
        )
        
        results = engine.screen_candidates(sample_candidates, "test_ranking")
        
        # All viable candidates should have scores
        for candidate in results.ranked_candidates:
            assert candidate.stability_score >= 0
            assert candidate.combined_score >= 0
    
    def test_multi_objective_ranking(self, stability_analyzer, sample_candidates):
        """Test multi-objective ranking with different weights."""
        # Config emphasizing stability
        config_stability = ScreeningConfig(
            stability_weight=0.7,
            performance_weight=0.2,
            cost_weight=0.1,
        )
        
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=config_stability
        )
        
        results = engine.screen_candidates(sample_candidates, "test_multi")
        
        # Candidates should be ranked
        assert len(results.ranked_candidates) > 0
        
        # Scores should be in descending order
        scores = [c.combined_score for c in results.ranked_candidates]
        assert scores == sorted(scores, reverse=True)
    
    def test_batch_screening(self, stability_analyzer, screening_config):
        """Test batch processing for large-scale screening."""
        # Create many candidates
        candidates = []
        for i in range(50):
            candidates.append(
                MaterialCandidate(
                    material_id=f"mat{i}",
                    formula=f"SiC:Y({i}%)",
                    base_ceramic="SiC",
                    dopant_element="Y",
                    dopant_concentration=0.01,
                    energy_above_hull=0.05,
                    formation_energy=-1.5,
                )
            )
        
        # Set small batch size
        screening_config.batch_size = 10
        
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config
        )
        
        batch_results = engine.batch_screen(candidates, "batch_test")
        
        # Should create multiple batches
        assert len(batch_results) == 5  # 50 / 10 = 5 batches
    
    def test_cost_efficiency_estimation(self, stability_analyzer, screening_config):
        """Test cost efficiency estimation."""
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config
        )
        
        # Common element (high efficiency)
        candidate_common = MaterialCandidate(
            material_id="mat_common",
            formula="SiC:Al(1%)",
            base_ceramic="SiC",
            dopant_element="Al",
            dopant_concentration=0.01,
            energy_above_hull=0.05,
            formation_energy=-1.5,
        )
        
        # Rare element (low efficiency)
        candidate_rare = MaterialCandidate(
            material_id="mat_rare",
            formula="SiC:Ta(1%)",
            base_ceramic="SiC",
            dopant_element="Ta",
            dopant_concentration=0.01,
            energy_above_hull=0.05,
            formation_energy=-1.5,
        )
        
        common_score = engine._estimate_cost_efficiency(candidate_common)
        rare_score = engine._estimate_cost_efficiency(candidate_rare)
        
        # Common element should have higher cost efficiency
        assert common_score > rare_score
    
    def test_results_export(self, stability_analyzer, screening_config, sample_candidates):
        """Test screening results export."""
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config
        )
        
        results = engine.screen_candidates(sample_candidates, "test_export")
        
        # Convert to DataFrame
        df = results.to_dataframe()
        assert len(df) == results.viable_candidates
        assert 'material_id' in df.columns
        assert 'combined_score' in df.columns
        
        # Save to file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            results.save(output_path)
            assert output_path.exists()


class TestRetryConfig:
    """Test retry configuration."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay_seconds=1.0,
            backoff_factor=2.0,
            max_delay_seconds=10.0
        )
        
        # First retry
        delay1 = config.get_delay(0)
        assert delay1 == 1.0
        
        # Second retry
        delay2 = config.get_delay(1)
        assert delay2 == 2.0
        
        # Third retry
        delay3 = config.get_delay(2)
        assert delay3 == 4.0
        
        # Should cap at max_delay
        delay_large = config.get_delay(10)
        assert delay_large == 10.0


class TestEnhancedScreeningWithApplicationRanking:
    """Integration tests for enhanced screening engine with application ranking."""
    
    @pytest.fixture
    def stability_analyzer(self):
        """Create stability analyzer."""
        return StabilityAnalyzer(metastable_threshold=0.1)
    
    @pytest.fixture
    def application_ranker(self):
        """Create application ranker with default applications."""
        return ApplicationRanker()
    
    @pytest.fixture
    def screening_config_with_apps(self):
        """Create screening configuration with application ranking enabled."""
        return ScreeningConfig(
            stability_threshold=0.1,
            require_viable=True,
            stability_weight=0.3,
            performance_weight=0.3,
            cost_weight=0.2,
            enable_application_ranking=True,
            application_ranking_weight=0.2,
            selected_applications=['aerospace_hypersonic', 'cutting_tools'],
        )
    
    @pytest.fixture
    def sample_candidates_with_properties(self):
        """Create sample material candidates with properties for application ranking."""
        return [
            MaterialCandidate(
                material_id="mat1",
                formula="SiC",
                base_ceramic="SiC",
                energy_above_hull=0.05,
                formation_energy=-2.5,
                predicted_hardness=32.0,  # Good for cutting tools
                predicted_fracture_toughness=4.5,
            ),
            MaterialCandidate(
                material_id="mat2",
                formula="TiC",
                base_ceramic="TiC",
                energy_above_hull=0.08,
                formation_energy=-3.0,
                predicted_hardness=35.0,  # Excellent for cutting tools
                predicted_fracture_toughness=5.0,
            ),
            MaterialCandidate(
                material_id="mat3",
                formula="B4C",
                base_ceramic="B4C",
                energy_above_hull=0.03,
                formation_energy=-1.8,
                predicted_hardness=28.0,  # Good for aerospace
                predicted_fracture_toughness=3.5,
            ),
        ]
    
    def test_screening_with_application_ranking(
        self,
        stability_analyzer,
        application_ranker,
        screening_config_with_apps,
        sample_candidates_with_properties
    ):
        """Test screening with application ranking enabled."""
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config_with_apps,
            application_ranker=application_ranker
        )
        
        results = engine.screen_candidates(
            sample_candidates_with_properties,
            "test_app_ranking"
        )
        
        # All candidates should pass stability filtering
        assert results.viable_candidates == 3
        
        # Application rankings should be present
        assert results.application_rankings is not None
        assert 'aerospace_hypersonic' in results.application_rankings
        assert 'cutting_tools' in results.application_rankings
        
        # Candidates should have application scores
        for candidate in results.ranked_candidates:
            assert candidate.application_scores is not None
            assert 'aerospace_hypersonic' in candidate.application_scores
            assert 'cutting_tools' in candidate.application_scores
            
            # Scores should be in [0, 1] range
            for app, score in candidate.application_scores.items():
                assert 0.0 <= score <= 1.0
    
    def test_multi_objective_optimization(
        self,
        stability_analyzer,
        application_ranker,
        sample_candidates_with_properties
    ):
        """Test multi-objective optimization with custom weights."""
        config = ScreeningConfig(
            stability_threshold=0.1,
            require_viable=True,
            enable_application_ranking=True,
        )
        
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=config,
            application_ranker=application_ranker
        )
        
        # Screen candidates
        results = engine.screen_candidates(
            sample_candidates_with_properties,
            "test_multi_obj"
        )
        
        # Define custom objectives
        objectives = {
            'stability': 0.3,
            'performance': 0.3,
            'application:cutting_tools': 0.4,
        }
        
        # Rank by multiple objectives
        ranked = engine.rank_by_multiple_objectives(
            results.ranked_candidates,
            objectives
        )
        
        # Should return all candidates
        assert len(ranked) == 3
        
        # Scores should be in descending order
        scores = [c.combined_score for c in ranked]
        assert scores == sorted(scores, reverse=True)
        
        # TiC should rank high for cutting tools (hardness 35.0)
        top_candidate = ranked[0]
        assert top_candidate.formula in ['TiC', 'SiC']
    
    def test_filter_by_application(
        self,
        stability_analyzer,
        application_ranker,
        screening_config_with_apps,
        sample_candidates_with_properties
    ):
        """Test filtering candidates by application score threshold."""
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config_with_apps,
            application_ranker=application_ranker
        )
        
        results = engine.screen_candidates(
            sample_candidates_with_properties,
            "test_filter"
        )
        
        # Filter by aerospace application with threshold
        filtered = engine.filter_by_application(
            results,
            'aerospace_hypersonic',
            min_score=0.3
        )
        
        # Should return some candidates
        assert len(filtered) >= 0
        
        # All filtered candidates should meet threshold
        for candidate in filtered:
            assert candidate.application_scores['aerospace_hypersonic'] >= 0.3
    
    def test_pareto_front_calculation(
        self,
        stability_analyzer,
        application_ranker,
        screening_config_with_apps,
        sample_candidates_with_properties
    ):
        """Test Pareto front calculation for multi-objective optimization."""
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config_with_apps,
            application_ranker=application_ranker
        )
        
        results = engine.screen_candidates(
            sample_candidates_with_properties,
            "test_pareto"
        )
        
        # Get Pareto front
        objectives = ['stability', 'performance', 'application:cutting_tools']
        pareto_front = engine.get_pareto_front(
            results.ranked_candidates,
            objectives
        )
        
        # Should have at least one Pareto-optimal candidate
        assert len(pareto_front) >= 1
        assert len(pareto_front) <= len(results.ranked_candidates)
        
        # All candidates in Pareto front should be non-dominated
        for candidate in pareto_front:
            assert candidate in results.ranked_candidates
    
    def test_backward_compatibility(
        self,
        stability_analyzer,
        sample_candidates_with_properties
    ):
        """Test backward compatibility - screening without application ranking."""
        # Create config without application ranking
        config = ScreeningConfig(
            stability_threshold=0.1,
            require_viable=True,
            stability_weight=0.4,
            performance_weight=0.4,
            cost_weight=0.2,
            enable_application_ranking=False,
        )
        
        # Create engine without application ranker
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=config
        )
        
        # Should work without application ranking
        results = engine.screen_candidates(
            sample_candidates_with_properties,
            "test_backward_compat"
        )
        
        # Should complete successfully
        assert results.viable_candidates == 3
        assert results.application_rankings is None
        
        # Candidates should not have application scores
        for candidate in results.ranked_candidates:
            assert candidate.application_scores is None or len(candidate.application_scores) == 0
    
    def test_application_ranking_with_missing_properties(
        self,
        stability_analyzer,
        application_ranker,
        screening_config_with_apps
    ):
        """Test application ranking handles missing properties gracefully."""
        # Create candidates with some missing properties
        candidates = [
            MaterialCandidate(
                material_id="mat1",
                formula="SiC",
                base_ceramic="SiC",
                energy_above_hull=0.05,
                formation_energy=-2.5,
                predicted_hardness=32.0,
                # Missing fracture_toughness
            ),
            MaterialCandidate(
                material_id="mat2",
                formula="TiC",
                base_ceramic="TiC",
                energy_above_hull=0.08,
                formation_energy=-3.0,
                # Missing both hardness and fracture_toughness
            ),
        ]
        
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=screening_config_with_apps,
            application_ranker=application_ranker
        )
        
        # Should handle missing properties without crashing
        results = engine.screen_candidates(candidates, "test_missing_props")
        
        assert results.viable_candidates == 2
        
        # Should still have application scores (partial scoring)
        for candidate in results.ranked_candidates:
            if candidate.application_scores:
                for score in candidate.application_scores.values():
                    assert 0.0 <= score <= 1.0
    
    def test_custom_application_specification(
        self,
        stability_analyzer,
        sample_candidates_with_properties
    ):
        """Test screening with custom application specification."""
        # Create custom application
        custom_app = ApplicationSpec(
            name='custom_armor',
            description='Custom armor application',
            target_hardness=(30.0, 40.0),
            target_formation_energy=(-4.0, -2.0),
            weight_hardness=0.6,
            weight_formation_energy=0.4,
        )
        
        # Create ranker with custom application
        ranker = ApplicationRanker(applications={'custom_armor': custom_app})
        
        # Create config for custom application
        config = ScreeningConfig(
            stability_threshold=0.1,
            require_viable=True,
            stability_weight=0.5,
            performance_weight=0.3,
            cost_weight=0.0,
            enable_application_ranking=True,
            application_ranking_weight=0.2,
            selected_applications=['custom_armor'],
        )
        
        engine = ScreeningEngine(
            stability_analyzer=stability_analyzer,
            config=config,
            application_ranker=ranker
        )
        
        results = engine.screen_candidates(
            sample_candidates_with_properties,
            "test_custom_app"
        )
        
        # Should have rankings for custom application
        assert results.application_rankings is not None
        assert 'custom_armor' in results.application_rankings
        
        # Candidates should have custom application scores
        for candidate in results.ranked_candidates:
            if candidate.application_scores:
                assert 'custom_armor' in candidate.application_scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
