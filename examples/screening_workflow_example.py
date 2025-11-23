"""Example of using the high-throughput screening system."""

from ceramic_discovery.screening import (
    WorkflowOrchestrator,
    ScreeningEngine,
    MaterialCandidate,
    ScreeningConfig,
    RankingCriterion,
)
from ceramic_discovery.dft.stability_analyzer import StabilityAnalyzer


def main():
    """Demonstrate high-throughput screening workflow."""
    
    print("=" * 80)
    print("Ceramic Armor Discovery: High-Throughput Screening Example")
    print("=" * 80)
    
    # Step 1: Create stability analyzer
    print("\n1. Initializing stability analyzer...")
    stability_analyzer = StabilityAnalyzer(metastable_threshold=0.1)
    
    # Step 2: Configure screening
    print("2. Configuring screening parameters...")
    config = ScreeningConfig(
        stability_threshold=0.1,
        require_viable=True,
        stability_weight=0.4,
        performance_weight=0.4,
        cost_weight=0.2,
        batch_size=50,
    )
    
    # Step 3: Create screening engine
    print("3. Creating screening engine...")
    engine = ScreeningEngine(
        stability_analyzer=stability_analyzer,
        config=config
    )
    
    # Step 4: Create sample candidates
    print("4. Generating material candidates...")
    candidates = []
    
    # SiC-based candidates
    for dopant, conc, e_hull, e_form in [
        ("Y", 0.01, 0.05, -1.5),
        ("Zr", 0.02, 0.08, -1.3),
        ("Ti", 0.03, 0.15, -1.1),  # Unstable
        ("Al", 0.01, 0.03, -1.7),
    ]:
        candidates.append(
            MaterialCandidate(
                material_id=f"SiC_{dopant}_{int(conc*100)}",
                formula=f"SiC:{dopant}({conc*100:.0f}%)",
                base_ceramic="SiC",
                dopant_element=dopant,
                dopant_concentration=conc,
                energy_above_hull=e_hull,
                formation_energy=e_form,
            )
        )
    
    # B4C-based candidates
    for dopant, conc, e_hull, e_form in [
        ("Ti", 0.02, 0.04, -1.6),
        ("Cr", 0.03, 0.09, -1.4),
        ("V", 0.01, 0.12, -1.2),  # Unstable
    ]:
        candidates.append(
            MaterialCandidate(
                material_id=f"B4C_{dopant}_{int(conc*100)}",
                formula=f"B₄C:{dopant}({conc*100:.0f}%)",
                base_ceramic="B4C",
                dopant_element=dopant,
                dopant_concentration=conc,
                energy_above_hull=e_hull,
                formation_energy=e_form,
            )
        )
    
    print(f"   Generated {len(candidates)} candidates")
    
    # Step 5: Run screening
    print("\n5. Running high-throughput screening...")
    results = engine.screen_candidates(candidates, screening_id="example_screening")
    
    print(f"   Total candidates: {results.total_candidates}")
    print(f"   Viable candidates: {results.viable_candidates}")
    print(f"   Filtered out: {results.total_candidates - results.viable_candidates}")
    
    # Step 6: Display results
    print("\n6. Top-ranked candidates:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Formula':<20} {'Stability':<12} {'Performance':<12} {'Cost':<12} {'Combined':<12}")
    print("-" * 80)
    
    for i, candidate in enumerate(results.ranked_candidates[:5], 1):
        print(
            f"{i:<6} "
            f"{candidate.formula:<20} "
            f"{candidate.stability_score:<12.3f} "
            f"{candidate.performance_score:<12.3f} "
            f"{candidate.cost_efficiency_score:<12.3f} "
            f"{candidate.combined_score:<12.3f}"
        )
    
    # Step 7: Show statistics
    print("\n7. Screening statistics:")
    print(f"   Average stability score: {results.avg_stability_score:.3f}")
    print(f"   Average performance score: {results.avg_performance_score:.3f}")
    print(f"   Average cost score: {results.avg_cost_score:.3f}")
    
    # Step 8: Demonstrate workflow orchestration
    print("\n8. Demonstrating workflow orchestration...")
    orchestrator = WorkflowOrchestrator(
        workflow_id="screening_workflow",
        max_parallel_jobs=4
    )
    
    # Register screening task
    def screening_task(candidates_batch):
        return engine.screen_candidates(candidates_batch, "batch_screening")
    
    orchestrator.register_task("screen_batch", screening_task)
    
    # Create workflow tasks
    tasks = [
        {
            'name': 'screen_batch',
            'args': [candidates[:3]],
        },
        {
            'name': 'screen_batch',
            'args': [candidates[3:]],
        },
    ]
    
    # Execute workflow
    progress = orchestrator.execute_workflow(tasks, use_parallel=False)
    
    print(f"   Workflow status: {progress.status.value}")
    print(f"   Completed tasks: {progress.completed_tasks}/{progress.total_tasks}")
    print(f"   Progress: {progress.progress_percentage:.1f}%")
    
    # Step 9: Export results
    print("\n9. Exporting results...")
    df = results.to_dataframe()
    print(f"   Results DataFrame shape: {df.shape}")
    print(f"   Columns: {', '.join(df.columns[:5])}...")
    
    print("\n" + "=" * 80)
    print("Screening workflow completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
