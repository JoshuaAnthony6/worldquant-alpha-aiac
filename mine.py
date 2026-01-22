#!/usr/bin/env python3
"""
Production Alpha Mining Script

Daily mining script targeting 1000 alpha evaluations per run.
Designed for automated execution with:
- Multiple search strategies (beam, MCTS, template-based)
- Optimization chain for weak alphas
- Comprehensive logging and reporting
- Submittable alpha export

Usage:
    # Standard run (1000 alphas)
    python mine.py --target 1000
    
    # Quick test run
    python mine.py --target 100 --quick
    
    # Focus on specific hypothesis
    python mine.py --hypothesis "momentum reversal in growth stocks"

Reference: Plan - Daily 1000-alpha mining runs
"""

import os
import sys
import argparse
import logging
import time
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"mine_{date.today().isoformat()}.log")
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    import ace_lib as ace
    HAS_ACE = True
except ImportError:
    logger.warning("ace_lib not available")
    HAS_ACE = False
    ace = None

from rl_alpha.grammar import (
    ExpressionGrammar,
    OPERATORS,
    create_default_operands,
    build_expression_from_ast
)
from rl_alpha.search import BeamSearchEngine, create_search_engine
from rl_alpha.logger import AlphaLogger, create_logger
from alpha_scoring import (
    calculate_alpha_score,
    AlphaPool,
    extract_alpha_metrics,
    is_submission_ready,
    rank_alphas
)
from optimization_chain import (
    generate_local_rewrites,
    generate_settings_variants,
    optimize_alpha,
    batch_optimize
)
from template_renderer import get_template_prompt_reference, render_expression


@dataclass
class MiningConfig:
    """Configuration for mining run."""
    target_alphas: int = 1000
    batch_size: int = 50
    region: str = "USA"
    universe: str = "TOP3000"
    dataset: str = "model110"
    delay: int = 1
    test_period: str = "P2Y"
    
    # Search config
    search_method: str = "beam"
    beam_width: int = 50
    
    # Optimization config
    enable_optimization: bool = True
    optimization_budget: int = 200  # Max optimization evaluations
    
    # Hypothesis
    hypothesis: str = "营收增长持续上升且波动性较低的股票表现优于市场"
    
    # Output
    output_dir: str = "mining_output"
    export_submittable: bool = True


@dataclass
class MiningStats:
    """Statistics from a mining run."""
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0
    
    # Generation stats
    total_generated: int = 0
    successful_simulations: int = 0
    failed_simulations: int = 0
    
    # Quality stats
    avg_score: float = 0.0
    max_score: float = 0.0
    submittable_count: int = 0
    
    # Best alpha
    best_expression: str = ""
    best_sharpe: float = 0.0
    best_fitness: float = 0.0
    
    # Optimization stats
    optimizations_attempted: int = 0
    optimizations_improved: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AlphaMiner:
    """
    Production alpha mining engine.
    
    Combines multiple strategies:
    1. Template-based generation with LLM
    2. Search-based generation (beam/MCTS)
    3. Optimization chain for weak alphas
    """
    
    def __init__(
        self,
        brain_session,
        config: MiningConfig,
        logger: AlphaLogger = None
    ):
        """
        Initialize miner.
        
        Args:
            brain_session: Brain API session
            config: Mining configuration
            logger: Alpha logger instance
        """
        self.session = brain_session
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.alpha_pool = AlphaPool(max_size=2000)
        self.stats = MiningStats(start_time=datetime.now().isoformat())
        
        # Load field list
        self.field_list = self._load_fields()
        
        # Initialize grammar and search
        self.operands = create_default_operands(self.field_list, config.dataset)
        self.grammar = ExpressionGrammar(operators=OPERATORS, operands=self.operands)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_fields(self) -> List[str]:
        """Load available fields from Brain API."""
        if self.session is None:
            # Mock field list
            return [f"mdl110_{name}" for name in [
                "growth", "value", "quality", "momentum", "volatility",
                "size", "leverage", "profitability", "investment", "trading"
            ]]
        
        try:
            fields_df = ace.get_datafields(
                self.session,
                dataset_id=self.config.dataset,
                region=self.config.region,
                universe=self.config.universe
            )
            
            # Sort by alpha count (proxy for usefulness)
            if 'alphaCount' in fields_df.columns:
                fields_df = fields_df.sort_values('alphaCount', ascending=False)
            
            return fields_df['id'].tolist()
        except Exception as e:
            logger.error(f"Failed to load fields: {e}")
            return []
    
    def run(self) -> MiningStats:
        """
        Execute mining run.
        
        Returns:
            Mining statistics
        """
        start_time = time.time()
        
        logger.info(f"{'='*60}")
        logger.info(f"Starting Alpha Mining Run")
        logger.info(f"Target: {self.config.target_alphas} alphas")
        logger.info(f"Region: {self.config.region}, Universe: {self.config.universe}")
        logger.info(f"{'='*60}")
        
        all_alphas = []
        
        try:
            # Phase 1: Template-based generation
            logger.info("\n[Phase 1] Template-based Generation")
            template_alphas = self._generate_from_templates(
                num_alphas=self.config.target_alphas // 3
            )
            all_alphas.extend(template_alphas)
            logger.info(f"  Generated {len(template_alphas)} template-based alphas")
            
            # Phase 2: Search-based generation
            logger.info("\n[Phase 2] Search-based Generation")
            search_alphas = self._generate_from_search(
                num_alphas=self.config.target_alphas // 3
            )
            all_alphas.extend(search_alphas)
            logger.info(f"  Generated {len(search_alphas)} search-based alphas")
            
            # Phase 3: Direct LLM generation
            logger.info("\n[Phase 3] LLM Direct Generation")
            llm_alphas = self._generate_from_llm(
                num_alphas=self.config.target_alphas // 3
            )
            all_alphas.extend(llm_alphas)
            logger.info(f"  Generated {len(llm_alphas)} LLM-generated alphas")
            
            # Phase 4: Simulate all candidates
            logger.info("\n[Phase 4] Batch Simulation")
            simulated_alphas = self._simulate_batch(all_alphas)
            
            # Phase 5: Optimization
            if self.config.enable_optimization:
                logger.info("\n[Phase 5] Optimization Chain")
                simulated_alphas = self._run_optimization(simulated_alphas)
            
            # Phase 6: Final ranking and export
            logger.info("\n[Phase 6] Ranking and Export")
            self._finalize(simulated_alphas)
            
        except Exception as e:
            logger.error(f"Mining failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Finalize stats
            self.stats.end_time = datetime.now().isoformat()
            self.stats.duration_seconds = time.time() - start_time
            
            # Generate report
            self._generate_report()
        
        return self.stats
    
    def _generate_from_templates(self, num_alphas: int) -> List[Dict]:
        """Generate alphas using template bank."""
        from test import generate_alpha_with_templates, get_operators_reference
        
        alphas = []
        
        # Get template reference
        template_ref = get_template_prompt_reference()
        
        # Field whitelist (top performing fields)
        field_whitelist = "\n".join([f"  - {f}" for f in self.field_list[:15]])
        
        # Get operators reference
        operators_ref = ""
        if self.session:
            try:
                operators_ref = get_operators_reference(self.session)
            except:
                pass
        
        # Generate in batches
        batch_size = min(10, num_alphas)
        num_batches = (num_alphas + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            try:
                batch_alphas = generate_alpha_with_templates(
                    hypothesis=self.config.hypothesis,
                    template_reference=template_ref,
                    field_whitelist=field_whitelist,
                    operators_reference=operators_ref,
                    num_alphas=batch_size
                )
                alphas.extend(batch_alphas)
                self.stats.total_generated += len(batch_alphas)
                
                if len(alphas) >= num_alphas:
                    break
                    
            except Exception as e:
                logger.warning(f"Template generation batch {i+1} failed: {e}")
        
        return alphas[:num_alphas]
    
    def _generate_from_search(self, num_alphas: int) -> List[Dict]:
        """Generate alphas using beam/MCTS search."""
        alphas = []
        
        # Create search engine
        search_engine = create_search_engine(
            grammar=self.grammar,
            method=self.config.search_method,
            beam_width=self.config.beam_width
        )
        
        # Run search
        try:
            candidates = search_engine.search(num_candidates=num_alphas)
            
            for ast, expression, score in candidates:
                if expression:
                    alphas.append({
                        'alpha_expression': expression,
                        'economic_rationale': f'Search-generated (score: {score:.3f})',
                        'search_score': score,
                        'ast_hash': ast.hash() if ast else ''
                    })
                    self.stats.total_generated += 1
                    
        except Exception as e:
            logger.warning(f"Search generation failed: {e}")
        
        return alphas
    
    def _generate_from_llm(self, num_alphas: int) -> List[Dict]:
        """Generate alphas using direct LLM generation."""
        from test import generate_alpha_expressions, get_operators_reference, get_dataset_reference
        
        alphas = []
        
        # Get references
        operators_ref = ""
        dataset_ref = ""
        
        if self.session:
            try:
                operators_ref = get_operators_reference(self.session)
                dataset_ref = get_dataset_reference(
                    self.session, 
                    self.config.dataset,
                    self.config.region,
                    self.config.universe
                )
            except Exception as e:
                logger.warning(f"Failed to get references: {e}")
        
        # Generate in batches
        batch_size = min(10, num_alphas)
        num_batches = (num_alphas + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            try:
                result = generate_alpha_expressions(
                    hypothesis=self.config.hypothesis,
                    operators_reference=operators_ref,
                    dataset_reference=dataset_ref,
                    num_alphas=batch_size
                )
                
                for alpha in result.alphas:
                    alphas.append({
                        'alpha_expression': alpha.alpha_expression,
                        'economic_rationale': alpha.economic_rationale,
                        'data_fields_used': alpha.data_fields_used,
                        'operators_used': alpha.operators_used
                    })
                    self.stats.total_generated += 1
                
                if len(alphas) >= num_alphas:
                    break
                    
            except Exception as e:
                logger.warning(f"LLM generation batch {i+1} failed: {e}")
        
        return alphas[:num_alphas]
    
    def _simulate_batch(self, alphas: List[Dict]) -> List[Dict]:
        """Simulate all alphas in batch."""
        if not self.session or not alphas:
            return alphas
        
        logger.info(f"  Simulating {len(alphas)} alphas...")
        
        # Prepare configs
        configs = []
        for alpha in alphas:
            expr = alpha.get('alpha_expression', alpha.get('expression', ''))
            if not expr:
                continue
            
            config = ace.generate_alpha(
                regular=expr,
                alpha_type="REGULAR",
                region=self.config.region,
                universe=self.config.universe,
                delay=self.config.delay,
                neutralization="INDUSTRY",
                decay=4,
                truncation=0.02,
                pasteurization="ON",
                test_period=self.config.test_period,
                unit_handling="VERIFY",
                nan_handling="ON",
                visualization=False
            )
            configs.append((config, alpha))
        
        # Batch simulate
        try:
            results = ace.simulate_alpha_list_multi(
                self.session,
                [c[0] for c in configs],
                limit_of_concurrent_simulations=5,
                limit_of_multi_simulations=5
            )
            
            # Map results back
            scores = []
            for res, (_, alpha) in zip(results, configs):
                if res.get('alpha_id'):
                    alpha['simulation'] = res
                    alpha['simulation_status'] = 'success'
                    self.stats.successful_simulations += 1
                    
                    # Calculate score
                    score = calculate_alpha_score(res)
                    alpha['score'] = score
                    scores.append(score)
                    
                    # Track best
                    metrics = extract_alpha_metrics(res)
                    if score > self.stats.max_score:
                        self.stats.max_score = score
                        self.stats.best_expression = alpha.get('alpha_expression', '')
                        self.stats.best_sharpe = metrics.train_sharpe
                        self.stats.best_fitness = metrics.train_fitness
                    
                    # Check submittable
                    if is_submission_ready(metrics):
                        self.stats.submittable_count += 1
                    
                    # Log to alpha pool
                    self.alpha_pool.add(
                        alpha.get('alpha_expression', ''),
                        ast_hash=alpha.get('ast_hash', ''),
                        score=score,
                        metrics=metrics
                    )
                    
                    # Log to file
                    if self.logger:
                        self.logger.log_alpha(
                            expression=alpha.get('alpha_expression', ''),
                            sim_result=res,
                            metrics=metrics
                        )
                else:
                    alpha['simulation_status'] = 'failed'
                    self.stats.failed_simulations += 1
            
            if scores:
                self.stats.avg_score = sum(scores) / len(scores)
            
            logger.info(f"  Success: {self.stats.successful_simulations}, "
                       f"Failed: {self.stats.failed_simulations}")
            logger.info(f"  Avg score: {self.stats.avg_score:.4f}, "
                       f"Max: {self.stats.max_score:.4f}")
            
        except Exception as e:
            logger.error(f"Batch simulation failed: {e}")
        
        return [a for a in alphas if a.get('simulation_status') == 'success']
    
    def _run_optimization(self, alphas: List[Dict]) -> List[Dict]:
        """Run optimization chain on weak alphas."""
        from alpha_scoring import should_optimize
        
        optimization_budget = self.config.optimization_budget
        optimized_alphas = []
        
        for alpha in alphas:
            sim = alpha.get('simulation', {})
            if not sim:
                optimized_alphas.append(alpha)
                continue
            
            should_opt, reason = should_optimize(sim)
            
            if should_opt and optimization_budget > 0:
                self.stats.optimizations_attempted += 1
                
                try:
                    result = optimize_alpha(
                        expression=alpha.get('alpha_expression', ''),
                        sim_result=sim,
                        brain_session=self.session,
                        region=self.config.region,
                        universe=self.config.universe,
                        max_variants=20,
                        max_eval=10,
                        use_llm=True
                    )
                    
                    optimization_budget -= 10  # Rough estimate of evals used
                    
                    if result.improvement > 0:
                        self.stats.optimizations_improved += 1
                        
                        # Update alpha with optimized version
                        alpha['alpha_expression'] = result.optimized_expression
                        alpha['optimization'] = {
                            'original': result.original_expression,
                            'change_type': result.change_type,
                            'improvement': result.improvement
                        }
                        
                        if result.simulation_result:
                            alpha['simulation'] = result.simulation_result
                            alpha['score'] = result.optimized_score
                            
                except Exception as e:
                    logger.warning(f"Optimization failed: {e}")
            
            optimized_alphas.append(alpha)
        
        logger.info(f"  Optimized: {self.stats.optimizations_attempted}, "
                   f"Improved: {self.stats.optimizations_improved}")
        
        return optimized_alphas
    
    def _finalize(self, alphas: List[Dict]):
        """Rank and export final alphas."""
        # Rank alphas
        ranked = rank_alphas(alphas, self.alpha_pool)
        
        # Log pool stats
        if self.logger:
            pool_stats = self.alpha_pool.get_stats()
            self.logger.log_pool_stats(
                pool_size=pool_stats['size'],
                avg_score=pool_stats['avg_score'],
                max_score=pool_stats['max_score'],
                submittable_count=self.stats.submittable_count
            )
        
        # Export all alphas
        output_path = Path(self.config.output_dir) / f"alphas_{date.today().isoformat()}.json"
        with open(output_path, 'w') as f:
            # Convert to serializable format
            export_data = []
            for alpha, score, metrics in ranked:
                export_data.append({
                    'alpha_expression': alpha.get('alpha_expression', ''),
                    'economic_rationale': alpha.get('economic_rationale', ''),
                    'score': score,
                    'metrics': metrics.to_dict(),
                    'simulation_status': alpha.get('simulation_status', ''),
                    'alpha_id': alpha.get('simulation', {}).get('alpha_id', '')
                })
            json.dump(export_data, f, indent=2)
        
        logger.info(f"  Exported {len(export_data)} alphas to {output_path}")
        
        # Export submittable alphas separately
        if self.config.export_submittable:
            submittable = [
                (a, s, m) for a, s, m in ranked if m.submission_ready
            ]
            
            if submittable:
                submit_path = Path(self.config.output_dir) / f"submittable_{date.today().isoformat()}.json"
                with open(submit_path, 'w') as f:
                    submit_data = [{
                        'alpha_expression': a.get('alpha_expression', ''),
                        'alpha_id': a.get('simulation', {}).get('alpha_id', ''),
                        'train_sharpe': m.train_sharpe,
                        'test_sharpe': m.test_sharpe,
                        'fitness': m.train_fitness,
                        'score': s
                    } for a, s, m in submittable]
                    json.dump(submit_data, f, indent=2)
                
                logger.info(f"  Exported {len(submittable)} submittable alphas to {submit_path}")
    
    def _generate_report(self):
        """Generate mining report."""
        report_path = Path(self.config.output_dir) / f"report_{date.today().isoformat()}.md"
        
        report = f"""# Alpha Mining Report - {date.today().isoformat()}

## Configuration
- Target: {self.config.target_alphas} alphas
- Region: {self.config.region}
- Universe: {self.config.universe}
- Dataset: {self.config.dataset}
- Hypothesis: {self.config.hypothesis[:100]}...

## Results Summary

| Metric | Value |
|--------|-------|
| Duration | {self.stats.duration_seconds:.1f} seconds |
| Total Generated | {self.stats.total_generated} |
| Successful Simulations | {self.stats.successful_simulations} |
| Failed Simulations | {self.stats.failed_simulations} |
| Average Score | {self.stats.avg_score:.4f} |
| Maximum Score | {self.stats.max_score:.4f} |
| Submittable Alphas | {self.stats.submittable_count} |

## Optimization Stats
- Attempted: {self.stats.optimizations_attempted}
- Improved: {self.stats.optimizations_improved}
- Success Rate: {self.stats.optimizations_improved / max(1, self.stats.optimizations_attempted) * 100:.1f}%

## Best Alpha
```
{self.stats.best_expression}
```
- Sharpe: {self.stats.best_sharpe:.3f}
- Fitness: {self.stats.best_fitness:.3f}

---
*Generated at {datetime.now().isoformat()}*
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        
        # Also log to alpha logger
        if self.logger:
            self.logger.generate_daily_report()


def main():
    """Main entry point for mining."""
    parser = argparse.ArgumentParser(description="Production Alpha Mining")
    
    # Target
    parser.add_argument("--target", type=int, default=1000,
                       help="Target number of alphas")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Simulation batch size")
    
    # Environment
    parser.add_argument("--region", type=str, default="USA",
                       help="Market region")
    parser.add_argument("--universe", type=str, default="TOP3000",
                       help="Stock universe")
    parser.add_argument("--dataset", type=str, default="model110",
                       help="Dataset ID")
    
    # Strategy
    parser.add_argument("--hypothesis", type=str,
                       default="营收增长持续上升且波动性较低的股票表现优于市场",
                       help="Research hypothesis")
    parser.add_argument("--method", type=str, default="beam",
                       choices=["beam", "mcts"],
                       help="Search method")
    
    # Optimization
    parser.add_argument("--no-optimize", action="store_true",
                       help="Disable optimization chain")
    parser.add_argument("--opt-budget", type=int, default=200,
                       help="Optimization evaluation budget")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="mining_output",
                       help="Output directory")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Log directory")
    
    # Mode
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode (fewer alphas)")
    parser.add_argument("--mock", action="store_true",
                       help="Mock mode without Brain API")
    
    args = parser.parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.target = min(100, args.target)
        args.opt_budget = min(50, args.opt_budget)
    
    # Create config
    config = MiningConfig(
        target_alphas=args.target,
        batch_size=args.batch_size,
        region=args.region,
        universe=args.universe,
        dataset=args.dataset,
        hypothesis=args.hypothesis,
        search_method=args.method,
        enable_optimization=not args.no_optimize,
        optimization_budget=args.opt_budget,
        output_dir=args.output_dir
    )
    
    # Initialize Brain session
    brain_session = None
    
    if not args.mock and HAS_ACE:
        try:
            brain_session = ace.start_session()
            if ace.check_session_timeout(brain_session) < 3000:
                brain_session = ace.check_session_and_relogin(brain_session)
            logger.info("Brain session initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Brain session: {e}")
            logger.info("Running in mock mode")
    
    # Initialize logger
    alpha_logger = create_logger(
        log_dir=args.log_dir,
        experiment_name=f"mine_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run mining
    try:
        miner = AlphaMiner(
            brain_session=brain_session,
            config=config,
            logger=alpha_logger
        )
        
        stats = miner.run()
        
        # Print summary
        print(f"\n{'='*60}")
        print("Mining Complete!")
        print(f"{'='*60}")
        print(f"Total Generated: {stats.total_generated}")
        print(f"Successful: {stats.successful_simulations}")
        print(f"Submittable: {stats.submittable_count}")
        print(f"Best Score: {stats.max_score:.4f}")
        print(f"Duration: {stats.duration_seconds:.1f}s")
        print(f"{'='*60}")
        
    finally:
        alpha_logger.close()


if __name__ == "__main__":
    main()
