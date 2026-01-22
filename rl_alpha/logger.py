"""
Structured Logging System for Alpha Mining

Implements comprehensive logging for:
- Episode logs: MCTS/Beam search episodes (JSONL)
- Alpha logs: Generated alphas with metrics (SQLite)
- Training logs: Loss, entropy, value estimates (TensorBoard)
- Pool logs: Alpha pool correlations, submission status (CSV)
- Daily reports: Summary markdown files

Reference: Plan Section 4 - Logging System Design
"""

import os
import json
import sqlite3
import csv
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    logger.warning("TensorBoard not available, training logs will be limited")


@dataclass
class EpisodeLog:
    """Log entry for a single search episode."""
    episode_id: int
    timestamp: str
    num_steps: int
    final_expression: str
    final_reward: float
    is_complete: bool
    search_method: str  # 'mcts', 'beam', 'hybrid'
    actions_taken: List[str]
    intermediate_rewards: List[float]
    duration_ms: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AlphaLogEntry:
    """Log entry for a generated alpha."""
    alpha_id: str
    expression: str
    ast_hash: str
    timestamp: str
    
    # Performance metrics
    train_sharpe: float
    test_sharpe: float
    train_fitness: float
    test_fitness: float
    turnover: float
    
    # Adjusted metrics
    risk_neutral_sharpe: float
    invest_constrained_sharpe: float
    
    # Correlation metrics
    prod_correlation: float
    pool_correlation: float
    
    # Test results
    tests_passed: int
    tests_total: int
    failed_tests: str  # JSON array string
    
    # Scores
    raw_score: float
    adjusted_score: float
    submission_ready: bool
    
    # Generation info
    search_method: str
    generation_time_ms: int
    episode_id: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingStats:
    """Training statistics for one update."""
    step: int
    timestamp: str
    
    # PPO metrics
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    clip_fraction: float
    
    # Episode stats
    avg_reward: float
    max_reward: float
    min_reward: float
    success_rate: float  # % of episodes producing valid alphas
    
    # Learning stats
    learning_rate: float
    explained_variance: float


class AlphaLogger:
    """
    Comprehensive logging system for alpha mining.
    
    Manages multiple log formats:
    - JSONL files for episode logs
    - SQLite database for alpha logs
    - TensorBoard for training metrics
    - CSV for pool statistics
    - Markdown for daily reports
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = None,
        enable_tensorboard: bool = True,
        db_name: str = "alphas.db"
    ):
        """
        Initialize logging system.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name for this experiment run
            enable_tensorboard: Whether to enable TensorBoard logging
            db_name: Name of SQLite database file
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize database
        self.db_path = self.log_dir / db_name
        self._init_database()
        
        # Initialize episode log file
        self.episode_log_path = self.episodes_dir / f"{self.experiment_name}.jsonl"
        
        # Initialize TensorBoard writer
        self.tb_writer = None
        if enable_tensorboard and HAS_TENSORBOARD:
            self.tb_writer = SummaryWriter(
                log_dir=str(self.runs_dir / self.experiment_name)
            )
        
        # Thread-safe counters
        self._lock = threading.Lock()
        self._episode_count = 0
        self._alpha_count = 0
        self._step_count = 0
        
        # Daily stats
        self._daily_stats = {
            'alphas_generated': 0,
            'alphas_valid': 0,
            'alphas_submittable': 0,
            'avg_score': 0.0,
            'max_score': 0.0,
            'total_episodes': 0
        }
        
        logger.info(f"AlphaLogger initialized: {self.log_dir}")
    
    def _setup_directories(self):
        """Create log directory structure."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.runs_dir = self.log_dir / "runs"  # TensorBoard
        self.episodes_dir = self.log_dir / "episodes"  # JSONL
        self.reports_dir = self.log_dir / "daily_reports"  # Markdown
        
        for d in [self.runs_dir, self.episodes_dir, self.reports_dir]:
            d.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Alphas table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alphas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alpha_id TEXT UNIQUE,
                    expression TEXT NOT NULL,
                    ast_hash TEXT,
                    timestamp TEXT NOT NULL,
                    
                    train_sharpe REAL,
                    test_sharpe REAL,
                    train_fitness REAL,
                    test_fitness REAL,
                    turnover REAL,
                    
                    risk_neutral_sharpe REAL,
                    invest_constrained_sharpe REAL,
                    
                    prod_correlation REAL,
                    pool_correlation REAL,
                    
                    tests_passed INTEGER,
                    tests_total INTEGER,
                    failed_tests TEXT,
                    
                    raw_score REAL,
                    adjusted_score REAL,
                    submission_ready INTEGER,
                    
                    search_method TEXT,
                    generation_time_ms INTEGER,
                    episode_id INTEGER,
                    
                    UNIQUE(expression)
                )
            """)
            
            # Pool history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pool_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pool_size INTEGER,
                    avg_score REAL,
                    max_score REAL,
                    submittable_count INTEGER,
                    diversity_index REAL
                )
            """)
            
            # Training runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    total_episodes INTEGER,
                    total_alphas INTEGER,
                    best_score REAL,
                    config TEXT
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alphas_score ON alphas(raw_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alphas_timestamp ON alphas(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alphas_submittable ON alphas(submission_ready)")
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get thread-safe database connection."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()
    
    # ========================================================================
    # Episode Logging
    # ========================================================================
    
    def log_episode(
        self,
        expression: str,
        reward: float,
        num_steps: int,
        is_complete: bool,
        search_method: str = "beam",
        actions: List[str] = None,
        intermediate_rewards: List[float] = None,
        duration_ms: int = 0
    ):
        """
        Log a search episode.
        
        Args:
            expression: Final generated expression
            reward: Final reward
            num_steps: Number of steps taken
            is_complete: Whether expression is complete
            search_method: Search method used
            actions: List of action strings
            intermediate_rewards: Rewards at each step
            duration_ms: Episode duration in milliseconds
        """
        with self._lock:
            self._episode_count += 1
            episode_id = self._episode_count
        
        entry = EpisodeLog(
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            num_steps=num_steps,
            final_expression=expression,
            final_reward=reward,
            is_complete=is_complete,
            search_method=search_method,
            actions_taken=actions or [],
            intermediate_rewards=intermediate_rewards or [],
            duration_ms=duration_ms
        )
        
        # Append to JSONL file
        with open(self.episode_log_path, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
        
        # Update daily stats
        with self._lock:
            self._daily_stats['total_episodes'] += 1
        
        return episode_id
    
    # ========================================================================
    # Alpha Logging
    # ========================================================================
    
    def log_alpha(
        self,
        expression: str,
        sim_result: Dict,
        metrics = None,
        ast_hash: str = None,
        search_method: str = "beam",
        generation_time_ms: int = 0,
        episode_id: int = None
    ) -> Optional[int]:
        """
        Log a generated alpha to database.
        
        Args:
            expression: Alpha expression
            sim_result: Simulation result from Brain API
            metrics: Pre-computed AlphaMetrics (optional)
            ast_hash: AST hash for deduplication
            search_method: Search method used
            generation_time_ms: Generation time
            episode_id: Associated episode ID
            
        Returns:
            Database row ID or None if duplicate
        """
        from alpha_scoring import extract_alpha_metrics
        
        if metrics is None:
            metrics = extract_alpha_metrics(sim_result, expression)
        
        entry = AlphaLogEntry(
            alpha_id=sim_result.get('alpha_id', ''),
            expression=expression,
            ast_hash=ast_hash or '',
            timestamp=datetime.now().isoformat(),
            train_sharpe=metrics.train_sharpe,
            test_sharpe=metrics.test_sharpe,
            train_fitness=metrics.train_fitness,
            test_fitness=metrics.test_fitness,
            turnover=metrics.turnover,
            risk_neutral_sharpe=metrics.risk_neutral_sharpe,
            invest_constrained_sharpe=metrics.invest_constrained_sharpe,
            prod_correlation=metrics.prod_correlation,
            pool_correlation=metrics.pool_correlation,
            tests_passed=metrics.tests_passed,
            tests_total=metrics.tests_total,
            failed_tests=json.dumps(metrics.failed_tests),
            raw_score=metrics.raw_score,
            adjusted_score=metrics.adjusted_score,
            submission_ready=metrics.submission_ready,
            search_method=search_method,
            generation_time_ms=generation_time_ms,
            episode_id=episode_id
        )
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO alphas (
                        alpha_id, expression, ast_hash, timestamp,
                        train_sharpe, test_sharpe, train_fitness, test_fitness, turnover,
                        risk_neutral_sharpe, invest_constrained_sharpe,
                        prod_correlation, pool_correlation,
                        tests_passed, tests_total, failed_tests,
                        raw_score, adjusted_score, submission_ready,
                        search_method, generation_time_ms, episode_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.alpha_id, entry.expression, entry.ast_hash, entry.timestamp,
                    entry.train_sharpe, entry.test_sharpe, entry.train_fitness,
                    entry.test_fitness, entry.turnover,
                    entry.risk_neutral_sharpe, entry.invest_constrained_sharpe,
                    entry.prod_correlation, entry.pool_correlation,
                    entry.tests_passed, entry.tests_total, entry.failed_tests,
                    entry.raw_score, entry.adjusted_score, int(entry.submission_ready),
                    entry.search_method, entry.generation_time_ms, entry.episode_id
                ))
                conn.commit()
                
                if cursor.rowcount > 0:
                    row_id = cursor.lastrowid
                    
                    # Update daily stats
                    with self._lock:
                        self._alpha_count += 1
                        self._daily_stats['alphas_generated'] += 1
                        if entry.raw_score > 0:
                            self._daily_stats['alphas_valid'] += 1
                        if entry.submission_ready:
                            self._daily_stats['alphas_submittable'] += 1
                        
                        # Update running average
                        n = self._daily_stats['alphas_generated']
                        self._daily_stats['avg_score'] = (
                            (self._daily_stats['avg_score'] * (n-1) + entry.raw_score) / n
                        )
                        self._daily_stats['max_score'] = max(
                            self._daily_stats['max_score'], entry.raw_score
                        )
                    
                    return row_id
                return None
                
        except Exception as e:
            logger.error(f"Failed to log alpha: {e}")
            return None
    
    # ========================================================================
    # Training Logging
    # ========================================================================
    
    def log_training_step(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        kl_divergence: float = 0.0,
        clip_fraction: float = 0.0,
        avg_reward: float = 0.0,
        max_reward: float = 0.0,
        min_reward: float = 0.0,
        success_rate: float = 0.0,
        learning_rate: float = 0.0,
        explained_variance: float = 0.0,
        **extra_metrics
    ):
        """
        Log training metrics to TensorBoard.
        
        Args:
            step: Training step number
            policy_loss: PPO policy loss
            value_loss: PPO value loss
            entropy: Policy entropy
            kl_divergence: KL divergence from old policy
            clip_fraction: Fraction of clipped gradients
            avg_reward: Average episode reward
            max_reward: Maximum episode reward
            min_reward: Minimum episode reward
            success_rate: Success rate (valid alphas / total)
            learning_rate: Current learning rate
            explained_variance: Value function explained variance
            **extra_metrics: Additional metrics to log
        """
        with self._lock:
            self._step_count = step
        
        if self.tb_writer:
            # Loss metrics
            self.tb_writer.add_scalar('Loss/policy', policy_loss, step)
            self.tb_writer.add_scalar('Loss/value', value_loss, step)
            self.tb_writer.add_scalar('Loss/entropy', entropy, step)
            
            # PPO metrics
            self.tb_writer.add_scalar('PPO/kl_divergence', kl_divergence, step)
            self.tb_writer.add_scalar('PPO/clip_fraction', clip_fraction, step)
            
            # Reward metrics
            self.tb_writer.add_scalar('Reward/avg', avg_reward, step)
            self.tb_writer.add_scalar('Reward/max', max_reward, step)
            self.tb_writer.add_scalar('Reward/min', min_reward, step)
            
            # Performance metrics
            self.tb_writer.add_scalar('Performance/success_rate', success_rate, step)
            self.tb_writer.add_scalar('Performance/explained_variance', explained_variance, step)
            
            # Learning metrics
            self.tb_writer.add_scalar('Learning/learning_rate', learning_rate, step)
            
            # Extra metrics
            for name, value in extra_metrics.items():
                self.tb_writer.add_scalar(f'Extra/{name}', value, step)
    
    def log_alpha_histogram(
        self,
        step: int,
        scores: List[float],
        name: str = "alpha_scores"
    ):
        """Log histogram of alpha scores."""
        if self.tb_writer and scores:
            import numpy as np
            self.tb_writer.add_histogram(name, np.array(scores), step)
    
    # ========================================================================
    # Pool Logging
    # ========================================================================
    
    def log_pool_stats(
        self,
        pool_size: int,
        avg_score: float,
        max_score: float,
        submittable_count: int,
        diversity_index: float = 0.0
    ):
        """Log alpha pool statistics."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pool_history (
                    timestamp, pool_size, avg_score, max_score,
                    submittable_count, diversity_index
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                pool_size, avg_score, max_score,
                submittable_count, diversity_index
            ))
            conn.commit()
        
        if self.tb_writer:
            step = self._step_count
            self.tb_writer.add_scalar('Pool/size', pool_size, step)
            self.tb_writer.add_scalar('Pool/avg_score', avg_score, step)
            self.tb_writer.add_scalar('Pool/max_score', max_score, step)
            self.tb_writer.add_scalar('Pool/submittable', submittable_count, step)
            self.tb_writer.add_scalar('Pool/diversity', diversity_index, step)
    
    # ========================================================================
    # Reporting
    # ========================================================================
    
    def generate_daily_report(self) -> str:
        """
        Generate daily summary report in Markdown format.
        
        Returns:
            Path to generated report file
        """
        today = date.today().isoformat()
        report_path = self.reports_dir / f"report_{today}.md"
        
        # Query database for today's alphas
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get today's stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN raw_score > 0 THEN 1 ELSE 0 END) as valid,
                    SUM(CASE WHEN submission_ready = 1 THEN 1 ELSE 0 END) as submittable,
                    AVG(raw_score) as avg_score,
                    MAX(raw_score) as max_score,
                    AVG(train_sharpe) as avg_sharpe,
                    MAX(train_sharpe) as max_sharpe
                FROM alphas
                WHERE DATE(timestamp) = DATE('now')
            """)
            stats = cursor.fetchone()
            
            # Get top alphas
            cursor.execute("""
                SELECT expression, train_sharpe, test_sharpe, raw_score, submission_ready
                FROM alphas
                WHERE DATE(timestamp) = DATE('now')
                ORDER BY raw_score DESC
                LIMIT 10
            """)
            top_alphas = cursor.fetchall()
            
            # Get failed test distribution
            cursor.execute("""
                SELECT failed_tests
                FROM alphas
                WHERE DATE(timestamp) = DATE('now') AND failed_tests != '[]'
            """)
            failed_tests_data = cursor.fetchall()
        
        # Count failed tests
        failed_test_counts = {}
        for (failed_json,) in failed_tests_data:
            try:
                failed = json.loads(failed_json)
                for test in failed:
                    failed_test_counts[test] = failed_test_counts.get(test, 0) + 1
            except:
                pass
        
        # Generate report
        report = f"""# Alpha Mining Daily Report - {today}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Alphas Generated | {stats[0] or 0} |
| Valid Alphas (score > 0) | {stats[1] or 0} |
| Submission-Ready Alphas | {stats[2] or 0} |
| Average Score | {stats[3] or 0:.4f} |
| Maximum Score | {stats[4] or 0:.4f} |
| Average Sharpe | {stats[5] or 0:.4f} |
| Maximum Sharpe | {stats[6] or 0:.4f} |

## Top 10 Alphas

| Expression | Train Sharpe | Test Sharpe | Score | Submittable |
|------------|--------------|-------------|-------|-------------|
"""
        
        for expr, ts, os, score, sub in top_alphas:
            expr_short = expr[:60] + "..." if len(expr) > 60 else expr
            report += f"| `{expr_short}` | {ts:.3f} | {os:.3f} | {score:.3f} | {'✓' if sub else '✗'} |\n"
        
        report += "\n## Failed Test Distribution\n\n"
        if failed_test_counts:
            report += "| Test Name | Failure Count |\n|-----------|---------------|\n"
            for test, count in sorted(failed_test_counts.items(), key=lambda x: -x[1])[:10]:
                report += f"| {test} | {count} |\n"
        else:
            report += "No test failures recorded.\n"
        
        report += f"\n---\n*Generated at {datetime.now().isoformat()}*\n"
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Daily report generated: {report_path}")
        return str(report_path)
    
    # ========================================================================
    # Query Methods
    # ========================================================================
    
    def get_best_alphas(
        self,
        limit: int = 10,
        submission_ready_only: bool = False
    ) -> List[Dict]:
        """Get best alphas from database."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM alphas"
            if submission_ready_only:
                query += " WHERE submission_ready = 1"
            query += " ORDER BY raw_score DESC LIMIT ?"
            
            cursor.execute(query, (limit,))
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_alpha_count(self) -> int:
        """Get total alpha count."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alphas")
            return cursor.fetchone()[0]
    
    def get_submission_ready_count(self) -> int:
        """Get count of submission-ready alphas."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alphas WHERE submission_ready = 1")
            return cursor.fetchone()[0]
    
    def export_submittable_alphas(self, output_path: str = "submittable_alphas.json"):
        """Export submission-ready alphas to JSON file."""
        alphas = self.get_best_alphas(limit=1000, submission_ready_only=True)
        
        with open(output_path, 'w') as f:
            json.dump(alphas, f, indent=2)
        
        logger.info(f"Exported {len(alphas)} submittable alphas to {output_path}")
        return output_path
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    
    def close(self):
        """Close logging resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        # Generate final report
        try:
            self.generate_daily_report()
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
        
        logger.info(f"AlphaLogger closed. Total alphas: {self._alpha_count}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_logger(
    log_dir: str = "logs",
    experiment_name: str = None,
    **kwargs
) -> AlphaLogger:
    """
    Factory function to create configured AlphaLogger.
    
    Args:
        log_dir: Log directory path
        experiment_name: Experiment name
        **kwargs: Additional arguments for AlphaLogger
        
    Returns:
        Configured AlphaLogger instance
    """
    return AlphaLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        **kwargs
    )
