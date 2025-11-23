"""Computational budget tracking and management."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Budget:
    """Budget configuration."""
    
    name: str
    total_amount: float
    currency: str = "USD"
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    
    def is_active(self) -> bool:
        """Check if budget is currently active."""
        now = datetime.now()
        if now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        return True
    
    def days_remaining(self) -> Optional[int]:
        """Get number of days remaining in budget period."""
        if not self.end_date:
            return None
        
        now = datetime.now()
        if now > self.end_date:
            return 0
        
        return (self.end_date - now).days


@dataclass
class BudgetExpense:
    """Record of budget expense."""
    
    budget_name: str
    amount: float
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    category: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'budget_name': self.budget_name,
            'amount': self.amount,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'metadata': self.metadata,
        }


class BudgetTracker:
    """
    Computational budget tracking and management.
    
    Tracks expenses against budgets, sends alerts when thresholds
    are reached, and provides budget forecasting.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize budget tracker.
        
        Args:
            data_dir: Directory for storing budget data
        """
        self.data_dir = data_dir or Path("./data/budgets")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.budgets: Dict[str, Budget] = {}
        self.expenses: List[BudgetExpense] = []
        self._alerts_sent: Dict[str, List[float]] = {}
        
        logger.info("Initialized BudgetTracker")
    
    def create_budget(
        self,
        name: str,
        total_amount: float,
        duration_days: Optional[int] = None,
        currency: str = "USD",
        alert_thresholds: Optional[List[float]] = None
    ) -> Budget:
        """
        Create a new budget.
        
        Args:
            name: Budget name
            total_amount: Total budget amount
            duration_days: Duration in days (None = no end date)
            currency: Currency code
            alert_thresholds: List of alert thresholds (0-1)
        
        Returns:
            Created budget
        """
        start_date = datetime.now()
        end_date = None
        
        if duration_days:
            end_date = start_date + timedelta(days=duration_days)
        
        budget = Budget(
            name=name,
            total_amount=total_amount,
            currency=currency,
            start_date=start_date,
            end_date=end_date,
            alert_thresholds=alert_thresholds or [0.5, 0.75, 0.9],
        )
        
        self.budgets[name] = budget
        self._alerts_sent[name] = []
        
        logger.info(
            f"Created budget '{name}': {currency} {total_amount:.2f} "
            f"({duration_days} days)" if duration_days else ""
        )
        
        return budget
    
    def record_expense(
        self,
        budget_name: str,
        amount: float,
        description: str,
        category: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> BudgetExpense:
        """
        Record an expense against a budget.
        
        Args:
            budget_name: Name of budget
            amount: Expense amount
            description: Expense description
            category: Optional category
            metadata: Optional metadata
        
        Returns:
            Created expense record
        """
        if budget_name not in self.budgets:
            raise ValueError(f"Budget not found: {budget_name}")
        
        expense = BudgetExpense(
            budget_name=budget_name,
            amount=amount,
            description=description,
            category=category,
            metadata=metadata or {},
        )
        
        self.expenses.append(expense)
        
        logger.info(
            f"Recorded expense: {budget_name} - "
            f"{self.budgets[budget_name].currency} {amount:.2f} - {description}"
        )
        
        # Check for alerts
        self._check_budget_alerts(budget_name)
        
        return expense
    
    def get_budget_status(self, budget_name: str) -> Dict:
        """
        Get status of a budget.
        
        Args:
            budget_name: Name of budget
        
        Returns:
            Dictionary with budget status
        """
        if budget_name not in self.budgets:
            raise ValueError(f"Budget not found: {budget_name}")
        
        budget = self.budgets[budget_name]
        
        # Calculate total spent
        budget_expenses = [
            e for e in self.expenses
            if e.budget_name == budget_name
        ]
        
        total_spent = sum(e.amount for e in budget_expenses)
        remaining = budget.total_amount - total_spent
        usage_pct = total_spent / budget.total_amount if budget.total_amount > 0 else 0
        
        # Calculate daily burn rate
        if budget_expenses:
            first_expense = min(budget_expenses, key=lambda e: e.timestamp)
            days_elapsed = (datetime.now() - first_expense.timestamp).days + 1
            daily_burn_rate = total_spent / days_elapsed if days_elapsed > 0 else 0
        else:
            daily_burn_rate = 0
        
        # Forecast
        days_remaining = budget.days_remaining()
        if days_remaining and daily_burn_rate > 0:
            projected_total = total_spent + (daily_burn_rate * days_remaining)
            projected_overrun = max(0, projected_total - budget.total_amount)
        else:
            projected_total = None
            projected_overrun = None
        
        return {
            'budget_name': budget_name,
            'total_amount': budget.total_amount,
            'total_spent': total_spent,
            'remaining': remaining,
            'usage_pct': usage_pct,
            'currency': budget.currency,
            'is_active': budget.is_active(),
            'days_remaining': days_remaining,
            'daily_burn_rate': daily_burn_rate,
            'projected_total': projected_total,
            'projected_overrun': projected_overrun,
            'n_expenses': len(budget_expenses),
        }
    
    def _check_budget_alerts(self, budget_name: str) -> None:
        """Check if budget alerts should be sent."""
        status = self.get_budget_status(budget_name)
        budget = self.budgets[budget_name]
        
        usage_pct = status['usage_pct']
        
        for threshold in budget.alert_thresholds:
            if usage_pct >= threshold and threshold not in self._alerts_sent[budget_name]:
                self._send_alert(budget_name, threshold, status)
                self._alerts_sent[budget_name].append(threshold)
    
    def _send_alert(self, budget_name: str, threshold: float, status: Dict) -> None:
        """Send budget alert."""
        logger.warning(
            f"BUDGET ALERT: '{budget_name}' has reached {threshold*100:.0f}% "
            f"({status['currency']} {status['total_spent']:.2f} / "
            f"{status['total_amount']:.2f})"
        )
        
        if status['projected_overrun'] and status['projected_overrun'] > 0:
            logger.warning(
                f"  Projected overrun: {status['currency']} "
                f"{status['projected_overrun']:.2f}"
            )
    
    def get_all_budgets_status(self) -> List[Dict]:
        """Get status of all budgets."""
        return [
            self.get_budget_status(name)
            for name in self.budgets.keys()
        ]
    
    def get_expenses_by_category(self, budget_name: str) -> Dict[str, float]:
        """
        Get expenses grouped by category.
        
        Args:
            budget_name: Name of budget
        
        Returns:
            Dictionary mapping category to total amount
        """
        budget_expenses = [
            e for e in self.expenses
            if e.budget_name == budget_name
        ]
        
        by_category = {}
        for expense in budget_expenses:
            category = expense.category or 'uncategorized'
            by_category[category] = by_category.get(category, 0) + expense.amount
        
        return by_category
    
    def save_state(self) -> None:
        """Save budget tracker state to disk."""
        state_file = self.data_dir / "budget_state.json"
        
        state = {
            'budgets': {
                name: {
                    'name': budget.name,
                    'total_amount': budget.total_amount,
                    'currency': budget.currency,
                    'start_date': budget.start_date.isoformat(),
                    'end_date': budget.end_date.isoformat() if budget.end_date else None,
                    'alert_thresholds': budget.alert_thresholds,
                }
                for name, budget in self.budgets.items()
            },
            'expenses': [expense.to_dict() for expense in self.expenses],
            'alerts_sent': self._alerts_sent,
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved budget state to {state_file}")
    
    def load_state(self) -> None:
        """Load budget tracker state from disk."""
        state_file = self.data_dir / "budget_state.json"
        
        if not state_file.exists():
            logger.info("No saved state found")
            return
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Restore budgets
        for name, budget_data in state['budgets'].items():
            budget = Budget(
                name=budget_data['name'],
                total_amount=budget_data['total_amount'],
                currency=budget_data['currency'],
                start_date=datetime.fromisoformat(budget_data['start_date']),
                end_date=datetime.fromisoformat(budget_data['end_date']) if budget_data['end_date'] else None,
                alert_thresholds=budget_data['alert_thresholds'],
            )
            self.budgets[name] = budget
        
        # Restore expenses
        for expense_data in state['expenses']:
            expense = BudgetExpense(
                budget_name=expense_data['budget_name'],
                amount=expense_data['amount'],
                description=expense_data['description'],
                timestamp=datetime.fromisoformat(expense_data['timestamp']),
                category=expense_data.get('category'),
                metadata=expense_data.get('metadata', {}),
            )
            self.expenses.append(expense)
        
        # Restore alerts
        self._alerts_sent = state.get('alerts_sent', {})
        
        logger.info(f"Loaded budget state from {state_file}")
    
    def print_summary(self) -> None:
        """Print budget summary."""
        print("\n=== Budget Summary ===")
        
        for budget_name in self.budgets.keys():
            status = self.get_budget_status(budget_name)
            
            print(f"\nBudget: {budget_name}")
            print(f"  Total: {status['currency']} {status['total_amount']:.2f}")
            print(f"  Spent: {status['currency']} {status['total_spent']:.2f} "
                  f"({status['usage_pct']:.1%})")
            print(f"  Remaining: {status['currency']} {status['remaining']:.2f}")
            
            if status['days_remaining']:
                print(f"  Days Remaining: {status['days_remaining']}")
            
            if status['daily_burn_rate'] > 0:
                print(f"  Daily Burn Rate: {status['currency']} "
                      f"{status['daily_burn_rate']:.2f}")
            
            if status['projected_overrun'] and status['projected_overrun'] > 0:
                print(f"  ⚠️  Projected Overrun: {status['currency']} "
                      f"{status['projected_overrun']:.2f}")
        
        print("\n======================\n")
