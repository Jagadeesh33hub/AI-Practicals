"""
Practical 9: STRIPS / Goal-Stack Planning for Blocks World
Aim: Implement a Goal-Stack Planner using STRIPS operators (PickUp, PutDown,
     Stack, Unstack) to transform an initial Blocks World state into a goal state.
"""


class BlocksWorldPlanner:
    """
    Predicates represented as strings:
      - On(A, B)   : Block A is directly on Block B
      - OnTable(A) : Block A is on the table
      - Clear(A)   : Nothing is on top of Block A
      - Holding(A) : Robotic arm is holding Block A
      - ArmEmpty   : Robotic arm is empty
    """

    def __init__(self, initial_state: set[str], goal_state: set[str]) -> None:
        self.state = set(initial_state)
        self.goal = set(goal_state)
        self.plan: list[str] = []

    def get_achieving_action(self, predicate: str) -> tuple[str, list[str], set[str], set[str]]:
        if predicate.startswith("Holding("):
            x = predicate[8:-1]
            on_block = next((p[3:-1].split(",")[1] for p in self.state if p.startswith(f"On({x},")), None)
            if on_block:
                return (
                    f"Unstack({x}, {on_block})",
                    [f"On({x},{on_block})", f"Clear({x})", "ArmEmpty"],
                    {f"Holding({x})", f"Clear({on_block})"},
                    {f"On({x},{on_block})", f"Clear({x})", "ArmEmpty"},
                )
            return (
                f"PickUp({x})",
                [f"OnTable({x})", f"Clear({x})", "ArmEmpty"],
                {f"Holding({x})"},
                {f"OnTable({x})", f"Clear({x})", "ArmEmpty"},
            )

        if predicate.startswith("OnTable("):
            x = predicate[8:-1]
            return (
                f"PutDown({x})",
                [f"Holding({x})"],
                {f"OnTable({x})", f"Clear({x})", "ArmEmpty"},
                {f"Holding({x})"},
            )

        if predicate.startswith("On("):
            x, y = predicate[3:-1].split(",")
            return (
                f"Stack({x}, {y})",
                [f"Holding({x})", f"Clear({y})"],
                {f"On({x},{y})", f"Clear({x})", "ArmEmpty"},
                {f"Holding({x})", f"Clear({y})"},
            )

        if predicate.startswith("Clear("):
            x = predicate[6:-1]
            above = next((p[3:-1].split(",")[0] for p in self.state if p.startswith("On(") and p.endswith(f",{x})")), None)
            if above:
                return (
                    f"Unstack({above}, {x})",
                    [f"On({above},{x})", f"Clear({above})", "ArmEmpty"],
                    {f"Holding({above})", f"Clear({x})"},
                    {f"On({above},{x})", f"Clear({above})", "ArmEmpty"},
                )

        if predicate == "ArmEmpty":
            held = next((p[8:-1] for p in self.state if p.startswith("Holding(")), None)
            if held:
                return (
                    f"PutDown({held})",
                    [f"Holding({held})"],
                    {f"OnTable({held})", f"Clear({held})", "ArmEmpty"},
                    {f"Holding({held})"},
                )

        raise ValueError(f"No STRIPS operator found for predicate: {predicate}")

    def solve(self) -> list[str]:
        stack: list = [list(self.goal)] + list(self.goal)

        while stack:
            top = stack.pop()

            if isinstance(top, list):
                unsatisfied = [g for g in top if g not in self.state]
                if unsatisfied:
                    stack.append(top)
                    stack.extend(unsatisfied)

            elif isinstance(top, tuple):
                action_name, add_set, del_set = top
                self.state.difference_update(del_set)
                self.state.update(add_set)
                self.plan.append(action_name)

            elif isinstance(top, str):
                if top not in self.state:
                    action_name, preconds, add_set, del_set = self.get_achieving_action(top)
                    stack.append((action_name, add_set, del_set))
                    stack.append(preconds)
                    stack.extend(preconds)

        return self.plan


if __name__ == "__main__":
    INITIAL = {
        "On(C,A)",
        "OnTable(A)",
        "OnTable(B)",
        "Clear(C)",
        "Clear(B)",
        "ArmEmpty",
    }

    GOAL = {
        "On(A,B)",
        "On(B,C)",
        "OnTable(C)",
    }

    print("=" * 70)
    print("STRIPS / GOAL-STACK PLANNING: BLOCKS WORLD")
    print("=" * 70)
    print(f"Initial State : {sorted(INITIAL)}")
    print(f"Goal State    : {sorted(GOAL)}")

    planner = BlocksWorldPlanner(INITIAL, GOAL)
    plan = planner.solve()

    print("\nGenerated STRIPS Action Plan:")
    for step, action in enumerate(plan, 1):
        print(f"  Step {step}: {action}")

    print(f"\nFinal State   : {sorted(planner.state)}")
    print(f"Goal Satisfied: {GOAL.issubset(planner.state)}")
