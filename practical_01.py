"""
Practical 1: Simple Reflex Agent & Goal-Based Agent in a Toy Environment
Aim: Set up Python environment for AI and implement a simple reflex and goal-based
     agent in a 2-room / 3-room Vacuum Cleaner toy environment.
"""

import platform
import sys


def check_environment() -> None:
    """Verify Python setup for AI experiments."""
    print("=" * 65)
    print("PYTHON AI ENVIRONMENT CHECK")
    print("=" * 65)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.system()} {platform.release()}")
    for pkg in ("numpy", "sklearn"):
        try:
            mod = __import__(pkg)
            print(f"Package '{pkg}' : Installed (v{getattr(mod, '__version__', 'unknown')})")
        except ImportError:
            print(f"Package '{pkg}' : Not installed (standard library mode active)")
    print()


class SimpleReflexVacuumAgent:
    """
    Simple Reflex Agent:
    Acts strictly on the current percept (location, status) using condition-action rules.
    """

    def act(self, location: str, status: str) -> str:
        if status == "Dirty":
            return "Suck"
        elif location == "Room_A":
            return "Move_Right"
        elif location == "Room_B":
            return "Move_Left"
        return "NoOp"


class GoalBasedVacuumAgent:
    """
    Goal-Based Agent:
    Maintains an internal model of the world and acts until the goal
    (all rooms are Clean) is achieved, then halts cleanly (NoOp).
    """

    def __init__(self, rooms: list[str]) -> None:
        self.model = {room: "Unknown" for room in rooms}

    def update_model(self, location: str, status: str) -> None:
        self.model[location] = status

    def goal_achieved(self) -> bool:
        return all(state == "Clean" for state in self.model.values())

    def act(self, location: str, status: str) -> str:
        self.update_model(location, status)
        if self.goal_achieved():
            return "NoOp (Goal Reached)"
        if status == "Dirty":
            self.model[location] = "Clean"
            return "Suck"
        for room, state in self.model.items():
            if state != "Clean" and room != location:
                return f"Move_To_{room}"
        return "NoOp"


def simulate_reflex_agent() -> None:
    print("=" * 65)
    print("1. SIMPLE REFLEX AGENT SIMULATION (4 Steps)")
    print("=" * 65)
    environment = {"Room_A": "Dirty", "Room_B": "Dirty"}
    agent_location = "Room_A"
    agent = SimpleReflexVacuumAgent()
    score = 0

    for step in range(1, 5):
        percept_status = environment[agent_location]
        action = agent.act(agent_location, percept_status)
        print(f"Step {step}: Location={agent_location:6s} | Percept={percept_status:5s} -> Action={action}")

        if action == "Suck":
            environment[agent_location] = "Clean"
            score += 10
        elif action == "Move_Right":
            agent_location = "Room_B"
            score -= 1
        elif action == "Move_Left":
            agent_location = "Room_A"
            score -= 1

    print(f"Final Environment State : {environment}")
    print(f"Performance Score       : {score}\n")


def simulate_goal_based_agent() -> None:
    print("=" * 65)
    print("2. GOAL-BASED AGENT SIMULATION (3-Room Environment)")
    print("=" * 65)
    rooms = ["Room_A", "Room_B", "Room_C"]
    environment = {"Room_A": "Clean", "Room_B": "Dirty", "Room_C": "Dirty"}
    agent_location = "Room_A"
    agent = GoalBasedVacuumAgent(rooms)
    score = 0

    for step in range(1, 7):
        percept_status = environment[agent_location]
        action = agent.act(agent_location, percept_status)
        print(
            f"Step {step}: Loc={agent_location:6s} | Percept={percept_status:5s} "
            f"| InternalModel={agent.model} -> Action={action}"
        )

        if action == "Suck":
            environment[agent_location] = "Clean"
            score += 10
        elif action.startswith("Move_To_"):
            agent_location = action.replace("Move_To_", "")
            score -= 1
        elif action.startswith("NoOp"):
            break

    print(f"Final Environment State : {environment}")
    print(f"Performance Score       : {score}")


if __name__ == "__main__":
    check_environment()
    simulate_reflex_agent()
    simulate_goal_based_agent()
