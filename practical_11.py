"""
Practical 11: Knowledge Representation & Logical Inference (Propositional & FOL)
Aim:
  Part 1: Propositional Logic Model-Checking Entailment (KB |= Query) via truth tables.
  Part 2: First-Order Logic (FOL) Unification & Forward-Chaining Deduction.
"""

from itertools import product


def implies(p: bool, q: bool) -> bool:
    return (not p) or q


def kb_entails_query() -> None:
    """
    Real-World Smart Home Security Scenario:
      P: MotionDetected
      Q: DoorForced
      R: AlarmTriggered
      S: OwnerHome
      T: PoliceNotified
    """
    symbols = ["P", "Q", "R", "S", "T"]
    models_where_kb_true = 0
    entails = True

    for values in product([False, True], repeat=len(symbols)):
        m = dict(zip(symbols, values))
        kb = (
            implies(m["P"] or m["Q"], m["R"])
            and implies(m["R"] and (not m["S"]), m["T"])
            and m["Q"]
            and (not m["S"])
        )
        if kb:
            models_where_kb_true += 1
            if not m["T"]:
                entails = False

    print("=" * 72)
    print("PART 1: PROPOSITIONAL LOGIC ENTAILMENT (TRUTH-TABLE MODEL CHECKING)")
    print("=" * 72)
    print("KB Rules:")
    print("  1. (MotionDetected OR DoorForced) -> AlarmTriggered")
    print("  2. (AlarmTriggered AND NOT OwnerHome) -> PoliceNotified")
    print("  3. Facts: DoorForced = True, OwnerHome = False")
    print(f"Satisfying Models Found : {models_where_kb_true}")
    print(f"Query 'PoliceNotified'  : Entailed by KB = {entails}")


def is_variable(term: str) -> bool:
    return term[0].islower()


def unify(term1: str, term2: str, theta: dict[str, str] | None = None) -> dict[str, str] | None:
    if theta is None:
        theta = {}
    if term1 == term2:
        return theta
    if is_variable(term1):
        return unify_var(term1, term2, theta)
    if is_variable(term2):
        return unify_var(term2, term1, theta)
    return None


def unify_var(var: str, val: str, theta: dict[str, str]) -> dict[str, str] | None:
    if var in theta:
        return unify(theta[var], val, theta)
    if val in theta:
        return unify(var, theta[val], theta)
    new_theta = dict(theta)
    new_theta[var] = val
    return new_theta


def fol_deduction_demo() -> None:
    facts = {
        ("Student", "Alice"),
        ("HighGPA", "Alice"),
        ("PublishedPaper", "Alice"),
        ("Student", "Bob"),
        ("HighGPA", "Bob"),
    }

    rules = [
        ([("Student", "x"), ("HighGPA", "x"), ("PublishedPaper", "x")], ("HonorsCandidate", "x")),
        ([("HonorsCandidate", "x")], ("EligibleForFellowship", "x")),
    ]

    print("\n" + "=" * 72)
    print("PART 2: FIRST-ORDER LOGIC (FOL) UNIFICATION & FORWARD DEDUCTION")
    print("=" * 72)
    print(f"Initial Ground Facts: {sorted(facts)}\n")

    added = True
    while added:
        added = False
        for premises, conclusion in rules:
            constants = {arg for _, arg in facts if not is_variable(arg)}
            for const in sorted(constants):
                theta = unify("x", const)
                instantiated_premises = {(pred, theta[arg]) for pred, arg in premises}
                if instantiated_premises.issubset(facts):
                    new_fact = (conclusion[0], theta[conclusion[1]])
                    if new_fact not in facts:
                        facts.add(new_fact)
                        added = True
                        print(f"Unified x/{const} -> Derived: {new_fact[0]}({new_fact[1]})")

    eligible = [arg for pred, arg in facts if pred == "EligibleForFellowship"]
    print(f"\nFinal Derived Fellowship Candidates: {eligible}")


if __name__ == "__main__":
    kb_entails_query()
    fol_deduction_demo()
